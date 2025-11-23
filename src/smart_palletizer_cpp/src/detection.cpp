#include <array>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "std_msgs/msg/header.hpp"

struct CombinedImage
{
    sensor_msgs::msg::Image::SharedPtr rgb;
    sensor_msgs::msg::Image::SharedPtr depth;
};


class Detection : public rclcpp::Node
{
public:
    Detection() : rclcpp::Node("detection")
    {
        this->declare_parameter("thresh", 255);
        this->declare_parameter("encoding", 8);
        this->declare_parameter("size", 41);
        this->declare_parameter("const", 0.0);
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/thresholded_img",
            10
        );
        _rgb_img_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 
            10, 
            std::bind(&Detection::add_rgb_data, this, std::placeholders::_1)
        );
        _depth_img_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/aligned_depth_to_color/image_raw", 
            10, 
            std::bind(&Detection::add_depth_data, this, std::placeholders::_1)
        );

        timer_ = this->create_wall_timer(std::chrono_literals::operator""ms(100), std::bind(&Detection::detect_contours, this));
    }

    void add_rgb_data(sensor_msgs::msg::Image::SharedPtr msg)
    {
        combined_img_.rgb = msg;
    }

    void add_depth_data(sensor_msgs::msg::Image::SharedPtr msg)
    {
        combined_img_.depth = msg;
    }

    void detect_contours()
    {
        if (combined_img_.rgb and combined_img_.depth)
        {
            auto img_rgb = cv_bridge::toCvCopy(*combined_img_.rgb, "bgr8");
            auto img_depth = cv_bridge::toCvCopy(*combined_img_.depth, "16UC1");
            cv::Mat img_gray;
            cv::Mat img_masked;
            cv::Mat img_processed;
            cv::Mat img_thresholded;
            cv::Mat mask;

            cv::threshold(img_depth->image, mask, 1925, 255, cv::THRESH_BINARY_INV);
            mask.convertTo(mask, CV_8UC1);
            cv::bitwise_and(img_rgb->image, img_rgb->image, img_masked, mask);

            cv::cvtColor(img_masked, img_gray, cv::COLOR_BGR2GRAY);
            //cv::threshold(img_gray, img_thresholded, this->get_parameter("thresh").as_int(), 255, this->get_parameter("encoding").as_int());
            cv::adaptiveThreshold(img_gray, img_thresholded, this->get_parameter("thresh").as_int(), cv::BORDER_REPLICATE, cv::THRESH_BINARY, this->get_parameter("size").as_int(), this->get_parameter("const").as_double());
            auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", img_thresholded).toImageMsg();
            publisher_->publish(*img_msg);
            //cv::imshow("Image", img_thresh);
            //cv::waitKey(1);

            combined_img_.rgb = nullptr;
            combined_img_.depth = nullptr;
        }
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _rgb_img_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _depth_img_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    CombinedImage combined_img_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Detection>());\
    rclcpp::shutdown();

    return 0;
}