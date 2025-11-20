#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.hpp"

class Detection : public rclcpp::Node
{
public:
    Detection() : rclcpp::Node("detection")
    {
        this->declare_parameter("thresh", 100);
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 
            10, 
            std::bind(&Detection::detect_contours, this, std::placeholders::_1)
        );
    }

    void detect_contours(sensor_msgs::msg::Image::SharedPtr msg)
    {
        auto img = cv_bridge::toCvCopy(*msg, "bgr8");
        cv::Mat img_gray;
        cv::Mat img_thresh;

        cv::cvtColor(img->image, img_gray, cv::COLOR_BGR2GRAY);
        cv::threshold(img_gray, img_thresh, this->get_parameter("thresh").as_int(), 255, cv::THRESH_BINARY);
        cv::imshow("Image", img_thresh);
        cv::waitKey(1);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Detection>());\
    rclcpp::shutdown();

    return 0;
}