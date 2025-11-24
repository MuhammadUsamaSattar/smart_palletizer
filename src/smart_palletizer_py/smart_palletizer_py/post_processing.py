import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import cv_bridge
from smart_palletizer_py import utils
import cv2
from copy import deepcopy

class PostProcessing(Node):
    def __init__(self):
        super().__init__("post_processing")
        self.declare_parameter("downscaling_factor", 2, 
                               ParameterDescriptor(description="Factor by which to downsacle the rgb and depth image"))

        self.filtered_depth_publisher_ = self.create_publisher(
            Image,
            "camera_filtered/aligned_depth_to_color/image_filtered",
            10,
        )
        self.filtered_rgb_publisher_ = self.create_publisher(
            Image,
            "camera_filtered/color/image_filtered",
            10,
        )
        self.filtered_depth_camera_info_publisher_ = self.create_publisher(
            CameraInfo,
            "camera_filtered/aligned_depth_to_color/camera_info",
            10,
        )
        self.filtered_rgb_camera_info_publisher_ = self.create_publisher(
            CameraInfo,
            "camera_filtered/color/camera_info",
            10,
        )

        self.img_depth_subscriber_ = self.create_subscription(
            Image,
            "camera/aligned_depth_to_color/image_raw",
            self.add_img_depth,
            10,
        )
        self.img_rgb_subscriber_ = self.create_subscription(
            Image,
            "camera/color/image_raw",
            self.add_img_rgb,
            10,
        )
        self.camera_info_subscriber_ = self.create_subscription(
            CameraInfo,
            "camera/color/camera_info",
            self.add_camera_info,
            10,
        )

        self.create_timer(0.1, self.process_imgs)

        self.img_depth = None
        self.img_rgb = None
        self.camera_info = None
        self.prev_img_depth = None

    def add_img_depth(self, msg):
        self.img_depth = msg

    def add_img_rgb(self, msg):
        self.img_rgb = msg

    def add_camera_info(self, msg):
        self.camera_info = msg

    def process_imgs(self):
        if self.img_depth is None or self.img_rgb is None or self.camera_info is None:
            return
        
        bridge = cv_bridge.CvBridge()
        img_depth = bridge.imgmsg_to_cv2(self.img_depth, "16UC1")
        img_rgb = bridge.imgmsg_to_cv2(self.img_rgb, "rgb8")
        

        down_scaling_factor = self.get_parameter("downscaling_factor").get_parameter_value().integer_value
        if down_scaling_factor != 1:
            img_depth = utils.subsample(img_depth, factor=down_scaling_factor, method='auto')
            img_rgb = utils.subsample(img_rgb, factor=down_scaling_factor, method='auto')

        img_depth = utils.holePatching(img_depth, 3, 'max', 'L-2D-Excl', max_iter=8)
        img_depth = cv2.medianBlur(img_depth, 5)
        self.prev_img_depth = img_depth = utils.time_filter(img_depth, self.prev_img_depth, 0.2, (5)*(2**8)).copy()
        
        header = Header()
        header.frame_id = "camera_color_optical_frame"
        header.stamp = self.get_clock().now().to_msg()

        msg = bridge.cv2_to_imgmsg(img_depth, "16UC1", header)
        self.filtered_depth_publisher_.publish(msg)

        msg = bridge.cv2_to_imgmsg(img_rgb, "rgb8", header)
        self.filtered_rgb_publisher_.publish(msg)

        camera_info_msg = deepcopy(self.camera_info)
        camera_info_msg.header = header
        camera_info_msg.height = camera_info_msg.height // down_scaling_factor
        camera_info_msg.width = camera_info_msg.width // down_scaling_factor
        camera_info_msg.k = camera_info_msg.k / down_scaling_factor
        camera_info_msg.p = camera_info_msg.p / down_scaling_factor
        self.filtered_depth_camera_info_publisher_.publish(camera_info_msg)
        self.filtered_rgb_camera_info_publisher_.publish(camera_info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PostProcessing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()