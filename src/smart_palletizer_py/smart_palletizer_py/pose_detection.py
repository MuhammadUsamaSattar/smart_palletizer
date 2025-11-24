import math

from geometry_msgs.msg import TransformStamped
import image_geometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from smart_palletizer_interfaces.msg import DetectedBoxes
from smart_palletizer_py import utils
from tf2_ros import TransformBroadcaster


class PoseDetection(Node):
    def __init__(self):
        super().__init__("pose_detection")
        self.camera = image_geometry.PinholeCameraModel()
        self.transform_broadcaster = TransformBroadcaster(self)

        self.detected_boxes_subscription_ = self.create_subscription(
            DetectedBoxes, "/detected_boxes", self.add_detected_boxes, 10
        )
        self.camera_info_subscription_ = self.create_subscription(
            CameraInfo, "/camera_filtered/color/camera_info", self.add_camera_info, 10
        )

        self.create_timer(1 / 30, self.detect_pose)

        self.detected_boxes = None

    def add_detected_boxes(self, msg):
        self.detected_boxes = msg

    def add_camera_info(self, msg):
        self.camera.from_camera_info(msg)

    def detect_pose(self):
        if self.detected_boxes is None or self.camera.tf_frame is None:
            return

        number_boxes = 1
        for detected_box in self.detected_boxes.data:

            x, y, z = self.camera.project_pixel_to_3d_ray(
                (detected_box.x, detected_box.y)
            )
            factor = detected_box.depth / (1000 * z)
            x, y, z = x * factor, y * factor, z * factor

            box_tf = TransformStamped()

            box_tf.header.stamp = self.get_clock().now().to_msg()
            box_tf.header.frame_id = "camera_color_optical_frame"
            box_tf.child_frame_id = "box_" + str(number_boxes)
            number_boxes += 1

            box_tf.transform.translation.x = float(x)
            box_tf.transform.translation.y = float(y)
            box_tf.transform.translation.z = float(z)
            theta = math.atan2(
                (
                    detected_box.longest_side_coords[0].y
                    - detected_box.longest_side_coords[1].y
                ),
                (
                    detected_box.longest_side_coords[0].x
                    - detected_box.longest_side_coords[1].x
                ),
            )

            q = utils.quaternion_from_euler(0, 0, theta)
            box_tf.transform.rotation.x = float(q[0])
            box_tf.transform.rotation.y = float(q[1])
            box_tf.transform.rotation.z = float(q[2])
            box_tf.transform.rotation.w = float(q[3])

            self.transform_broadcaster.sendTransform(box_tf)


def main(args=None):
    rclpy.init(args=args)
    node = PoseDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
