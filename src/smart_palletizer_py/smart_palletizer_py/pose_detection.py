import math

from geometry_msgs.msg import TransformStamped
import image_geometry
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from smart_palletizer_interfaces.msg import DetectedBoxes, BoxInfo, BoxInfoArray
from smart_palletizer_py import utils
from tf2_ros import TransformBroadcaster


class PoseDetection(Node):
    """Detect pose of boxes."""

    def __init__(self) -> None:
        super().__init__("pose_detection")

        self.camera = image_geometry.PinholeCameraModel()
        self.transform_broadcaster = TransformBroadcaster(self)
        self.detected_boxes = None

        # Subscription for box information
        self.detected_boxes_subscription_ = self.create_subscription(
            DetectedBoxes, "/detected_boxes", self.add_detected_boxes, 10
        )
        # Subscription for filtered RGB image
        self.camera_info_subscription_ = self.create_subscription(
            CameraInfo, "/camera_filtered/color/camera_info", self.add_camera_info, 10
        )

        self.box_info_array_publisher_ = self.create_publisher(
            BoxInfoArray, "/box_info_array", 10
        )

        self.create_timer(1 / 30, self.detect_pose)

    def add_detected_boxes(self, msg: DetectedBoxes) -> None:
        """Add box information to instance attribute.

        Args:
            msg (DetectedBoxes): DetectedBoxes message.
        """
        self.detected_boxes = msg

    def add_camera_info(self, msg: CameraInfo) -> None:
        """Add camera information to instance attribute.

        Args:
            msg (CameraInfo): CameraInfo message.
        """
        self.camera.from_camera_info(msg)

    def detect_pose(self) -> None:
        """Detect poses of boxes."""
        # If instance attributes have not been assigned, then skip
        if self.detected_boxes is None or np.isnan(self.camera.P).any():
            return

        number_boxes = 1
        header = self.detected_boxes.header

        box_info_array_msg = BoxInfoArray()
        box_info_array_msg.header = header
        for detected_box in self.detected_boxes.data:
            # Create message
            box_tf = TransformStamped()

            box_tf.header = header
            # box_tf.child_frame_id = detected_box.classification + "_" + str(number_boxes)
            box_tf.child_frame_id = detected_box.id
            number_boxes += 1

            # Get real world coordinates of point
            x, y, z = utils.get_XYZ_from_Pixels(
                self.camera, detected_box.x, detected_box.y, detected_box.depth
            )
            box_tf.transform.translation.x = float(x)
            box_tf.transform.translation.y = float(y)
            box_tf.transform.translation.z = float(z)

            # Calculate the angle of longest side from x-axis and convert rotation to quaternion
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

            box_info_msg = BoxInfo()
            box_info_msg.id = box_tf.child_frame_id
            box_info_msg.pose.position.x = box_tf.transform.translation.x
            box_info_msg.pose.position.y = box_tf.transform.translation.y
            box_info_msg.pose.position.z = box_tf.transform.translation.z
            box_info_msg.pose.orientation = box_tf.transform.rotation
            box_info_msg.classification = detected_box.classification
            box_info_array_msg.infos.append(box_info_msg)

            self.transform_broadcaster.sendTransform(box_tf)

        self.box_info_array_publisher_.publish(box_info_array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
