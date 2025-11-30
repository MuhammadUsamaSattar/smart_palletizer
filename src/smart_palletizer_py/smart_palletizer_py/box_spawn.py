import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, TransformStamped, TransformListener
from sensor_msgs.msg import JointState

class BoxSpawn(Node):
    def __init__(self):
        super().__init__()

        self.transform_broadcaster_ = TransformBroadcaster()

