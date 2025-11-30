from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    main_node_arg = DeclareLaunchArgument(
        "main_node",
        description="Launches the rviz2 window for this task if it is the main node",
        default_value="True",
    )
    bag_path_arg = DeclareLaunchArgument(
        "bag_path", description="Relative path of the bag folder containing data"
    )
    run_bag_arg = DeclareLaunchArgument(
        "run_bag", description="Launch the bag or not", default_value="True"
    )
    downscaling_factor_arg = DeclareLaunchArgument(
        "downscaling_factor",
        description="Downscaling factor for rgb and depth image",
        default_value="2",
    )
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        description="Namespace of the node",
        default_value="/camera_filtered",
    )

    return LaunchDescription(
        [
            # Declare launch arguments
            main_node_arg,
            bag_path_arg,
            run_bag_arg,
            downscaling_factor_arg,
            namespace_arg,
            # Run post processing node's rviz2 file if this is main launch file
            Node(
                condition=IfCondition(LaunchConfiguration("main_node")),
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=[
                    (
                        "-d",
                        PathJoinSubstitution(
                            [
                                FindPackageShare("smart_palletizer_py"),
                                "rviz2",
                                "post_processing.rviz",
                            ]
                        ),
                    )
                ],
                parameters=[
                    {
                        "use_sim_time": True,
                    }
                ],
            ),
            # Run ros2 bag using dataset file
            ExecuteProcess(
                condition=IfCondition(LaunchConfiguration("run_bag")),
                cmd=[
                    "ros2",
                    "bag",
                    "play",
                    "-l",
                    "--clock",
                    "30",
                    LaunchConfiguration("bag_path"),
                ],
                output="screen",
            ),
            # Run post processing node
            Node(
                package="smart_palletizer_py",
                executable="post_processing",
                name="post_processing_node",
                namespace=LaunchConfiguration("namespace"),
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "downscaling_factor": LaunchConfiguration("downscaling_factor"),
                    }
                ],
            ),
        ]
    )
