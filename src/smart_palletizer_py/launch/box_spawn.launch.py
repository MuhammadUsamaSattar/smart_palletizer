from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
    viz_downscaling_factor_arg = DeclareLaunchArgument(
        "viz_downscaling_factor",
        description="Downscaling factor for visualization of rgb, depth image and depth cloud",
        default_value="2",
    )

    return LaunchDescription(
        [
            # Declare launch arguments
            main_node_arg,
            bag_path_arg,
            viz_downscaling_factor_arg,
            # Run rviz2
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
                                "box_spawn.rviz",
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
            # Run pose detection launch file
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("smart_palletizer_py"),
                            "launch",
                            "pose_detection.launch.py",
                        ]
                    ),
                ),
                launch_arguments={
                    "main_node": "False",
                    "bag_path": LaunchConfiguration("bag_path"),
                }.items(),
            ),
            # Run pose detection node
            Node(
                package="smart_palletizer_py",
                executable="box_spawn",
                name="box_spawn_node",
                parameters=[{"use_sim_time": True}],
            ),
        ]
    )
