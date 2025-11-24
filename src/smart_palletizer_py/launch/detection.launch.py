from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    bag_path_arg = DeclareLaunchArgument(
        'bag_path', 
        description='Relative path of the bag folder containing data'
        )
    downscaling_factor_arg = DeclareLaunchArgument(
        'downscaling_factor', 
        description='Dwnscaling factor for rgb and depth image', 
        default_value='1'
        )

    return LaunchDescription([
        bag_path_arg,
        downscaling_factor_arg,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('smart_palletizer_py'),
                    'launch',
                    'post_processing.launch.py'
                ]),
            ),
            launch_arguments={
                'bag_path': LaunchConfiguration('bag_path'),
                'downscaling_factor': LaunchConfiguration('downscaling_factor'),
            }.items()
        ),
        Node(
            package='smart_palletizer_py',
            executable='detection',
            name='detecton_node',
            parameters=[{
                'use_sim_time' : True
            }]
        ),
    ])
    