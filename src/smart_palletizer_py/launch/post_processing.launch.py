from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

def generate_launch_description():
    bag_path_arg = DeclareLaunchArgument(
        'bag_path', 
        description='Relative path of the bag folder containing data'
        )
    downscaling_factor_arg = DeclareLaunchArgument(
        'downscaling_factor', 
        description='Dwnscaling factor for rgb and depth image', 
        default_value='2'
        )

    return LaunchDescription([
        bag_path_arg,
        downscaling_factor_arg,
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '-l', '--clock', '30', LaunchConfiguration('bag_path')],
            output='screen',
            shell=True,
        ),
        Node(
            package='smart_palletizer_py',
            executable='post_processing',
            name='post_processing_node',
            output='screen',
            parameters=[{
                'use_sim_time' : True,
                'downscaling_factor' : LaunchConfiguration('downscaling_factor')
            }]
        ),
    ])
    