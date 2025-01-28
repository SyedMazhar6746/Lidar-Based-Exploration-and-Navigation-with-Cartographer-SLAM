import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    # Path to turtlebot3_cartographer package
    cartographer_launch_file = os.path.join(
        get_package_share_directory('turtlebot3_cartographer'),
        'launch',
        'cartographer.launch.py'
    )

    return LaunchDescription([

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch', 'turtlebot3_dqn_stage4.launch.py')]),
        ),
    
    # Include the Cartographer SLAM launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(cartographer_launch_file),
            launch_arguments={
                'use_sim_time': 'True'
            }.items(),
        ),

    ])
