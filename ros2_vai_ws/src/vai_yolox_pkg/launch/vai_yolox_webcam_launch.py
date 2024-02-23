import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config=os.path.join(
        get_package_share_directory('vai_yolox_pkg'),
        'config',
        'vai_yolox_param.yaml'
        )

    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
        ),
        Node(
            package='vai_yolox_pkg',
            executable='yolox_node',
            output='screen',
            emulate_tty=True,
            parameters=[config]
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
        ),
    ])
