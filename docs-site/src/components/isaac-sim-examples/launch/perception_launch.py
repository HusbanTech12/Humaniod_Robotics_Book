#!/usr/bin/env python3
# perception_launch.py

"""
Isaac ROS Perception Pipeline Launch File
This launch file sets up the Isaac ROS perception pipeline for humanoid robot
including object detection, sensor fusion, and visual processing nodes.
"""

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for Isaac ROS perception pipeline."""

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for all perception nodes'
    )

    # Get launch configuration
    namespace = LaunchConfiguration('namespace')

    # Define perception nodes
    perception_nodes = ComposableNodeContainer(
        name='perception_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # Image rectification node
            ComposableNode(
                package='isaac_ros_image_proc',
                plugin='isaac_ros::ImageProc::RectifyNode',
                name='image_rectifier',
                parameters=[{
                    'output_width': 640,
                    'output_height': 480,
                }],
                remappings=[
                    ('image', '/camera/rgb/image_raw'),
                    ('camera_info', '/camera/rgb/camera_info'),
                    ('image_rect', '/camera/rgb/image_rect_color'),
                ],
            ),

            # Depth image processing node
            ComposableNode(
                package='isaac_ros_depth_image_proc',
                plugin='nvidia::isaac_ros::depth_image_proc::ConvertMetricNode',
                name='convert_metric_node',
                remappings=[
                    ('image_raw', '/camera/depth/image_raw'),
                    ('image', '/camera/depth/image_metric'),
                ],
            ),

            # Point cloud creation node
            ComposableNode(
                package='isaac_ros_pointcloud_utils',
                plugin='nvidia::isaac_ros::pointcloud_utils::PointCloudPclNode',
                name='pointcloud_pcl_node',
                remappings=[
                    ('image', '/camera/rgb/image_rect_color'),
                    ('depth', '/camera/depth/image_metric'),
                    ('points', '/camera/depth/color/points'),
                ],
            ),

            # Object detection node (placeholder - would use actual Isaac ROS detection)
            ComposableNode(
                package='isaac_ros_detectnet',
                plugin='nvidia::isaac_ros::detectnet::DetectNetNode',
                name='detectnet_node',
                parameters=[{
                    'model_name': 'ssd_mobilenet_v2_coco',
                    'input_topic': '/camera/rgb/image_rect_color',
                    'publish_topic': '/detectnet/detections',
                    'confidence_threshold': 0.7,
                }],
                remappings=[
                    ('image_input', '/camera/rgb/image_rect_color'),
                    ('detections_output', '/detectnet/detections'),
                ],
            ),
        ],
        output='screen',
    )

    return launch.LaunchDescription([
        namespace_arg,
        perception_nodes,
    ])


if __name__ == '__main__':
    """Main function to run the launch file."""
    generate_launch_description()