#!/usr/bin/env python3
# vslam_launch.py

"""
Isaac ROS Visual SLAM Launch Configuration
This launch file configures the Isaac ROS Visual SLAM pipeline for the humanoid robot
including camera setup, visual SLAM processing, and mapping components.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for Isaac ROS Visual SLAM pipeline."""

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='humanoid_robot',
        description='Namespace for VSLAM nodes'
    )

    enable_stereo_arg = DeclareLaunchArgument(
        'enable_stereo',
        default_value='false',
        description='Enable stereo visual SLAM (true) or monocular (false)'
    )

    enable_mapping_arg = DeclareLaunchArgument(
        'enable_mapping',
        default_value='true',
        description='Enable map building and occupancy grid generation'
    )

    enable_localization_arg = DeclareLaunchArgument(
        'enable_localization',
        default_value='true',
        description='Enable localization against existing map'
    )

    enable_loop_closure_arg = DeclareLaunchArgument(
        'enable_loop_closure',
        default_value='true',
        description='Enable loop closure detection and correction'
    )

    # Get launch configurations
    namespace = LaunchConfiguration('namespace')
    enable_stereo = LaunchConfiguration('enable_stereo')
    enable_mapping = LaunchConfiguration('enable_mapping')
    enable_localization = LaunchConfiguration('enable_localization')
    enable_loop_closure = LaunchConfiguration('enable_loop_closure')

    # Get package directories
    isaac_ros_workspace = get_package_share_directory('isaac_ros_workspace')

    # Define VSLAM container with all VSLAM nodes
    vslam_container = ComposableNodeContainer(
        name='vslam_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # Isaac ROS Visual SLAM node
            ComposableNode(
                package='isaac_ros_visual_slam',
                plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
                name='visual_slam_node',
                parameters=[
                    os.path.join(
                        get_package_share_directory('isaac_ros_visual_slam'),
                        'config', 'slam_toolbox_config.yaml'
                    ),
                    {
                        'use_sim_time': True,
                        'map_frame': 'map',
                        'odom_frame': 'odom',
                        'base_frame': 'base_link',
                        'enable_occupancy_map': True,
                        'enable_mapper': enable_mapping,
                        'enable_localization': enable_localization,
                        'enable_loop_closure': enable_loop_closure,
                        'enable_ground_truth': False,
                        'max_num_landmarks': 10000,
                        'min_num_landmarks': 100,
                        'min_tracked_landmarks': 10,
                        'min_distance_between_landmarks': 0.5,
                        'min_visual_parallax_degrees': 10.0,
                        'enable_slam_visualization': True,
                        'enable_pointcloud_output': True,
                        'calibration_resolution': 640.0,
                        'num_matchers': 2,
                        'enable_deterministic_outcomes': False,
                        'stereo_camera': enable_stereo,
                        'enable_rectification': True,
                        'image_input_width': 640,
                        'image_input_height': 480,
                        'publish_tf': True,
                        'publish_map_odom_transform': True,
                        'publish_tracked_map_frames': True,
                        'min_z': 0.1,
                        'max_z': 10.0,
                        'min_disparity': 0.1,
                        'max_disparity': 64.0,
                    }
                ],
                remappings=[
                    ('visual_slam/imu', 'imu/data'),
                    ('visual_slam/left/camera_info', 'camera/rgb/camera_info'),
                    ('visual_slam/left/image', 'camera/rgb/image_rect_color'),
                    ('visual_slam/right/camera_info', 'camera/rgb/camera_info'),  # For stereo
                    ('visual_slam/right/image', 'camera/rgb/image_rect_color'),   # For stereo
                    ('visual_slam/tracked_map_frames', 'visual_slam/tracked_map_frames'),
                    ('visual_slam/visual_slam_graph', 'visual_slam/visual_slam_graph'),
                    ('visual_slam/optimized_graph', 'visual_slam/optimized_graph'),
                    ('visual_slam/map_odom', 'visual_slam/map_odom'),
                    ('visual_slam/landmarks', 'visual_slam/landmarks'),
                    ('visual_slam/pointcloud', 'visual_slam/pointcloud'),
                    ('visual_slam/occupancy_grid', 'visual_slam/occupancy_grid'),
                ],
                # Only enable if stereo is enabled or we're using monocular VSLAM
            ),

            # Feature extraction node for visual SLAM
            ComposableNode(
                package='isaac_ros_visual_slam',
                plugin='nvidia::isaac_ros::visual_slam::FeatureExtractorNode',
                name='feature_extractor',
                parameters=[{
                    'max_features': 1000,
                    'min_distance_between_features': 10.0,
                    'quality_level': 0.01,
                    'use_sim_time': True,
                }],
                remappings=[
                    ('image', 'camera/rgb/image_rect_color'),
                    ('camera_info', 'camera/rgb/camera_info'),
                    ('features', 'visual_slam/features'),
                ],
            ),

            # Feature matching node for visual SLAM
            ComposableNode(
                package='isaac_ros_visual_slam',
                plugin='nvidia::isaac_ros::visual_slam::FeatureMatchingNode',
                name='feature_matcher',
                parameters=[{
                    'max_num_matches': 100,
                    'max_ratio_threshold': 0.8,
                    'min_matches': 10,
                    'use_sim_time': True,
                }],
                remappings=[
                    ('features1', 'visual_slam/features'),
                    ('features2', 'visual_slam/features'),
                    ('matches', 'visual_slam/matches'),
                ],
            ),

            # IMU preprocessor for visual SLAM
            ComposableNode(
                package='isaac_ros_imu_pipeline',
                plugin='nvidia::isaac_ros::imu_pipeline::IMUPreprocessorNode',
                name='imu_preprocessor',
                parameters=[{
                    'use_sim_time': True,
                    'linear_acceleration_stddev': 0.017,
                    'angular_velocity_stddev': 0.0087,
                    'orientation_stddev': 0.01,
                }],
                remappings=[
                    ('imu', 'imu/data'),
                    ('imu_processed', 'visual_slam/imu_processed'),
                ],
            ),
        ],
        output='screen',
    )

    # Additional nodes that run outside containers
    # Occupancy grid mapper node
    occupancy_grid_node = Node(
        package='isaac_ros_occupancy_grid',
        executable='isaac_ros_occupancy_grid_node',
        name='occupancy_grid_mapper',
        namespace=namespace,
        parameters=[
            {
                'use_sim_time': True,
                'resolution': 0.05,  # 5cm resolution
                'width': 200,       # 10m x 10m grid
                'height': 200,      # 10m x 10m grid
                'origin_x': -10.0,
                'origin_y': -10.0,
                'free_threshold': 0.2,
                'occupied_threshold': 0.65,
                'unknown_threshold': 0.5,
            }
        ],
        remappings=[
            ('pointcloud', 'visual_slam/pointcloud'),
            ('map', 'visual_slam/occupancy_grid'),
        ],
    )

    # Map to odom transform broadcaster
    map_odom_broadcaster = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_broadcaster',
        namespace=namespace,
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--qx', '0.0', '--qy', '0.0', '--qz', '0.0', '--qw', '1.0',
            '--frame-id', 'map',
            '--child-frame-id', 'odom'
        ]
    )

    # RViz2 for visualization
    rviz_config_file = os.path.join(
        isaac_ros_workspace,
        'rviz',
        'vslam_config.rviz'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('enable_viz', default='true')),
    )

    # Return launch description
    return LaunchDescription([
        namespace_arg,
        enable_stereo_arg,
        enable_mapping_arg,
        enable_localization_arg,
        enable_loop_closure_arg,
        vslam_container,
        occupancy_grid_node,
        map_odom_broadcaster,
    ])


def main():
    """Main function to run the launch file."""
    ld = generate_launch_description()
    print("Isaac ROS Visual SLAM Pipeline Launch Configuration Created")
    print("This configuration includes:")
    print("- Visual SLAM processing with landmark tracking")
    print("- Occupancy grid mapping from point cloud data")
    print("- IMU integration for improved pose estimation")
    print("- Feature extraction and matching for loop closure")
    print("- Map to odom transform broadcasting")
    return ld


if __name__ == '__main__':
    main()