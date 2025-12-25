#!/usr/bin/env python3
# perception_launch_config.py

"""
Isaac ROS Perception Launch Configuration
This launch file configures the Isaac ROS perception pipeline for the humanoid robot
including object detection, sensor fusion, and visual SLAM components.
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
    """Generate launch description for Isaac ROS perception pipeline."""

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='humanoid_robot',
        description='Namespace for perception nodes'
    )

    enable_rectification_arg = DeclareLaunchArgument(
        'enable_rectification',
        default_value='true',
        description='Enable camera image rectification'
    )

    enable_object_detection_arg = DeclareLaunchArgument(
        'enable_object_detection',
        default_value='true',
        description='Enable object detection pipeline'
    )

    enable_sensor_fusion_arg = DeclareLaunchArgument(
        'enable_sensor_fusion',
        default_value='true',
        description='Enable sensor fusion pipeline'
    )

    enable_visual_slam_arg = DeclareLaunchArgument(
        'enable_visual_slam',
        default_value='true',
        description='Enable visual SLAM pipeline'
    )

    # Get launch configurations
    namespace = LaunchConfiguration('namespace')
    enable_rectification = LaunchConfiguration('enable_rectification')
    enable_object_detection = LaunchConfiguration('enable_object_detection')
    enable_sensor_fusion = LaunchConfiguration('enable_sensor_fusion')
    enable_visual_slam = LaunchConfiguration('enable_visual_slam')

    # Get package directories
    isaac_ros_workspace = get_package_share_directory('isaac_ros_workspace')

    # Define perception container with all perception nodes
    perception_container = ComposableNodeContainer(
        name='perception_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # Image Rectification Node
            ComposableNode(
                package='isaac_ros_image_proc',
                plugin='isaac_ros::ImageProc::RectifyNode',
                name='image_rectifier',
                parameters=[{
                    'output_width': 640,
                    'output_height': 480,
                    'scale_to_fit_image': True,
                }],
                remappings=[
                    ('image', 'camera/rgb/image_raw'),
                    ('camera_info', 'camera/rgb/camera_info'),
                    ('image_rect', 'camera/rgb/image_rect_color'),
                ],
                condition=IfCondition(enable_rectification)
            ),

            # Depth Image Processing Node
            ComposableNode(
                package='isaac_ros_depth_image_proc',
                plugin='nvidia::isaac_ros::depth_image_proc::ConvertMetricNode',
                name='convert_metric_node',
                remappings=[
                    ('image_raw', 'camera/depth/image_raw'),
                    ('image', 'camera/depth/image_metric'),
                ],
                condition=IfCondition(enable_rectification)
            ),

            # Point Cloud Creation Node
            ComposableNode(
                package='isaac_ros_pointcloud_utils',
                plugin='nvidia::isaac_ros::pointcloud_utils::PointCloudPclNode',
                name='pointcloud_pcl_node',
                parameters=[{
                    'use_color': True,
                    'fill_nan': False,
                    'min_height': 0.0,
                    'max_height': 3.0,
                }],
                remappings=[
                    ('image', 'camera/rgb/image_rect_color'),
                    ('depth', 'camera/depth/image_metric'),
                    ('points', 'camera/depth/color/points'),
                ],
                condition=IfCondition(enable_sensor_fusion)
            ),

            # Object Detection Node (DetectNet)
            ComposableNode(
                package='isaac_ros_detectnet',
                plugin='nvidia::isaac_ros::detectnet::DetectNetNode',
                name='detectnet_node',
                parameters=[{
                    'model_name': 'ssd_mobilenet_v2_coco',
                    'input_topic': 'camera/rgb/image_rect_color',
                    'publish_topic': 'detectnet/detections',
                    'confidence_threshold': 0.7,
                    'max_objects': 10,
                }],
                remappings=[
                    ('image_input', 'camera/rgb/image_rect_color'),
                    ('detections_output', 'detectnet/detections'),
                    ('camera_info_input', 'camera/rgb/camera_info'),
                ],
                condition=IfCondition(enable_object_detection)
            ),

            # Stereo Disparity Node (for depth from stereo)
            ComposableNode(
                package='isaac_ros_stereo_image_proc',
                plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
                name='disparity_node',
                parameters=[{
                    'min_disparity': 0.0,
                    'max_disparity': 64.0,
                    'num_disparities': 64,
                    'stereo_algorithm': 1,  # 0 for BM, 1 for SGBM
                    'kernel_size': 3,
                    'disp_mode': 0,
                    'speckle_size': 100,
                    'speckle_range': 4,
                    'disp_12_max_diff': 1,
                    'pre_filter_cap': 63,
                    'uniqueness_ratio': 15,
                    'p1': 200,
                    'p2': 400,
                    'full_dp': False,
                }],
                remappings=[
                    ('left/image_rect', 'camera/rgb/left/image_rect_color'),
                    ('right/image_rect', 'camera/rgb/right/image_rect_color'),
                    ('left/camera_info', 'camera/rgb/left/camera_info'),
                    ('right/camera_info', 'camera/rgb/right/camera_info'),
                    ('disparity', 'stereo/disparity'),
                ],
                condition=IfCondition(enable_sensor_fusion)
            ),

            # Spatial Detection Node (combines detection with depth)
            ComposableNode(
                package='isaac_ros_detectnet',
                plugin='nvidia::isaac_ros::detectnet::SpatialDetectionNode',
                name='spatial_detection_node',
                parameters=[{
                    'input_topic': 'detectnet/detections',
                    'depth_image_topic': 'camera/depth/image_rect_raw',
                    'camera_info_topic': 'camera/rgb/camera_info',
                    'max_object_points': 500,
                    'depth_unit': 'meters',
                    'output_topic': 'detectnet/spatial_detections',
                    'mask_image_topic': 'detectnet/mask_image',
                    'mask_input_topic': 'detectnet/mask_input',
                }],
                remappings=[
                    ('detections', 'detectnet/detections'),
                    ('depth_image', 'camera/depth/image_rect_raw'),
                    ('camera_info', 'camera/rgb/camera_info'),
                    ('spatial_detections', 'detectnet/spatial_detections'),
                ],
                condition=IfCondition(enable_object_detection)
            ),

            # Isaac ROS Apriltag Node
            ComposableNode(
                package='isaac_ros_apriltag',
                plugin='nvidia::isaac_ros::apriltag::AprilTagNode',
                name='apriltag',
                parameters=[{
                    'family': 'tag36h11',
                    'max_hamming': 1,
                    'quad_decimate': 1.0,
                    'quad_sigma': 0.0,
                    'nthreads': 1,
                    'decode_sharpening': 0.25,
                    'min_tag_perimeter': 3,
                    'min_corner_side': 3,
                    'output_frame': 'camera_rgb_optical_frame',
                }],
                remappings=[
                    ('image', 'camera/rgb/image_rect_color'),
                    ('camera_info', 'camera/rgb/camera_info'),
                    ('detections', 'apriltag/detections'),
                    ('transforms', 'apriltag/transforms'),
                ],
                condition=IfCondition(enable_object_detection)
            ),
        ],
        output='screen',
    )

    # Separate container for VSLAM (Visual Simultaneous Localization and Mapping)
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
                name='visual_slam',
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
                        'enable_mapper': True,
                        'enable_localization': True,
                        'enable_loop_closure': True,
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
                ],
                condition=IfCondition(enable_visual_slam)
            ),
        ],
        output='screen',
        condition=IfCondition(enable_visual_slam)
    )

    # Additional nodes that run outside containers
    # Image processing node for depth refinement
    depth_processing_node = Node(
        package='isaac_ros_depth_proc',
        executable='isaac_ros_depth_proc',
        name='depth_processing',
        namespace=namespace,
        parameters=[
            {
                'use_sim_time': True,
                'fill_holes': True,
                'hole_size': 3,
                'min_depth': 0.1,
                'max_depth': 10.0,
            }
        ],
        remappings=[
            ('image_raw', 'camera/depth/image_raw'),
            ('image_processed', 'camera/depth/image_processed'),
        ],
        condition=IfCondition(enable_sensor_fusion)
    )

    # TF broadcaster for sensor frames
    tf_broadcaster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='sensor_tf_broadcaster',
        namespace=namespace,
        arguments=[
            '--x', '0.1', '--y', '0.0', '--z', '0.2',
            '--qx', '0.0', '--qy', '0.0', '--qz', '0.0', '--qw', '1.0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_rgb_optical_frame'
        ]
    )

    # Return launch description
    return LaunchDescription([
        namespace_arg,
        enable_rectification_arg,
        enable_object_detection_arg,
        enable_sensor_fusion_arg,
        enable_visual_slam_arg,
        perception_container,
        vslam_container,
        depth_processing_node,
        tf_broadcaster_node,
    ])


def main():
    """Main function to run the launch file."""
    ld = generate_launch_description()
    print("Isaac ROS Perception Pipeline Launch Configuration Created")
    print("This configuration includes:")
    print("- Image rectification and processing")
    print("- Object detection with spatial information")
    print("- Point cloud creation from RGB-D data")
    print("- Visual SLAM for localization and mapping")
    print("- Stereo processing for depth estimation")
    return ld


if __name__ == '__main__':
    main()