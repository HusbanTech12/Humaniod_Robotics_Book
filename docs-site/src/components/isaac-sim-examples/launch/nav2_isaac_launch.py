#!/usr/bin/env python3
# nav2_isaac_launch.py

"""
Nav2 Isaac Sim Launch Configuration
This launch file configures the Nav2 navigation stack for use with Isaac Sim
and the humanoid robot, integrating with VSLAM for localization and perception.
"""

import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    """Generate launch description for Nav2 with Isaac Sim integration."""

    # Get package directories
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    isaac_ros_workspace = get_package_share_directory('isaac_ros_workspace')
    nav2_bt_dir = get_package_share_directory('nav2_bt_navigator')
    vslam_dir = get_package_share_directory('isaac_ros_visual_slam')

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='humanoid_robot',
        description='Namespace for the robot'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    autostart_arg = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically start lifecycle nodes'
    )

    default_bt_xml_filename_arg = DeclareLaunchArgument(
        'default_bt_xml_filename',
        default_value=os.path.join(nav2_bt_dir, 'behavior_trees', 'navigate_w_replanning_and_recovery.xml'),
        description='Full path to the behavior tree xml file to use'
    )

    map_yaml_file_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(nav2_bringup_dir, 'maps', 'turtlebot3_world.yaml'),
        description='Full path to map file to load'
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(isaac_ros_workspace, 'config', 'humanoid_nav2_config.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes'
    )

    # Get launch configurations
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    default_bt_xml_filename = LaunchConfiguration('default_bt_xml_filename')
    map_yaml_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')

    # Create parameter substitutions with namespace
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'default_bt_xml_filename': default_bt_xml_filename,
        'autostart': autostart
    }

    # Create configured parameters
    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True
    )

    # Include the main navigation launch file
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'namespace': namespace,
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'params_file': configured_params,
            'default_bt_xml_filename': default_bt_xml_filename
        }.items()
    )

    # Include the local costmap launch file
    local_costmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'local_costmap.launch.py')),
        launch_arguments={
            'namespace': namespace,
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'params_file': configured_params
        }.items()
    )

    # Include the global costmap launch file
    global_costmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'global_costmap.launch.py')),
        launch_arguments={
            'namespace': namespace,
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'params_file': configured_params
        }.items()
    )

    # Map server node (configured for Isaac Sim VSLAM integration)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'yaml_filename': map_yaml_file},
            {'topic_name': '/vslam/occupancy_grid'},  # Subscribe to VSLAM-generated map
            {'frame_id': 'map'},
            {'output': 'screen'},
            {'topic_rate': 1.0}
        ],
        output='screen'
    )

    # Local planner node with Isaac Sim-specific configuration
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        namespace=namespace,
        parameters=[configured_params],
        remappings=[
            ('cmd_vel', 'cmd_vel'),
            ('odom', 'odom'),
            ('global_costmap', 'local_costmap'),
            ('global_costmap/global_costmap', 'local_costmap/costmap'),
            ('global_costmap/costmap_updates', 'local_costmap/costmap_updates')
        ],
        output='screen'
    )

    # Planner server node with Isaac Sim-specific configuration
    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        namespace=namespace,
        parameters=[configured_params],
        remappings=[
            ('global_costmap', 'global_costmap'),
            ('global_costmap/costmap', 'global_costmap/costmap'),
            ('global_costmap/costmap_updates', 'global_costmap/costmap_updates')
        ],
        output='screen'
    )

    # Behavior tree navigator node
    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        namespace=namespace,
        parameters=[configured_params],
        remappings=[
            ('local_costmap', 'local_costmap'),
            ('global_costmap', 'global_costmap'),
            ('local_costmap/costmap', 'local_costmap/costmap'),
            ('global_costmap/costmap', 'global_costmap/costmap'),
            ('local_costmap/costmap_updates', 'local_costmap/costmap_updates'),
            ('global_costmap/costmap_updates', 'global_costmap/costmap_updates')
        ],
        output='screen'
    )

    # Lifecycle manager for navigation nodes
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': autostart},
            {'node_names': [
                'map_server',
                'planner_server',
                'controller_server',
                'bt_navigator',
                'local_costmap',
                'global_costmap'
            ]}
        ],
        output='screen'
    )

    # Isaac Sim-specific nodes for integration
    # VSLAM integration node
    vslam_integration_node = Node(
        package='isaac_ros_visual_slam',
        executable='vslam_integration_node',  # Custom node for VSLAM-Nav2 integration
        name='vslam_nav_integration',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'map_frame': 'map'},
            {'odom_frame': 'odom'},
            {'base_frame': 'base_link'},
            {'vslam_pose_topic': '/vslam/pose'},
            {'vslam_map_topic': '/vslam/occupancy_grid'},
            {'output': 'screen'}
        ],
        output='screen'
    )

    # Perception integration node
    perception_integration_node = Node(
        package='isaac_ros_workspace',
        executable='perception_nav_integration_node',  # Custom node for perception-nav integration
        name='perception_nav_integration',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'detection_topic': '/detectnet/spatial_detections'},
            {'obstacle_topic': '/obstacle_markers'},
            {'output': 'screen'}
        ],
        output='screen'
    )

    # TF broadcaster for Isaac Sim coordinate frames
    tf_broadcaster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='nav2_tf_broadcaster',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--qx', '0.0', '--qy', '0.0', '--qz', '0.0', '--qw', '1.0',
            '--frame-id', 'odom',
            '--child-frame-id', 'base_link'
        ]
    )

    # RViz2 for visualization with Nav2-specific configuration
    rviz_config_file = os.path.join(
        isaac_ros_workspace,
        'rviz',
        'nav2_isaac_config.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('enable_rviz', default='true'))
    )

    # Return launch description
    return LaunchDescription([
        namespace_arg,
        use_sim_time_arg,
        autostart_arg,
        default_bt_xml_filename_arg,
        map_yaml_file_arg,
        params_file_arg,
        navigation_launch,
        local_costmap_launch,
        global_costmap_launch,
        map_server_node,
        controller_server_node,
        planner_server_node,
        bt_navigator_node,
        lifecycle_manager_node,
        vslam_integration_node,
        perception_integration_node,
        tf_broadcaster_node,
        rviz_node
    ])


def main():
    """Main function to run the launch file."""
    ld = generate_launch_description()
    print("Isaac ROS Nav2 Integration Launch Configuration Created")
    print("This configuration includes:")
    print("- Nav2 navigation stack with humanoid-specific parameters")
    print("- VSLAM integration for localization and mapping")
    print("- Perception system integration for dynamic obstacle avoidance")
    print("- Custom behavior trees for humanoid navigation")
    print("- TF broadcasting for coordinate frame management")
    print("- RViz2 visualization for navigation monitoring")
    return ld


if __name__ == '__main__':
    main()