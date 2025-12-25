#!/usr/bin/env python3
# isaac_sim_launch.py

"""
Isaac Sim Launch File for Humanoid Robot
This launch file sets up the Isaac Sim environment with a humanoid robot
including physics, rendering, and sensor configurations.
"""

import sys
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    """Generate launch description for Isaac Sim with humanoid robot."""

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='humanoid_robot',
        description='Namespace for the robot'
    )

    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Isaac Sim) clock if true'
    )

    # Get launch configurations
    namespace = LaunchConfiguration('namespace')
    launch_rviz = LaunchConfiguration('launch_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Get package directories
    pkg_share = get_package_share_directory('isaac_ros_common')

    # Define Isaac Sim process
    isaac_sim_process = ExecuteProcess(
        cmd=[
            'isaac-sim',  # Command to launch Isaac Sim
            '--exec', 'from omni.isaac.kit import SimulationApp; '
                      'config = {"headless": False, "window_width": 1280, "window_height": 720}; '
                      'simulation_app = SimulationApp(config); '
                      'from omni.isaac.core import World; '
                      'from omni.isaac.core.utils.nucleus import get_assets_root_path; '
                      'from omni.isaac.core.utils.stage import add_reference_to_stage; '
                      'import carb; '
                      'world = World(stage_units_in_meters=1.0); '
                      'assets_root_path = get_assets_root_path(); '
                      'if assets_root_path is None: '
                      '    carb.log_error("Could not find Isaac Sim assets. Please check your Isaac Sim installation."); '
                      'else: '
                      '    add_reference_to_stage(assets_root_path + "/Isaac/Robots/Unitree/A1/unitree_a1.usd", "/World/Robot"); '
                      '    world.reset(); '
                      '    for i in range(100): world.step(render=True); '
                      'simulation_app.close()',
            '--no-window'
        ],
        output='screen',
        shell=True
    )

    # Robot state publisher node (for URDF)
    robot_description_content = xacro.process_file(
        os.path.join(get_package_share_directory('isaac_sim_examples'),
                     'robot', 'unitree_a1.urdf')
    ).toprettyxml(indent='  ')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': robot_description_content}
        ],
        remappings=[
            ('/joint_states', [namespace, '/joint_states'].join(''))
        ]
    )

    # Joint state publisher node
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=namespace,
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/joint_states', [namespace, '/joint_states'].join(''))
        ]
    )

    # Isaac Sim bridge node (placeholder - would connect Isaac Sim to ROS)
    isaac_sim_bridge_node = Node(
        package='isaac_ros_bridge',
        executable='isaac_sim_bridge',
        name='isaac_sim_bridge',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_namespace': namespace},
            {'config_file': os.path.join(
                get_package_share_directory('isaac_sim_examples'),
                'config', 'isaac_sim_bridge_config.yaml'
            )}
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ]
    )

    # Sensor processing nodes
    camera_processing_node = Node(
        package='image_proc',
        executable='image_proc',
        name='camera_processing',
        namespace=namespace,
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('image_raw', '/camera/rgb/image_raw'),
            ('camera_info', '/camera/rgb/camera_info'),
            ('image_rect_color', '/camera/rgb/image_rect_color'),
        ]
    )

    # Depth processing node
    depth_processing_node = Node(
        package='depth_image_proc',
        executable='convert_metric',
        name='depth_processing',
        namespace=namespace,
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('image_raw', '/camera/depth/image_raw'),
            ('image', '/camera/depth/image_metric'),
        ]
    )

    # Point cloud processing node
    pointcloud_processing_node = Node(
        package='depth_image_proc',
        executable='point_cloud_xyzrgb',
        name='pointcloud_processing',
        namespace=namespace,
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('rgb/image_rect_color', '/camera/rgb/image_rect_color'),
            ('depth/image_rect', '/camera/depth/image_rect'),
            ('rgb/camera_info', '/camera/rgb/camera_info'),
            ('points', '/camera/depth/color/points'),
        ]
    )

    # Launch RViz if requested
    rviz_config_file = os.path.join(
        get_package_share_directory('isaac_sim_examples'),
        'rviz',
        'humanoid_robot.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(launch_rviz),
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Return launch description
    return LaunchDescription([
        namespace_arg,
        launch_rviz_arg,
        use_sim_time_arg,
        isaac_sim_process,
        robot_state_publisher_node,
        joint_state_publisher_node,
        isaac_sim_bridge_node,
        camera_processing_node,
        depth_processing_node,
        pointcloud_processing_node,
        rviz_node,
    ])


def main(argv=sys.argv[1:]):
    """Main function to run the launch file."""
    ld = generate_launch_description()
    # In a real implementation, this would be launched differently
    # This is just a placeholder showing the structure
    print("Isaac Sim launch file structure created.")
    print("To run: ros2 launch docs-site/src/components/isaac-sim-examples/launch/isaac_sim_launch.py")
    return 0


if __name__ == '__main__':
    """Entry point for the launch file."""
    main()