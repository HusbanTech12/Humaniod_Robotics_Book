import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch file for complete VLA system."""

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='vla',
        description='Namespace for VLA system'
    )

    # Get launch configurations
    namespace = LaunchConfiguration('namespace')

    # Include voice processing launch file
    voice_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('vla_integration'),
            '/launch/voice_processing.launch.py'
        ]),
        launch_arguments={'namespace': namespace}.items()
    )

    # Include vision pipeline launch file
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('vla_integration'),
            '/launch/vision_pipeline.launch.py'
        ]),
        launch_arguments={'namespace': namespace}.items()
    )

    # Include action execution launch file
    action_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('vla_integration'),
            '/launch/action_execution.launch.py'
        ]),
        launch_arguments={'namespace': namespace}.items()
    )

    # VLA orchestrator node
    vla_orchestrator_node = Node(
        package='vla_integration',
        executable='vla_orchestrator',
        name='vla_orchestrator',
        namespace=namespace,
        parameters=[
            os.path.join(
                get_package_share_directory('vla_integration'),
                'config',
                'vla_pipeline_config.yaml'
            )
        ],
        output='screen'
    )

    # Voice-to-action interface node
    voice_to_action_interface_node = Node(
        package='vla_integration',
        executable='voice_to_action_interface',
        name='voice_to_action_interface',
        namespace=namespace,
        output='screen'
    )

    # Pipeline monitoring node
    pipeline_monitor_node = Node(
        package='vla_integration',
        executable='pipeline_monitor',
        name='pipeline_monitor',
        namespace=namespace,
        parameters=[
            os.path.join(
                get_package_share_directory('vla_integration'),
                'config',
                'vla_pipeline_config.yaml'
            )
        ],
        output='screen'
    )

    return LaunchDescription([
        namespace_arg,
        voice_launch,
        vision_launch,
        action_launch,
        vla_orchestrator_node,
        voice_to_action_interface_node,
        pipeline_monitor_node,
    ])