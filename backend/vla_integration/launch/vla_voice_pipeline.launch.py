import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch file for VLA voice pipeline components."""

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='vla',
        description='Namespace for voice pipeline nodes'
    )

    # Get launch configurations
    namespace = LaunchConfiguration('namespace')

    # VLA pipeline node
    vla_pipeline_node = Node(
        package='vla_integration',
        executable='vla_pipeline',
        name='vla_pipeline',
        namespace=namespace,
        parameters=[
            os.path.join(
                get_package_share_directory('vla_integration'),
                'config',
                'vla_pipeline_config.yaml'
            )
        ],
        remappings=[
            ('/vla/voice/command', 'vla/voice/command'),
            ('/vla/execution_status', 'vla/execution_status'),
        ],
        output='screen'
    )

    # Voice command service node
    voice_command_service_node = Node(
        package='vla_integration',
        executable='voice_command_service',
        name='voice_command_service',
        namespace=namespace,
        parameters=[
            os.path.join(
                get_package_share_directory('vla_integration'),
                'config',
                'voice_config.yaml'
            )
        ],
        remappings=[
            ('/vla/voice/process_command', 'vla/voice/process_command'),
        ],
        output='screen'
    )

    return LaunchDescription([
        namespace_arg,
        vla_pipeline_node,
        voice_command_service_node,
    ])