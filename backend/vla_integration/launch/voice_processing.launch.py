import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch file for VLA voice processing components."""

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='vla',
        description='Namespace for voice processing nodes'
    )

    # Get launch configurations
    namespace = LaunchConfiguration('namespace')

    # Voice processor node
    voice_processor_node = Node(
        package='vla_integration',
        executable='voice_processor',
        name='voice_processor',
        namespace=namespace,
        parameters=[
            os.path.join(
                get_package_share_directory('vla_integration'),
                'config',
                'voice_config.yaml'
            )
        ],
        remappings=[
            ('/voice/command', 'vla/voice/command'),
            ('/voice/transcription', 'vla/voice/transcription'),
        ],
        output='screen'
    )

    # LLM planner node
    llm_planner_node = Node(
        package='vla_integration',
        executable='llm_planner',
        name='llm_planner',
        namespace=namespace,
        parameters=[
            os.path.join(
                get_package_share_directory('vla_integration'),
                'config',
                'voice_config.yaml'
            )
        ],
        remappings=[
            ('/voice/transcription', 'vla/voice/transcription'),
            ('/vla/action_plan', 'vla/action_plan'),
        ],
        output='screen'
    )

    # Action planner node
    action_planner_node = Node(
        package='vla_integration',
        executable='action_planner',
        name='action_planner',
        namespace=namespace,
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
            ('/voice/process_command', 'vla/voice/process_command'),
        ],
        output='screen'
    )

    return LaunchDescription([
        namespace_arg,
        voice_processor_node,
        llm_planner_node,
        action_planner_node,
        voice_command_service_node,
    ])