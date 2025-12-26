import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch file for VLA action execution components."""

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='vla',
        description='Namespace for action execution nodes'
    )

    # Get launch configurations
    namespace = LaunchConfiguration('namespace')

    # Action executor node
    action_executor_node = Node(
        package='vla_integration',
        executable='action_executor',
        name='action_executor',
        namespace=namespace,
        output='screen'
    )

    # Task execution context node
    task_execution_context_node = Node(
        package='vla_integration',
        executable='task_execution_context',
        name='task_execution_context',
        namespace=namespace,
        output='screen'
    )

    # Action sequencer node
    action_sequencer_node = Node(
        package='vla_integration',
        executable='action_sequencer',
        name='action_sequencer',
        namespace=namespace,
        output='screen'
    )

    # Action execution service node
    action_execution_service_node = Node(
        package='vla_integration',
        executable='action_execution_service',
        name='action_execution_service',
        namespace=namespace,
        output='screen'
    )

    # Action status service node
    action_status_service_node = Node(
        package='vla_integration',
        executable='action_status_service',
        name='action_status_service',
        namespace=namespace,
        output='screen'
    )

    # Error recovery node
    error_recovery_node = Node(
        package='vla_integration',
        executable='error_recovery',
        name='error_recovery',
        namespace=namespace,
        output='screen'
    )

    # Safety validator node
    safety_validator_node = Node(
        package='vla_integration',
        executable='safety_validator',
        name='safety_validator',
        namespace=namespace,
        output='screen'
    )

    # Safety service node
    safety_service_node = Node(
        package='vla_integration',
        executable='safety_service',
        name='safety_service',
        namespace=namespace,
        output='screen'
    )

    # Robot state monitor node
    robot_state_monitor_node = Node(
        package='vla_integration',
        executable='robot_state_monitor',
        name='robot_state_monitor',
        namespace=namespace,
        output='screen'
    )

    # VLA integration node
    vla_integration_node = Node(
        package='vla_integration',
        executable='vla_integration',
        name='vla_integration',
        namespace=namespace,
        output='screen'
    )

    return LaunchDescription([
        namespace_arg,
        action_executor_node,
        task_execution_context_node,
        action_sequencer_node,
        action_execution_service_node,
        action_status_service_node,
        error_recovery_node,
        safety_validator_node,
        safety_service_node,
        robot_state_monitor_node,
        vla_integration_node,
    ])