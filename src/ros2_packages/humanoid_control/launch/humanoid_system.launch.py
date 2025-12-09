from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    launch_joint_control_arg = DeclareLaunchArgument(
        'launch_joint_control',
        default_value='true',
        description='Launch joint control nodes if true'
    )

    launch_ai_bridge_arg = DeclareLaunchArgument(
        'launch_ai_bridge',
        default_value='true',
        description='Launch AI bridge node if true'
    )

    launch_visualization_arg = DeclareLaunchArgument(
        'launch_visualization',
        default_value='true',
        description='Launch visualization nodes if true'
    )

    # Set use_sim_time parameter globally
    set_use_sim_time = SetParameter(
        name='use_sim_time',
        value=LaunchConfiguration('use_sim_time')
    )

    # Include the joint control launch file if requested
    joint_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'launch',
                'joint_control.launch.py'
            ])
        ]),
        condition=IfCondition(LaunchConfiguration('launch_joint_control')),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )

    # AI Bridge node
    ai_bridge_node = Node(
        package='ai_bridge',
        executable='ai_bridge',
        name='ai_bridge',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'config',
                'humanoid_params.yaml'
            ])
        ],
        condition=IfCondition(LaunchConfiguration('launch_ai_bridge')),
        output='screen'
    )

    # Sensor processing node
    sensor_processing_node = Node(
        package='humanoid_control',
        executable='sensor_processing_node',
        name='sensor_processing_node',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'config',
                'humanoid_params.yaml'
            ])
        ],
        output='screen'
    )

    # State estimation node
    state_estimation_node = Node(
        package='humanoid_control',
        executable='state_estimation_node',
        name='state_estimation_node',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'config',
                'humanoid_params.yaml'
            ])
        ],
        output='screen'
    )

    # Behavior manager node
    behavior_manager_node = Node(
        package='humanoid_control',
        executable='behavior_manager_node',
        name='behavior_manager_node',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'config',
                'humanoid_params.yaml'
            ])
        ],
        output='screen'
    )

    # Robot state publisher (to publish TF transforms)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'robot_description':
                PathJoinSubstitution([
                    FindPackageShare('humanoid_control'),
                    'urdf',
                    'basic_humanoid.urdf'
                ])
            }
        ],
        output='screen'
    )

    # Joint state publisher (for GUI control during testing)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('launch_visualization')),
        output='screen'
    )

    # RViz2 node
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d',
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'rviz',
                'humanoid_config.rviz'
            ])
        ],
        condition=IfCondition(LaunchConfiguration('launch_visualization')),
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments and parameter setting
    ld.add_action(use_sim_time_arg)
    ld.add_action(launch_joint_control_arg)
    ld.add_action(launch_ai_bridge_arg)
    ld.add_action(launch_visualization_arg)
    ld.add_action(set_use_sim_time)

    # Add launch files and nodes
    ld.add_action(joint_control_launch)
    ld.add_action(ai_bridge_node)
    ld.add_action(sensor_processing_node)
    ld.add_action(state_estimation_node)
    ld.add_action(behavior_manager_node)
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher_gui)
    ld.add_action(rviz2)

    return ld