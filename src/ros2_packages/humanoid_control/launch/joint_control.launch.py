from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    # Set use_sim_time parameter globally
    set_use_sim_time = SetParameter(
        name='use_sim_time',
        value=LaunchConfiguration('use_sim_time')
    )

    # Joint command publisher node
    joint_command_publisher = Node(
        package='humanoid_control',
        executable='joint_command_publisher',
        name='joint_command_publisher',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Sensor subscriber node
    sensor_subscriber = Node(
        package='humanoid_control',
        executable='sensor_subscriber',
        name='sensor_subscriber',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Joint state publisher (for visualization and feedback)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
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

    # Controller manager (for ROS2 Control)
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'config',
                'controllers.yaml'
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Joint trajectory controller spawner
    joint_trajectory_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller'],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    # Joint state broadcaster spawner
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments and parameter setting
    ld.add_action(use_sim_time_arg)
    ld.add_action(set_use_sim_time)

    # Add nodes
    ld.add_action(joint_command_publisher)
    ld.add_action(sensor_subscriber)
    ld.add_action(joint_state_publisher)
    ld.add_action(robot_state_publisher)
    ld.add_action(controller_manager)
    ld.add_action(joint_trajectory_spawner)
    ld.add_action(joint_state_broadcaster_spawner)

    return ld