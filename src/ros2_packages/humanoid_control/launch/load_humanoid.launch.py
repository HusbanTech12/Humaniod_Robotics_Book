from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    use_gui_arg = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='Whether to use the Gazebo GUI'
    )

    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value=os.path.join(
            get_package_share_directory('humanoid_control'),
            'worlds',
            'simple_room.world'
        ),
        description='Path to the world file to load'
    )

    robot_model_arg = DeclareLaunchArgument(
        'robot_model',
        default_value=os.path.join(
            get_package_share_directory('humanoid_control'),
            'urdf',
            'basic_humanoid.urdf'
        ),
        description='Path to the robot URDF file'
    )

    # Get launch configurations
    use_gui = LaunchConfiguration('use_gui')
    world_file = LaunchConfiguration('world_file')
    robot_model = LaunchConfiguration('robot_model')

    # Get the package share directory
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # Launch Gazebo server and GUI
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items()
    )

    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ),
        launch_arguments={'verbose': 'true'}.items(),
        condition=IfCondition(use_gui)
    )

    # Read the robot model file
    with open(robot_model, 'r') as infp:
        robot_desc = infp.read()

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_desc,
            'publish_frequency': 50.0
        }],
        output='screen'
    )

    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'  # Start slightly above ground to avoid collision with floor
        ],
        output='screen'
    )

    # Joint State Publisher node (for visualization in RViz)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_gui': use_gui,
            'rate': 50.0
        }],
        output='screen'
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(use_gui_arg)
    ld.add_action(world_file_arg)
    ld.add_action(robot_model_arg)

    # Add actions
    ld.add_action(gazebo_server)
    ld.add_action(gazebo_gui)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_entity)
    ld.add_action(joint_state_publisher)

    return ld