from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    use_gui_arg = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='Set to "false" to run headless'
    )

    model_arg = DeclareLaunchArgument(
        'model',
        default_value='basic_humanoid.urdf',
        description='URDF file name'
    )

    # Get URDF file path
    urdf_path = PathJoinSubstitution([
        FindPackageShare('humanoid_control'),
        'urdf',
        LaunchConfiguration('model')
    ])

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': [
                'file://',
                urdf_path,
                ' use_sim_time:=false'
            ]
        }]
    )

    # Joint State Publisher node
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_gui': LaunchConfiguration('use_gui')
        }]
    )

    # Joint State Publisher GUI (only if use_gui is true)
    joint_state_publisher_gui = Node(
        condition=IfCondition(LaunchConfiguration('use_gui')),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
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
                'urdf_config.rviz'
            ])
        ],
        output='screen'
    )

    return LaunchDescription([
        use_gui_arg,
        model_arg,
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        rviz2
    ])