# Launch Files for ROS 2 Packages

## Overview

Launch files are Python scripts that allow you to start multiple ROS 2 nodes with a single command. They provide a way to configure and manage complex systems with many interconnected nodes. This document covers how to create and use launch files for your humanoid robotics system.

## Why Use Launch Files?

Launch files provide several advantages:
- **Convenience**: Start multiple nodes with a single command
- **Configuration**: Set parameters and arguments for nodes
- **Conditional execution**: Start nodes based on conditions
- **Parameter management**: Centralize configuration settings
- **Reusability**: Create different launch configurations for different scenarios

## Basic Launch File Structure

A basic launch file has the following structure:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Define nodes to launch
    my_node = Node(
        package='package_name',
        executable='executable_name',
        name='node_name',
        parameters=[
            {'param_name': 'param_value'}
        ],
        output='screen'
    )

    # Return launch description
    return LaunchDescription([
        my_node
    ])
```

## Creating Launch Arguments

Launch arguments allow you to pass parameters to your launch file:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    my_param_arg = DeclareLaunchArgument(
        'my_param',
        default_value='default_value',
        description='Description of the parameter'
    )

    # Use the launch configuration in a node
    my_node = Node(
        package='package_name',
        executable='executable_name',
        name='node_name',
        parameters=[
            {'param_name': LaunchConfiguration('my_param')}
        ]
    )

    return LaunchDescription([
        my_param_arg,
        my_node
    ])
```

## Conditional Launch

Use conditions to start nodes based on launch arguments:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch argument
    use_gui_arg = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='Use GUI if true'
    )

    # Node that starts only if use_gui is true
    gui_node = Node(
        condition=IfCondition(LaunchConfiguration('use_gui')),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    return LaunchDescription([
        use_gui_arg,
        gui_node
    ])
```

## Working with Parameters

### Setting Global Parameters

Set parameters that apply to all nodes:

```python
from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    # Set global parameter
    set_use_sim_time = SetParameter(
        name='use_sim_time',
        value=True
    )

    # Node that will use the global parameter
    my_node = Node(
        package='package_name',
        executable='executable_name',
        name='node_name'
    )

    return LaunchDescription([
        set_use_sim_time,
        my_node
    ])
```

### Loading Parameters from Files

Load parameters from YAML configuration files:

```python
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Node with parameters from a file
    my_node = Node(
        package='package_name',
        executable='executable_name',
        name='node_name',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('package_name'),
                'config',
                'my_config.yaml'
            ])
        ]
    )

    return LaunchDescription([
        my_node
    ])
```

## Including Other Launch Files

Include other launch files to compose complex systems:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Include another launch file
    other_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('other_package'),
                'launch',
                'other_launch_file.launch.py'
            ])
        ])
    )

    return LaunchDescription([
        other_launch
    ])
```

## Launch File for Humanoid Control System

### Joint Control Launch File

The joint control launch file starts the basic nodes needed for joint control:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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

    # Robot state publisher
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

    return LaunchDescription([
        use_sim_time_arg,
        set_use_sim_time,
        joint_command_publisher,
        robot_state_publisher
    ])
```

### Complete Humanoid System Launch File

The complete system launch file combines all components:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
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

    launch_ai_bridge_arg = DeclareLaunchArgument(
        'launch_ai_bridge',
        default_value='true',
        description='Launch AI bridge node if true'
    )

    # Set use_sim_time parameter globally
    set_use_sim_time = SetParameter(
        name='use_sim_time',
        value=LaunchConfiguration('use_sim_time')
    )

    # Include joint control launch file
    joint_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'launch',
                'joint_control.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )

    # AI Bridge node (conditional)
    ai_bridge_node = Node(
        package='ai_bridge',
        executable='ai_bridge',
        name='ai_bridge',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        condition=IfCondition(LaunchConfiguration('launch_ai_bridge')),
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        launch_ai_bridge_arg,
        set_use_sim_time,
        joint_control_launch,
        ai_bridge_node
    ])
```

## Best Practices for Launch Files

### 1. Use Descriptive Names

Use clear, descriptive names for launch arguments and nodes to make your launch files self-documenting.

### 2. Provide Default Values

Always provide sensible default values for launch arguments to make your launch files easier to use.

### 3. Document Your Launch Files

Add comments to explain the purpose of each node and parameter in your launch file.

### 4. Organize by Function

Group related nodes together and use consistent parameter naming conventions.

### 5. Use Path Substitutions

Use `PathJoinSubstitution` and `FindPackageShare` for robust file path handling across different systems.

### 6. Handle Dependencies Properly

Use event handlers if nodes need to start in a specific order or wait for other nodes to be ready.

## Running Launch Files

### Basic Usage

```bash
# Run a launch file with default parameters
ros2 launch package_name launch_file.launch.py

# Run with specific arguments
ros2 launch package_name launch_file.launch.py use_sim_time:=true
```

### Getting Help

```bash
# Show available launch arguments
ros2 launch package_name launch_file.launch.py --show-args
```

## Advanced Launch Concepts

### Event Handlers

Use event handlers to react to node lifecycle events:

```python
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    node1 = Node(
        package='package_name',
        executable='executable1',
        name='node1'
    )

    node2 = Node(
        package='package_name',
        executable='executable2',
        name='node2'
    )

    # Start node2 when node1 exits
    event_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=node1,
            on_exit=[node2]
        )
    )

    return LaunchDescription([
        node1,
        event_handler
    ])
```

## Testing Launch Files

### Verification Steps

1. **Syntax Check**: Verify the Python syntax of your launch file
2. **Dry Run**: Use `--dry-run` to see what would be launched without actually starting nodes
3. **Parameter Validation**: Check that all parameters are correctly loaded
4. **Node Communication**: Verify that nodes can communicate as expected

### Common Testing Commands

```bash
# Dry run to check what would be launched
ros2 launch package_name launch_file.launch.py --dry-run

# List nodes after launch to verify they started
ros2 node list

# Check parameters
ros2 param list node_name
```

## Troubleshooting Launch Files

### Common Issues

- **Import Errors**: Missing imports or incorrect package names
- **Path Issues**: Incorrect file paths in PathJoinSubstitution
- **Parameter Issues**: Mismatched parameter types or missing parameter files
- **Node Issues**: Executables not found or incorrect node names

### Debugging Tips

- Check the console output for error messages
- Use `--show-args` to see available arguments
- Test nodes individually before including them in launch files
- Verify that all required files exist at the specified paths

## Summary

Launch files are essential for managing complex ROS 2 systems. They provide a way to:
- Start multiple nodes with a single command
- Configure nodes with parameters
- Conditionally launch nodes based on arguments
- Include other launch files to compose systems
- Set global parameters for all nodes

By following best practices and using the patterns shown in this document, you can create robust and maintainable launch files for your humanoid robotics system.