# Quickstart Guide: ROS 2 for Humanoid Robotics

## Overview

This quickstart guide provides a fast path to getting your ROS 2 humanoid robotics system up and running. It covers the essential steps to set up, build, and run a basic humanoid robot control system with AI integration.

## Prerequisites

Before starting, ensure you have:
- Ubuntu 22.04 LTS (recommended) or compatible Linux system
- ROS 2 Humble Hawksbill installed
- Python 3.8 or higher
- Git installed

## Step 1: Install ROS 2 Humble

If you haven't already installed ROS 2 Humble:

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros-humble.repos | sudo tee /etc/ros/ros.repos
sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2 environment
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Step 2: Set Up Your ROS 2 Workspace

Create and set up your workspace for the humanoid robotics project:

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Clone the humanoid robotics packages (if available in a repository)
# Or copy the package files you've created to the src directory
# For this example, we'll assume you have the packages locally

# Build the workspace
colcon build --packages-select humanoid_control ai_bridge

# Source the workspace
source install/setup.bash
```

## Step 3: Verify Package Structure

Ensure your package structure looks like this:

```
~/ros2_ws/src/
├── humanoid_control/
│   ├── package.xml
│   ├── setup.py
│   ├── humanoid_control/
│   │   ├── __init__.py
│   │   ├── joint_command_publisher.py
│   │   ├── sensor_subscriber.py
│   │   ├── config_service.py
│   │   ├── behavior_action_server.py
│   │   ├── sensor_processing_node.py
│   │   ├── state_estimation_node.py
│   │   └── behavior_manager_node.py
│   ├── launch/
│   │   ├── joint_control.launch.py
│   │   └── humanoid_system.launch.py
│   ├── urdf/
│   │   └── basic_humanoid.urdf
│   ├── config/
│   │   ├── controllers.yaml
│   │   └── humanoid_params.yaml
│   └── rviz/
│       └── humanoid_config.rviz
└── ai_bridge/
    ├── package.xml
    ├── setup.py
    ├── ai_bridge/
    │   ├── __init__.py
    │   └── ai_bridge.py
    └── launch/
```

## Step 4: Build and Source the Packages

```bash
cd ~/ros2_ws
colcon build --packages-select humanoid_control ai_bridge
source install/setup.bash
```

## Step 5: Run the Basic System

### Option A: Run Individual Nodes

Run nodes separately in different terminals:

**Terminal 1 - Launch the robot state publisher:**
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(xacro $(ros2 pkg prefix humanoid_control)/urdf/basic_humanoid.urdf)'
```

**Terminal 2 - Launch the joint state publisher:**
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run joint_state_publisher joint_state_publisher
```

**Terminal 3 - Launch the sensor processing node:**
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run humanoid_control sensor_processing_node
```

**Terminal 4 - Launch the AI bridge:**
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run ai_bridge ai_bridge
```

### Option B: Use Launch Files (Recommended)

Launch the complete system with a single command:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch humanoid_control humanoid_system.launch.py
```

## Step 6: Test Communication

Verify that nodes are communicating properly:

```bash
# List all active nodes
ros2 node list

# Check active topics
ros2 topic list

# Monitor joint commands being published
ros2 topic echo /joint_commands

# Monitor processed sensor data
ros2 topic echo /processed_sensor_data

# Send a behavior command
ros2 topic pub /behavior_command std_msgs/String "data: 'standing'"
```

## Step 7: Visualize the Robot

Launch RViz to visualize your humanoid robot:

```bash
# In a new terminal
cd ~/ros2_ws
source install/setup.bash
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix humanoid_control)/rviz/humanoid_config.rviz
```

Or launch with visualization as part of the system:

```bash
ros2 launch humanoid_control humanoid_system.launch.py launch_visualization:=true
```

## Step 8: Test AI Integration

Send commands to test the AI integration:

```bash
# Send a behavior command
ros2 topic pub /behavior_command std_msgs/String "data: 'walk'" --once

# Monitor the AI bridge output
ros2 topic echo /joint_commands

# Check the robot's estimated state
ros2 topic echo /estimated_state
```

## Common Commands Reference

### Package Management
```bash
# List all packages
ros2 pkg list

# Check package contents
ros2 pkg executables <package_name>

# Find a specific package
ros2 pkg prefix <package_name>
```

### Node Management
```bash
# List all nodes
ros2 node list

# Get information about a specific node
ros2 node info /node_name

# Get parameters for a node
ros2 param list /node_name
```

### Topic Management
```bash
# List all topics
ros2 topic list

# Get information about a topic
ros2 topic info /topic_name

# Monitor a topic
ros2 topic echo /topic_name

# Check message rate
ros2 topic hz /topic_name

# Publish to a topic
ros2 topic pub /topic_name MessageType "field1: value1; field2: value2"
```

### Service Management
```bash
# List all services
ros2 service list

# Call a service
ros2 service call /service_name ServiceType "{field1: value1, field2: value2}"
```

### Action Management
```bash
# List all actions
ros2 action list

# Send an action goal
ros2 action send_goal /action_name ActionType "{goal_field: goal_value}"
```

## Troubleshooting

### If nodes don't appear in `ros2 node list`:
- Make sure you've sourced the workspace: `source install/setup.bash`
- Verify the node scripts are executable
- Check for Python import errors in the terminal output

### If topics aren't showing up:
- Verify that nodes are running
- Check for typos in topic names
- Ensure message types match between publishers and subscribers

### If the robot model doesn't appear in RViz:
- Verify that robot_state_publisher is running
- Check that the URDF file path is correct
- Ensure joint_state_publisher is providing data

### If you get "command not found" errors:
- Make sure ROS 2 is sourced in your terminal
- Verify that packages have been built successfully
- Check that executables are properly declared in setup.py

## Next Steps

Once your basic system is running:

1. **Experiment with different behaviors**:
   ```bash
   ros2 topic pub /behavior_command std_msgs/String "data: 'balance'"
   ros2 topic pub /behavior_command std_msgs/String "data: 'gesturing'"
   ```

2. **Modify the AI bridge logic** in `ai_bridge.py` to implement custom control algorithms

3. **Add more sensors** to your URDF model and processing nodes

4. **Create custom launch files** for different scenarios

5. **Integrate with simulation** using Gazebo:
   ```bash
   # Launch with Gazebo simulation
   ros2 launch humanoid_control humanoid_with_gazebo.launch.py
   ```

## Summary

This quickstart guide provided the essential steps to:
- Set up your ROS 2 environment for humanoid robotics
- Build and run the basic system components
- Test communication between nodes
- Visualize your humanoid robot
- Verify AI integration is working

Your ROS 2 humanoid robotics system is now operational and ready for further development and experimentation!