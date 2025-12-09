# Testing Package Deployment and Multi-Node Initialization

## Overview

This document explains how to test the deployment of your ROS 2 packages and the initialization of multi-node systems for your humanoid robot. Proper testing ensures that all components work together as expected in a coordinated system.

## Prerequisites

Before testing, ensure that:
1. ROS 2 Humble Hawksbill is installed and sourced
2. Both `humanoid_control` and `ai_bridge` packages are built
3. You have terminal access to run multiple ROS 2 nodes
4. All required dependencies are installed

## Building the Packages

### Step 1: Navigate to Your Workspace

```bash
cd ~/ros2_ws  # or wherever your ROS 2 workspace is located
```

### Step 2: Build the Packages

```bash
# Build specific packages
colcon build --packages-select humanoid_control ai_bridge

# Or build all packages in the workspace
colcon build

# Source the workspace
source install/setup.bash
```

## Testing Individual Package Deployment

### Test humanoid_control Package

```bash
# Check that the package is recognized
ros2 pkg list | grep humanoid_control

# Check executables in the package
ros2 pkg executables humanoid_control
```

### Test ai_bridge Package

```bash
# Check that the package is recognized
ros2 pkg list | grep ai_bridge

# Check executables in the package
ros2 pkg executables ai_bridge
```

## Testing Multi-Node Initialization

### Method 1: Launch Individual Nodes

Test each node individually to ensure they start properly:

```bash
# Terminal 1: Launch the joint command publisher
source ~/ros2_ws/install/setup.bash
ros2 run humanoid_control joint_command_publisher

# Terminal 2: Launch the sensor subscriber
source ~/ros2_ws/install/setup.bash
ros2 run humanoid_control sensor_subscriber

# Terminal 3: Launch the AI bridge
source ~/ros2_ws/install/setup.bash
ros2 run ai_bridge ai_bridge
```

### Method 2: Use Launch Files

Use the launch files you created to start multiple nodes at once:

```bash
# Launch the joint control system
source ~/ros2_ws/install/setup.bash
ros2 launch humanoid_control joint_control.launch.py
```

In another terminal:

```bash
# Launch the complete humanoid system
source ~/ros2_ws/install/setup.bash
ros2 launch humanoid_control humanoid_system.launch.py
```

## Verifying Node Communication

### Check Active Nodes

```bash
# List all active nodes
ros2 node list
```

You should see nodes like:
- `/joint_command_publisher`
- `/sensor_subscriber`
- `/ai_bridge`
- `/robot_state_publisher`
- `/sensor_processing_node`
- `/state_estimation_node`
- `/behavior_manager_node`

### Check Active Topics

```bash
# List all active topics
ros2 topic list

# Check topic information
ros2 topic info /joint_commands
ros2 topic info /joint_states
ros2 topic info /sensor_data
```

### Test Topic Communication

```bash
# Echo a topic to see if data is flowing
ros2 topic echo /joint_states sensor_msgs/msg/JointState

# Publish to a topic to test the communication
ros2 topic pub /test_topic std_msgs/msg/String "data: 'test message'"
```

## Testing Parameter Configuration

### Check Parameter Values

```bash
# Get parameters from a specific node
ros2 param list /joint_command_publisher
ros2 param get /joint_command_publisher use_sim_time

# Set a parameter
ros2 param set /joint_command_publisher use_sim_time true
```

### Verify Configuration Files

Check that your parameter files are loaded correctly:

```bash
# Verify the contents of your parameter files
cat ~/ros2_ws/src/ros2_packages/humanoid_control/config/humanoid_params.yaml
```

## Testing the Complete System

### Launch the Full System

```bash
# Launch the complete humanoid system with all components
source ~/ros2_ws/install/setup.bash
ros2 launch humanoid_control humanoid_system.launch.py launch_visualization:=true
```

### Verify System Components

With the system running, verify each component:

1. **Check that all nodes are active:**
   ```bash
   ros2 node list
   ```

2. **Verify parameter loading:**
   ```bash
   ros2 param list /ai_bridge
   ```

3. **Test topic communication:**
   ```bash
   ros2 topic list
   ros2 topic echo /joint_commands --field data -n 5
   ```

4. **Check service availability:**
   ```bash
   ros2 service list
   ros2 service call /configure_ai_control std_srvs/srv/SetBool "{data: true}"
   ```

## Testing with Different Launch Arguments

Test your launch files with different configurations:

```bash
# Launch with simulation time
ros2 launch humanoid_control humanoid_system.launch.py use_sim_time:=true

# Launch without visualization
ros2 launch humanoid_control humanoid_system.launch.py launch_visualization:=false

# Launch without AI bridge
ros2 launch humanoid_control humanoid_system.launch.py launch_ai_bridge:=false
```

## Troubleshooting Common Issues

### Nodes Not Starting

If nodes fail to start:
- Check that the packages are built and sourced
- Verify that all dependencies are installed
- Check the console output for specific error messages
- Ensure that required configuration files exist

### Parameter Loading Issues

If parameters aren't loading:
- Verify that parameter file paths are correct
- Check that parameter file syntax is valid YAML
- Ensure the node is configured to read the parameter file

### Communication Issues

If nodes can't communicate:
- Check that nodes are using the correct topic names
- Verify that message types match between publishers and subscribers
- Ensure all nodes are using the same `use_sim_time` setting if applicable

### Launch File Issues

If launch files fail:
- Check for syntax errors in the Python launch files
- Verify that referenced packages and executables exist
- Ensure all path substitutions are correct

## Expected Results

When the system is working correctly:
- All nodes start without errors
- Parameter values are correctly loaded from configuration files
- Topics are connected and messages flow between nodes
- Services are available and respond to requests
- The complete humanoid system operates as an integrated whole
- No error messages appear in the console output

## Performance Testing

### Check System Resources

Monitor the system to ensure nodes are running efficiently:

```bash
# Check CPU and memory usage of ROS 2 processes
htop
# Look for processes named after your nodes

# Monitor network usage if applicable
iftop
```

### Test System Responsiveness

Verify that the system responds appropriately to commands and maintains performance under load.

## Next Steps

Once your package deployment and multi-node initialization are working:
- Test with real robot hardware (if available)
- Implement more sophisticated control algorithms
- Add additional sensors and actuators
- Optimize performance and resource usage
- Implement error handling and recovery procedures