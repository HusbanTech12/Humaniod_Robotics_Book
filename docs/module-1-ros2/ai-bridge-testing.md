# Testing the AI Bridge Node

## Overview

This document explains how to test the AI Bridge node that connects AI agents to ROS 2 controllers. The AI Bridge node processes sensor data and generates control commands for humanoid robots.

## Prerequisites

Before testing, ensure that:
1. ROS 2 Humble Hawksbill is installed and sourced
2. The `ai_bridge` package is built
3. You have terminal access to run multiple ROS 2 nodes
4. You have the `humanoid_control` package built as well

## Testing the AI Bridge

### Step 1: Build the Package

First, make sure your packages are built:

```bash
cd ~/ros2_ws  # or wherever your ROS 2 workspace is located
colcon build --packages-select ai_bridge humanoid_control
source install/setup.bash
```

### Step 2: Run a Sensor Node (Simulator)

Since we don't have a real robot, we'll use a simple joint state publisher to simulate sensor data:

```bash
# Terminal 1: Run the joint state publisher
source ~/ros2_ws/install/setup.bash
ros2 run joint_state_publisher joint_state_publisher
```

### Step 3: Run the AI Bridge Node

In another terminal, run the AI bridge node:

```bash
# Terminal 2: Run the AI bridge
source ~/ros2_ws/install/setup.bash
ros2 run ai_bridge ai_bridge
```

You should see output similar to:
```
[INFO] [1699123456.789] [ai_bridge]: AI Bridge node initialized
[INFO] [1699123456.889] [ai_bridge]: AI output: [0.123, -0.456, 0.789]...
```

### Step 4: Monitor the Output

In a third terminal, monitor the commands being published by the AI bridge:

```bash
# Terminal 3: Monitor the joint commands
source ~/ros2_ws/install/setup.bash
ros2 topic echo /joint_commands std_msgs/msg/Float64MultiArray
```

You should see the AI bridge publishing control commands based on the simulated sensor data.

### Step 5: Using ROS 2 Tools to Verify Communication

You can also use ROS 2 command-line tools to verify communication:

```bash
# List all available topics
ros2 topic list

# Check the publishers and subscribers for the joint commands topic
ros2 topic info /joint_commands

# Check the publishers and subscribers for joint states
ros2 topic info /joint_states
```

## Testing with a Simple Robot Controller

### Step 1: Run a Simple Controller Node

```bash
# Terminal 4: Run a simple controller to receive AI commands
source ~/ros2_ws/install/setup.bash
# You can use the joint_command_publisher from humanoid_control as a simple receiver
ros2 run humanoid_control joint_command_publisher
```

### Step 2: Verify End-to-End Communication

With all nodes running, you should see:
1. Joint state publisher sending simulated sensor data
2. AI bridge receiving sensor data and publishing control commands
3. Controller receiving and potentially using the AI-generated commands

## Expected Results

When the AI bridge is working correctly:
- The node should subscribe to sensor topics (joint_states, etc.)
- The node should process sensor data through the AI algorithm
- The node should publish control commands at the specified rate
- No errors should appear in the terminal output
- The AI algorithm should generate meaningful control outputs

## Troubleshooting

- If the AI bridge doesn't receive sensor data, check topic names and message types
- If no commands are published, verify that sensor data is being received
- Make sure all terminals have sourced the ROS 2 setup file
- Check for any error messages in the terminal output
- Verify that the AI model (or placeholder) is functioning correctly

## Integration Testing

For a more complete test, you can integrate the AI bridge with other nodes in a launch file that starts multiple nodes together, allowing you to test the complete AI-robot interaction loop.