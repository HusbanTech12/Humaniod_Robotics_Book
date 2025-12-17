# Testing ROS 2 Communication

## Overview

This document explains how to test the basic publisher/subscriber communication that was implemented in the previous steps.

## Prerequisites

Before testing, ensure that:
1. ROS 2 Humble Hawksbill is installed and sourced
2. The `humanoid_control` package is built
3. You have terminal access to run multiple ROS 2 nodes

## Testing Publisher/Subscriber Communication

### Step 1: Build the Package

First, make sure your package is built:

```bash
cd ~/ros2_ws  # or wherever your ROS 2 workspace is located
colcon build --packages-select humanoid_control
source install/setup.bash
```

### Step 2: Run the Publisher Node

In one terminal, run the publisher node:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run humanoid_control joint_command_publisher
```

You should see output similar to:
```
[INFO] [1699123456.789] [joint_command_publisher]: Publishing joint commands: [0.0, 0.09983341664682815, 0.0, 0.0, 0.19866933079506122, 0.0]
[INFO] [1699123456.889] [joint_command_publisher]: Publishing joint commands: [0.009999833334166664, 0.19933384615038052, 0.009999833334166664, 0.009999833334166664, 0.19933384615038052, 0.009999833334166664]
```

### Step 3: Run the Subscriber Node

In another terminal, run the subscriber node:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run humanoid_control sensor_subscriber
```

Note: The subscriber node in our example listens to the `sensor_data` topic, but the publisher publishes to `joint_commands`. To see actual communication, you would need to modify one of the nodes to use the same topic, or create a bridge node.

### Step 4: Using ROS 2 Tools to Verify Communication

You can also use ROS 2 command-line tools to verify communication:

```bash
# List all available topics
ros2 topic list

# Echo messages from a specific topic
ros2 topic echo /joint_commands std_msgs/msg/Float64MultiArray

# Check the publishers and subscribers for a topic
ros2 topic info /joint_commands
```

## Testing Service Communication

### Step 1: Run the Service Server

```bash
source ~/ros2_ws/install/setup.bash
ros2 run humanoid_control config_service
```

### Step 2: Call the Service from Another Terminal

```bash
source ~/ros2_ws/install/setup.bash
ros2 service call /configure_ai_control std_srvs/srv/SetBool "{data: true}"
```

You should see the service respond with success and a message.

## Testing Action Communication

### Step 1: Run the Action Server

```bash
source ~/ros2_ws/install/setup.bash
ros2 run humanoid_control behavior_action_server
```

### Step 2: Send an Action Goal from Another Terminal

```bash
source ~/ros2_ws/install/setup.bash
ros2 run humanoid_control behavior_action_client
```

You should see progress updates as the action executes.

## Expected Results

When all components are working correctly:
- Publisher nodes should continuously publish messages to their topics
- Subscriber nodes should receive and process messages from their subscribed topics
- Service calls should return appropriate responses
- Action goals should execute with feedback updates

## Troubleshooting

- If nodes don't appear to communicate, check that they're using the same topic/service/action names
- Verify that the message/service/action types match between publishers/subscribers and servers/clients
- Make sure all terminals have sourced the ROS 2 setup file
- Check for any error messages in the terminal output