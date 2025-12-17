# Testing Multi-Node Communication in ROS 2

## Overview

This document explains how to test communication patterns between multiple nodes in your humanoid robotics system. Multi-node communication is essential for distributed robot control, where different nodes handle specialized tasks like sensor processing, state estimation, behavior management, and AI decision-making.

## Learning Objectives

After completing this tutorial, you will understand:
- How to verify communication between different node types
- How to test data flow through the system
- How to debug communication issues
- How to validate that nodes work together as a coordinated system

## Prerequisites

Before testing multi-node communication, ensure that:
1. ROS 2 Humble Hawksbill is installed and sourced
2. All packages (`humanoid_control`, `ai_bridge`) are built
3. You have multiple terminals available for running nodes
4. You understand basic ROS 2 concepts (nodes, topics, services, actions)

## Testing Communication Patterns

### 1. Topic-Based Communication

Topic-based communication is the most common pattern in ROS 2, enabling asynchronous data flow between publishers and subscribers.

#### Testing Joint State Communication

```bash
# Terminal 1: Run a joint state publisher (simulated sensor data)
ros2 run joint_state_publisher joint_state_publisher

# Terminal 2: Run the sensor processing node
ros2 run humanoid_control sensor_processing_node

# Terminal 3: Monitor the processed sensor data
ros2 topic echo /processed_sensor_data
```

#### Testing Command Communication

```bash
# Terminal 1: Run the AI bridge node
ros2 run ai_bridge ai_bridge

# Terminal 2: Run the behavior manager node
ros2 run humanoid_control behavior_manager_node

# Terminal 3: Send behavior commands and observe the system response
ros2 topic pub /behavior_command std_msgs/String "data: 'stand'"
```

### 2. Verifying Node Connectivity

#### List Active Nodes

```bash
# Check which nodes are currently running
ros2 node list
```

You should see nodes like:
- `/sensor_processing_node`
- `/state_estimation_node`
- `/behavior_manager_node`
- `/ai_bridge`

#### Check Topic Connections

```bash
# List all topics
ros2 topic list

# Check connections for a specific topic
ros2 topic info /joint_states
ros2 topic info /processed_sensor_data
ros2 topic info /behavior_command
```

### 3. Testing Data Flow

#### End-to-End Communication Test

Test the complete data flow from sensors to actuators:

```bash
# Terminal 1: Run the joint state publisher (simulated sensors)
ros2 run joint_state_publisher joint_state_publisher

# Terminal 2: Run the sensor processing node
ros2 run humanoid_control sensor_processing_node

# Terminal 3: Run the state estimation node
ros2 run humanoid_control state_estimation_node

# Terminal 4: Run the AI bridge node
ros2 run ai_bridge ai_bridge

# Terminal 5: Run the behavior manager node
ros2 run humanoid_control behavior_manager_node

# Terminal 6: Monitor the final control commands
ros2 topic echo /joint_commands
```

#### Monitoring Data Rates

Check the rate at which data flows through the system:

```bash
# Check the rate of joint state messages
ros2 topic hz /joint_states

# Check the rate of processed sensor data
ros2 topic hz /processed_sensor_data

# Check the rate of control commands
ros2 topic hz /joint_commands
```

### 4. Testing Behavior Coordination

#### Behavior Command Flow

Test how behavior commands flow through the system:

```bash
# Terminal 1: Run the AI bridge node
ros2 run ai_bridge ai_bridge

# Terminal 2: Run the behavior manager node
ros2 run humanoid_control behavior_manager_node

# Terminal 3: Send behavior commands and monitor responses
ros2 topic pub /behavior_command std_msgs/String "data: 'standing'" --times 1
ros2 topic pub /behavior_command std_msgs/String "data: 'walking'" --times 1
ros2 topic pub /behavior_command std_msgs/String "data: 'balance'" --times 1
```

#### State Feedback Loop

Verify that state information flows back to influence behavior:

```bash
# Terminal 1: Run the state estimation node
ros2 run humanoid_control state_estimation_node

# Terminal 2: Run the AI bridge node
ros2 run ai_bridge ai_bridge

# Terminal 3: Monitor the estimated state
ros2 topic echo /estimated_state
```

## Debugging Communication Issues

### 1. Common Communication Problems

#### Nodes Not Connecting

If nodes aren't communicating:
- Verify that nodes are running
- Check that topic names match exactly
- Ensure message types are compatible
- Confirm that nodes are in the same ROS domain

#### Message Loss

If messages are being lost:
- Check system resource usage (CPU, memory)
- Verify that processing nodes aren't blocking
- Consider adjusting QoS settings for reliability

#### Incorrect Data

If data is incorrect:
- Verify message structure and content
- Check that data is being processed correctly
- Ensure proper coordinate frame transformations

### 2. Debugging Tools

#### Using ros2 topic Commands

```bash
# Get detailed information about a topic
ros2 topic info /topic_name

# Echo messages with field selection
ros2 topic echo /joint_states --field position

# Test publishing to a topic
ros2 topic pub /test_topic std_msgs/String "data: 'test'"
```

#### Using ros2 node Commands

```bash
# Get information about a specific node
ros2 node info /node_name

# List parameters for a node
ros2 param list /node_name

# Get a specific parameter value
ros2 param get /node_name parameter_name
```

#### Using rqt Tools

```bash
# Launch rqt with various plugins
rqt

# Launch specific plugins
rqt_graph  # Visualize node connections
rqt_topic  # Monitor topics
rqt_plot   # Plot numeric values over time
```

## Performance Testing

### 1. Latency Measurement

Measure the time it takes for data to flow through the system:

```bash
# Add timestamps to your messages and calculate the difference
# between when data is published and when it's received
```

### 2. Throughput Testing

Test the system's ability to handle high message rates:

```bash
# Publish messages at high frequency and monitor for drops
ros2 topic pub /test_topic std_msgs/String "data: 'test'" --rate 100
```

### 3. Resource Usage

Monitor system resources during operation:

```bash
# Monitor CPU and memory usage
htop

# Monitor network usage (if applicable)
iftop
```

## Integration Testing

### 1. Complete System Test

Run all nodes together and verify the complete system works:

```bash
# Use the system launch file to start all nodes
ros2 launch humanoid_control humanoid_system.launch.py
```

### 2. Scenario Testing

Test specific scenarios that require coordination:

#### Balance Recovery Test
1. Start with the robot in a balanced state
2. Send a command that destabilizes the robot
3. Verify that the balance behavior is automatically engaged

#### Sensor Failure Simulation
1. Stop one of the sensor nodes
2. Verify that the system gracefully handles the missing data
3. Check that appropriate fallback behaviors are activated

## Expected Results

When multi-node communication is working correctly:
- Messages flow between nodes without loss
- Data is processed correctly at each stage
- Nodes respond appropriately to commands
- Behavior coordination works as expected
- No errors appear in the terminal output
- The system maintains stability and responsiveness

## Troubleshooting Checklist

### If Nodes Aren't Communicating

- [ ] Verify all nodes are running
- [ ] Check topic names match exactly
- [ ] Confirm message types are compatible
- [ ] Ensure nodes are in the same ROS domain
- [ ] Check for typos in topic names
- [ ] Verify QoS settings are compatible

### If Data Is Incorrect

- [ ] Check message structure and field types
- [ ] Verify data processing logic
- [ ] Ensure proper coordinate frame transformations
- [ ] Check for unit conversion issues
- [ ] Validate sensor calibration

### If System Is Unresponsive

- [ ] Check system resource usage
- [ ] Verify that no node is blocking
- [ ] Check for deadlocks or infinite loops
- [ ] Monitor message queue sizes
- [ ] Consider optimizing processing algorithms

## Best Practices

### 1. Design for Communication

- Use appropriate QoS settings for your use case
- Design message structures for efficient processing
- Consider message size and frequency
- Plan for graceful degradation when communication fails

### 2. Test Incrementally

- Test individual nodes before integration
- Verify communication between pairs of nodes
- Gradually add more nodes to the system
- Monitor system performance as complexity increases

### 3. Monitor Continuously

- Implement logging for debugging
- Use monitoring tools during operation
- Set up alerts for communication failures
- Regularly check system health

## Next Steps

Once multi-node communication is validated:
- Implement more sophisticated coordination algorithms
- Add redundancy for critical communication paths
- Optimize message rates and processing efficiency
- Implement advanced debugging and monitoring tools
- Document communication patterns for future reference

## Summary

Testing multi-node communication is crucial for ensuring that your humanoid robot system operates as an integrated whole. By systematically verifying data flow, testing coordination patterns, and debugging issues, you can create a robust distributed system that leverages the strengths of ROS 2's communication architecture.