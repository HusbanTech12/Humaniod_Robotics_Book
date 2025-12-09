# ROS 2 Architecture: Nodes, Topics, Services, and Actions

## Overview

ROS 2 uses a distributed architecture where computation is spread across many processes (potentially running on different machines) that interact by passing messages. The main concepts are nodes, topics, services, and actions.

## Nodes

A **node** is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 system. Each node runs independently and communicates with other nodes through messages.

### Key Characteristics:
- Each node runs in its own process
- Nodes can be written in different programming languages
- Nodes must be part of a graph managed by a ROS 2 daemon
- Nodes can be started and stopped independently

### Creating a Node:
In Python, nodes inherit from the `rclpy.Node` class:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        # Node initialization code here

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Topics and Message Passing

**Topics** enable asynchronous, many-to-many communication between nodes. A node can publish messages to a topic, and other nodes can subscribe to that topic to receive the messages.

### Key Characteristics:
- Unidirectional data flow from publishers to subscribers
- Asynchronous communication
- Multiple publishers and subscribers can use the same topic
- Data is sent as messages of specific types

### Example:
```python
# Publisher
publisher = self.create_publisher(String, 'topic_name', 10)

# Subscriber
subscription = self.create_subscription(
    String,
    'topic_name',
    self.callback,
    10
)
```

## Services

**Services** enable synchronous, request-response communication between nodes. One node provides a service, and other nodes can send requests to that service and receive responses.

### Key Characteristics:
- Synchronous communication
- Request-response pattern
- One-to-one communication (one client, one server)
- Request and response message types are defined in service files

### Example:
```python
# Service server
srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

# Service client
client = self.create_client(AddTwoInts, 'add_two_ints')
```

## Actions

**Actions** are a goal-oriented communication pattern that includes feedback and status information. They are ideal for long-running tasks where you want to track progress.

### Key Characteristics:
- Goal-oriented communication
- Includes feedback during execution
- Provides status information
- Supports preemption (canceling goals)
- Three message types: goal, feedback, and result

### Example:
```python
# Action server
self._action_server = ActionServer(
    self,
    Fibonacci,
    'fibonacci',
    self.execute_callback
)

# Action client
self._action_client = ActionClient(self, Fibonacci, 'fibonacci')
```

## Quality of Service (QoS)

ROS 2 includes Quality of Service settings that allow you to tune communication behavior for different use cases:

- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local (for late-joining subscribers)
- **History**: Keep all messages vs. keep last N messages
- **Deadline**: Maximum time between consecutive messages

## Communication Patterns for Humanoid Robotics

Different communication patterns are appropriate for different aspects of humanoid robot control:

- **Topics**: Sensor data streaming, joint state publishing, command publishing
- **Services**: Configuration changes, calibration, one-time commands
- **Actions**: Walking patterns, manipulation sequences, navigation goals

## Lifecycle Nodes

For more complex systems, ROS 2 provides lifecycle nodes that have explicit states (unconfigured, inactive, active, finalized) and transitions between them. This is useful for humanoid robots where components need to be initialized, activated, and deactivated in a controlled manner.

## Summary

Understanding these core concepts is essential for building humanoid robot systems with ROS 2. The distributed architecture allows for modular, maintainable robot software where different components can be developed and tested independently.