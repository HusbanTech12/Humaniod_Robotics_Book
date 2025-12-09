# Multi-Node Communication: Nodes, Topics, Services, and Actions

## Overview

In ROS 2, communication between nodes is fundamental to creating distributed robotic systems. This document covers the core communication patterns: nodes (processes), topics (publish/subscribe), services (request/response), and actions (goal-oriented with feedback). Understanding these patterns is crucial for developing complex humanoid robotics systems.

## Communication Patterns Overview

### 1. Nodes

Nodes are the fundamental building blocks of a ROS 2 system. Each node is a process that performs computation and communicates with other nodes through messages.

**Key Characteristics:**
- Each node runs in its own process
- Nodes can be written in different programming languages
- Nodes must be part of a graph managed by a ROS 2 daemon
- Nodes can be started and stopped independently

**Creating a Node:**
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

### 2. Topics (Publish/Subscribe)

Topics enable asynchronous, many-to-many communication between nodes. A node can publish messages to a topic, and other nodes can subscribe to that topic to receive the messages.

**Key Characteristics:**
- Unidirectional data flow from publishers to subscribers
- Asynchronous communication
- Multiple publishers and subscribers can use the same topic
- Data is sent as messages of specific types

**Creating a Publisher:**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'topic_name', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
```

**Creating a Subscriber:**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

### 3. Services (Request/Response)

Services enable synchronous, request-response communication between nodes. One node provides a service, and other nodes can send requests to that service and receive responses.

**Key Characteristics:**
- Synchronous communication
- Request-response pattern
- One-to-one communication (one client, one server)
- Request and response message types are defined in service files

**Creating a Service Server:**
```python
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceServer(Node):
    def __init__(self):
        super().__init__('service_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

**Creating a Service Client:**
```python
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

### 4. Actions (Goal-Oriented with Feedback)

Actions are a goal-oriented communication pattern that includes feedback and status information. They are ideal for long-running tasks where you want to track progress.

**Key Characteristics:**
- Goal-oriented communication
- Includes feedback during execution
- Provides status information
- Supports preemption (canceling goals)
- Three message types: goal, feedback, and result

**Creating an Action Server:**
```python
import time
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info('Returning result: {0}'.format(result.sequence))
        return result
```

**Creating an Action Client:**
```python
import time
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.sequence))
```

## Communication Patterns in Humanoid Robotics

### 1. Sensor Data Flow (Topics)

Sensors generate continuous streams of data that need to be processed by multiple nodes:

```
Sensors → Sensor Processing Node → AI Bridge → Behavior Manager
         ↓
         State Estimation Node
```

**Example Implementation:**
```python
# Sensor Processing Node
class SensorProcessingNode(Node):
    def __init__(self):
        super().__init__('sensor_processing_node')

        # Subscribe to raw sensor data
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_cb, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu_data', self.imu_cb, 10)

        # Publish processed data
        self.processed_pub = self.create_publisher(Float64MultiArray, 'processed_sensor_data', 10)

    def joint_cb(self, msg):
        # Process joint data
        processed_data = self.process_joint_data(msg)
        self.processed_pub.publish(processed_data)
```

### 2. Control Command Flow (Topics)

Control commands flow from high-level decision makers to low-level controllers:

```
AI Bridge → Behavior Manager → Joint Controllers → Actuators
```

### 3. Behavior Coordination (Services/Actions)

Complex behaviors often require request/response or goal-oriented communication:

```python
# Service for requesting specific behaviors
class BehaviorService(Node):
    def __init__(self):
        super().__init__('behavior_service')
        self.srv = self.create_service(Trigger, 'execute_behavior', self.behavior_callback)

    def behavior_callback(self, request, response):
        # Execute the requested behavior
        success = self.execute_behavior(request.command)
        response.success = success
        response.message = "Behavior executed" if success else "Behavior failed"
        return response
```

### 4. State Monitoring (Topics)

Robot state information flows continuously through the system:

```python
# State Estimation Node
class StateEstimationNode(Node):
    def __init__(self):
        super().__init__('state_estimation_node')

        # Subscribe to sensor data
        self.sensor_sub = self.create_subscription(Float64MultiArray, 'processed_sensor_data', self.sensor_cb, 10)

        # Publish estimated state
        self.state_pub = self.create_publisher(PoseStamped, 'estimated_pose', 10)
        self.state_pub2 = self.create_publisher(TwistStamped, 'estimated_twist', 10)
```

## Quality of Service (QoS) Settings

QoS settings allow you to tune communication behavior for different use cases:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# For critical control commands (reliable delivery)
reliable_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# For sensor data (best effort, high frequency)
sensor_qos = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# Create publisher with specific QoS
self.control_pub = self.create_publisher(Float64MultiArray, 'joint_commands', reliable_qos)
```

## Best Practices for Multi-Node Communication

### 1. Naming Conventions

Use consistent naming conventions for topics, services, and actions:
- Use descriptive names that indicate purpose
- Group related topics with common prefixes (e.g., `/sensors/joint_states`, `/sensors/imu`)
- Use underscores for multi-word names
- Follow ROS 2 naming conventions

### 2. Message Design

Design messages for efficient communication:
- Include only necessary data in messages
- Use appropriate data types (avoid strings for numeric data)
- Consider message size for bandwidth-constrained systems
- Design for extensibility

### 3. Error Handling

Implement robust error handling:
- Check for valid message data
- Handle missing or corrupted messages gracefully
- Implement timeouts for synchronous communication
- Provide fallback behaviors when communication fails

### 4. Performance Optimization

Optimize communication for performance:
- Use appropriate QoS settings for each use case
- Minimize message size for high-frequency topics
- Use appropriate update rates for different data types
- Consider message compression for large data

## Debugging Multi-Node Communication

### Common Tools

```bash
# List all active nodes
ros2 node list

# List all active topics
ros2 topic list

# Check topic information
ros2 topic info /topic_name

# Monitor topic data
ros2 topic echo /topic_name

# Check message rate
ros2 topic hz /topic_name

# Visualize the node graph
rqt_graph
```

### Debugging Tips

1. **Start Simple**: Test communication between two nodes before adding complexity
2. **Use Standard Messages**: When possible, use standard ROS 2 message types
3. **Monitor Message Rates**: Ensure nodes are publishing at expected rates
4. **Check Data Validity**: Verify that message content is as expected
5. **Use Logging**: Add detailed logging to understand data flow

## Summary

Multi-node communication is the backbone of ROS 2 systems. By understanding and properly implementing nodes, topics, services, and actions, you can create sophisticated distributed systems for humanoid robotics. Each communication pattern has its place in the system architecture, and choosing the right pattern for each use case is crucial for system performance and reliability.