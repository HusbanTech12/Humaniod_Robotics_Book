# AI Agent to ROS 2 Bridge Tutorial

## Overview

This tutorial demonstrates how to connect AI agents to ROS 2 controllers using the AI Bridge pattern. This approach enables AI algorithms to interact with robotic systems through ROS 2's communication infrastructure.

## Learning Objectives

By the end of this tutorial, you will understand:
- How to design an AI bridge node that connects AI agents to ROS 2
- How to process sensor data and generate control commands
- How to handle timing and synchronization between AI and robot systems
- Best practices for AI-ROS 2 integration

## The AI Bridge Pattern

The AI Bridge pattern separates AI processing from ROS 2 communication, providing a clean interface between the two systems:

```
┌─────────────┐    Sensor     ┌──────────────┐    Control    ┌─────────────┐
│             │    Data       │              │    Commands   │             │
│ AI Agent    │ ────────────▶ │ AI Bridge    │ ────────────▶ │ Robot       │
│ (Python)    │               │ Node (rclpy) │               │ Controllers │
│             │ ◀──────────── │              │ ◀──────────── │             │
│             │   Feedback    │              │   Status      │             │
└─────────────┘               └──────────────┘               └─────────────┘
```

## Implementation Steps

### 1. Create the AI Bridge Node

The AI Bridge node serves as the intermediary between AI algorithms and ROS 2:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class AIBridge(Node):
    def __init__(self):
        super().__init__('ai_bridge')

        # Subscribers for sensor data
        self.sensor_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.sensor_callback,
            10
        )

        # Publishers for control commands
        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            'joint_commands',
            10
        )

        # Timer for AI processing
        self.timer = self.create_timer(0.1, self.ai_processing_callback)  # 10 Hz

    def sensor_callback(self, msg):
        # Store sensor data for AI processing
        self.latest_sensor_data = msg

    def ai_processing_callback(self):
        if self.latest_sensor_data is not None:
            # Process sensor data with AI algorithm
            control_commands = self.process_with_ai(self.latest_sensor_data)

            # Publish control commands
            cmd_msg = Float64MultiArray()
            cmd_msg.data = control_commands
            self.command_publisher.publish(cmd_msg)
```

### 2. Integrate AI Models

The AI bridge can work with various types of AI models:

```python
def process_with_ai(self, sensor_data):
    # Convert ROS message to AI-friendly format
    processed_data = self.preprocess_sensor_data(sensor_data)

    # Apply AI algorithm (example with simple control logic)
    ai_output = self.apply_control_logic(processed_data)

    # Convert to ROS message format
    return self.postprocess_ai_output(ai_output)

def apply_control_logic(self, data):
    # Placeholder for actual AI model
    # In practice, this would call your ML model:
    # return self.ml_model.predict(data)

    # Simple example: move to center position
    return [0.0] * len(data)  # Return zeros for all joints
```

### 3. Handle Timing and Synchronization

Ensure proper timing between AI processing and robot control:

```python
def __init__(self):
    # ... other initialization ...

    # Set processing frequency based on robot requirements
    self.ai_frequency = 10.0  # Hz
    self.timer = self.create_timer(
        1.0 / self.ai_frequency,
        self.ai_processing_callback
    )
```

## Advanced Patterns

### Asynchronous AI Processing

For computationally expensive AI models, use separate threads:

```python
import threading
import queue

class AIBridge(Node):
    def __init__(self):
        # ... initialization ...

        # Queue for sensor data
        self.sensor_queue = queue.Queue(maxsize=1)

        # Start AI processing thread
        self.ai_thread = threading.Thread(target=self.ai_worker)
        self.ai_thread.daemon = True
        self.ai_thread.start()

    def sensor_callback(self, msg):
        try:
            self.sensor_queue.put_nowait(msg)
        except queue.Full:
            # Drop old data if queue is full
            pass

    def ai_worker(self):
        while rclpy.ok():
            try:
                sensor_msg = self.sensor_queue.get(timeout=1.0)
                ai_output = self.process_with_ai(sensor_msg)
                # Publish result back to main thread
                # (use threading-safe publishing mechanism)
            except queue.Empty:
                continue
```

### Error Handling and Safety

Implement robust error handling:

```python
def ai_processing_callback(self):
    try:
        if self.latest_sensor_data is not None:
            control_commands = self.process_with_ai(self.latest_sensor_data)
            cmd_msg = Float64MultiArray()
            cmd_msg.data = control_commands
            self.command_publisher.publish(cmd_msg)
    except Exception as e:
        self.get_logger().error(f'AI processing error: {e}')
        # Publish safe default commands
        safe_commands = self.get_safe_default_commands()
        cmd_msg = Float64MultiArray()
        cmd_msg.data = safe_commands
        self.command_publisher.publish(cmd_msg)
```

## Integration with Different AI Frameworks

### TensorFlow/Keras

```python
import tensorflow as tf

class AIBridge(Node):
    def __init__(self):
        # Load trained model
        self.ml_model = tf.keras.models.load_model('path/to/model')

    def process_with_ai(self, sensor_data):
        # Preprocess sensor data
        input_tensor = self.preprocess(sensor_data)

        # Run inference
        prediction = self.ml_model(input_tensor)

        # Post-process and return
        return self.postprocess(prediction)
```

### PyTorch

```python
import torch

class AIBridge(Node):
    def __init__(self):
        # Load trained model
        self.ml_model = torch.load('path/to/model.pth')
        self.ml_model.eval()

    def process_with_ai(self, sensor_data):
        with torch.no_grad():
            # Preprocess sensor data
            input_tensor = self.preprocess(sensor_data)

            # Run inference
            prediction = self.ml_model(input_tensor)

            # Post-process and return
            return self.postprocess(prediction)
```

## Testing the AI Bridge

### Unit Testing

Test the AI bridge components separately:

```python
import unittest
from ai_bridge import AIBridge

class TestAIBridge(unittest.TestCase):
    def setUp(self):
        self.ai_bridge = AIBridge()

    def test_sensor_callback(self):
        # Create mock sensor data
        mock_sensor_data = JointState()
        mock_sensor_data.position = [0.1, 0.2, 0.3]

        # Call callback
        self.ai_bridge.sensor_callback(mock_sensor_data)

        # Verify data was stored
        self.assertIsNotNone(self.ai_bridge.latest_sensor_data)
```

### Integration Testing

Test the complete AI-robot interaction:

```bash
# Terminal 1: Start the AI bridge
ros2 run ai_bridge ai_bridge

# Terminal 2: Simulate sensor data
ros2 topic pub /joint_states sensor_msgs/JointState '{position: [0.1, 0.2, 0.3]}'

# Terminal 3: Monitor control commands
ros2 topic echo /joint_commands
```

## Best Practices

1. **Separation of Concerns**: Keep AI logic separate from ROS 2 communication
2. **Error Handling**: Always include fallback behaviors for AI failures
3. **Timing**: Match AI processing frequency to robot control requirements
4. **Safety**: Implement safety limits on AI-generated commands
5. **Testing**: Test both AI algorithms and ROS 2 integration separately
6. **Monitoring**: Include logging and monitoring for debugging

## Conclusion

The AI Bridge pattern provides a robust way to connect AI agents to ROS 2 systems. By following these patterns and best practices, you can create reliable AI-powered robotic systems that leverage both the flexibility of Python AI libraries and the robustness of ROS 2 communication infrastructure.