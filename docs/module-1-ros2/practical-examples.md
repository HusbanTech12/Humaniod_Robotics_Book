# Practical Examples: Implementing ROS 2 for Humanoid Robotics

## Overview

This document provides practical, hands-on examples of implementing ROS 2 concepts for humanoid robotics. Each example builds upon the previous ones, creating a complete system for controlling a humanoid robot with AI integration.

## Example 1: Basic Publisher/Subscriber

Let's start with the most fundamental ROS 2 concept: publishing and subscribing to topics.

### Publisher Node (Joint Command Publisher)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math


class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        self.publisher = self.create_publisher(Float64MultiArray, 'joint_commands', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Float64MultiArray()
        # Generate example joint positions (oscillating pattern)
        msg.data = [
            0.1 * math.sin(self.i * 0.1),      # left_hip
            0.2 * math.sin(self.i * 0.1),      # left_knee
            0.1 * math.sin(self.i * 0.1),      # left_ankle
            0.1 * math.sin(self.i * 0.1),      # right_hip
            0.2 * math.sin(self.i * 0.1),      # right_knee
            0.1 * math.sin(self.i * 0.1)       # right_ankle
        ]
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing joint commands: {msg.data}')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    joint_command_publisher = JointCommandPublisher()
    rclpy.spin(joint_command_publisher)
    joint_command_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Subscriber Node (Sensor Subscriber)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class SensorSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_subscriber')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'sensor_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received sensor data: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    sensor_subscriber = SensorSubscriber()
    rclpy.spin(sensor_subscriber)
    sensor_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 2: Service Implementation

Now let's implement a service for configuring the robot's behavior.

### Service Server (Configuration Service)

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class ConfigService(Node):
    def __init__(self):
        super().__init__('config_service')
        self.srv = self.create_service(SetBool, 'configure_ai_control', self.configure_ai_control_callback)
        self.ai_control_enabled = False

    def configure_ai_control_callback(self, request, response):
        self.ai_control_enabled = request.data
        response.success = True
        if self.ai_control_enabled:
            response.message = "AI control enabled"
            self.get_logger().info("AI control enabled")
        else:
            response.message = "AI control disabled"
            self.get_logger().info("AI control disabled")
        return response


def main(args=None):
    rclpy.init(args=args)
    config_service = ConfigService()
    rclpy.spin(config_service)
    config_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Service Client

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
import sys


class ConfigClient(Node):
    def __init__(self):
        super().__init__('config_client')
        self.cli = self.create_client(SetBool, 'configure_ai_control')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = SetBool.Request()

    def send_request(self, enable_ai_control):
        self.req.data = enable_ai_control
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)
    config_client = ConfigClient()

    if len(sys.argv) != 2:
        print('Usage: ros2 run package_name config_client [enable|disable]')
        sys.exit(1)

    enable_ai = sys.argv[1].lower() == 'enable'
    response = config_client.send_request(enable_ai)
    print(f'Response: {response.success}, {response.message}')

    config_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 3: Action Implementation

Let's create an action for executing complex behaviors like walking patterns.

### Action Server (Behavior Action Server)

```python
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class BehaviorActionServer(Node):
    def __init__(self):
        super().__init__('behavior_action_server')
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'execute_behavior_trajectory',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing behavior...')

        trajectory = goal_handle.request.trajectory
        n_points = len(trajectory.points)

        for i, point in enumerate(trajectory.points):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return FollowJointTrajectory.Result()

            # Simulate executing the trajectory point
            time.sleep(0.1)

            # Publish feedback
            feedback_msg = FollowJointTrajectory.Feedback()
            feedback_msg.joint_names = trajectory.joint_names
            feedback_msg.actual.positions = point.positions
            feedback_msg.actual.velocities = point.velocities
            feedback_msg.desired = point
            feedback_msg.error.positions = [0.0] * len(point.positions)

            progress = float(i + 1) / float(n_points) * 100.0
            feedback_msg.progress = progress

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Progress: {progress:.2f}%')

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self.get_logger().info('Goal canceled during execution')
            return FollowJointTrajectory.Result()

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = 'Behavior executed successfully'

        self.get_logger().info('Behavior execution completed successfully')
        return result


def main(args=None):
    rclpy.init(args=args)
    behavior_action_server = BehaviorActionServer()
    rclpy.spin(behavior_action_server)
    behavior_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 4: Complete AI Integration System

Now let's combine all concepts into a complete AI integration system.

### AI Bridge Node with All Communication Patterns

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.client import Client
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
from control_msgs.action import FollowJointTrajectory
import numpy as np
import time


class CompleteAIBridge(Node):
    """
    Complete AI Bridge Node that demonstrates all communication patterns.
    """

    def __init__(self):
        super().__init__('complete_ai_bridge')

        # Store sensor data
        self.latest_joint_states = None
        self.latest_processed_sensor_data = None

        # Publishers
        self.joint_command_publisher = self.create_publisher(Float64MultiArray, 'joint_commands', 10)
        self.behavior_command_publisher = self.create_publisher(String, 'behavior_command', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.joint_state_subscription = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.processed_sensor_subscription = self.create_subscription(Float64MultiArray, 'processed_sensor_data', self.processed_sensor_callback, 10)

        # Service client for configuration
        self.config_client = self.create_client(SetBool, 'configure_ai_control')
        while not self.config_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Config service not available, waiting again...')

        # Action client for complex behaviors
        self.behavior_action_client = ActionClient(self, FollowJointTrajectory, 'execute_behavior_trajectory')

        # Timer for AI processing
        self.timer = self.create_timer(0.1, self.ai_processing_callback)

        self.get_logger().info('Complete AI Bridge node initialized')

    def joint_state_callback(self, msg):
        self.latest_joint_states = msg
        self.get_logger().debug(f'Received joint states for {len(msg.name)} joints')

    def processed_sensor_callback(self, msg):
        self.latest_processed_sensor_data = msg.data
        self.get_logger().debug(f'Received processed sensor data with {len(msg.data)} values')

    def ai_processing_callback(self):
        if self.latest_joint_states is not None:
            # Process sensor data and generate control commands
            control_commands = self.process_sensor_data()

            # Publish control commands
            cmd_msg = Float64MultiArray()
            cmd_msg.data = control_commands
            self.joint_command_publisher.publish(cmd_msg)

            # Send behavior command based on state
            behavior_msg = String()
            if self.is_robot_unstable():
                behavior_msg.data = 'balance'
            else:
                behavior_msg.data = 'standing'
            self.behavior_command_publisher.publish(behavior_msg)

            self.get_logger().info(f'AI output: {control_commands[:3]}...')

    def process_sensor_data(self):
        """Process sensor data to generate control commands."""
        if self.latest_joint_states is not None:
            positions = list(self.latest_joint_states.position)
            control_commands = []

            for i, pos in enumerate(positions):
                # Simple PD controller
                desired_pos = 0.0
                error = desired_pos - pos
                control_cmd = 0.5 * error + 0.1 * np.sin(time.time() + i)
                control_cmd = max(-1.0, min(1.0, control_cmd))
                control_commands.append(control_cmd)

            return control_commands
        else:
            return [0.0] * 13  # Default for humanoid

    def is_robot_unstable(self):
        """Simple check for robot stability."""
        # This would be more sophisticated in a real implementation
        return False

    def send_config_request(self, enable_ai):
        """Send a configuration request via service."""
        req = SetBool.Request()
        req.data = enable_ai
        future = self.config_client.call_async(req)
        return future

    def send_behavior_goal(self, trajectory_points):
        """Send a behavior goal via action."""
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint1', 'joint2', 'joint3']  # Example joint names

        for point_data in trajectory_points:
            point = JointTrajectoryPoint()
            point.positions = point_data['positions']
            point.velocities = point_data.get('velocities', [0.0] * len(point_data['positions']))
            point.time_from_start = Duration(sec=1)
            goal_msg.trajectory.points.append(point)

        self.behavior_action_client.wait_for_server()
        return self.behavior_action_client.send_goal_async(goal_msg)


def main(args=None):
    rclpy.init(args=args)
    ai_bridge = CompleteAIBridge()

    try:
        rclpy.spin(ai_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        ai_bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 5: Launch File for Complete System

Create a launch file that brings up the complete system:

```python
# launch/complete_humanoid_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    launch_visualization_arg = DeclareLaunchArgument(
        'launch_visualization',
        default_value='true',
        description='Launch visualization if true'
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
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Sensor subscriber node
    sensor_subscriber = Node(
        package='humanoid_control',
        executable='sensor_subscriber',
        name='sensor_subscriber',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Configuration service
    config_service = Node(
        package='humanoid_control',
        executable='config_service',
        name='config_service',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # AI bridge node
    ai_bridge = Node(
        package='ai_bridge',
        executable='complete_ai_bridge',
        name='complete_ai_bridge',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Behavior action server
    behavior_action_server = Node(
        package='humanoid_control',
        executable='behavior_action_server',
        name='behavior_action_server',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
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

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Visualization nodes (conditional)
    joint_state_publisher_gui = Node(
        condition=IfCondition(LaunchConfiguration('launch_visualization')),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    rviz2 = Node(
        condition=IfCondition(LaunchConfiguration('launch_visualization')),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d',
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'rviz',
                'humanoid_config.rviz'
            ])
        ],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        launch_visualization_arg,
        set_use_sim_time,
        joint_command_publisher,
        sensor_subscriber,
        config_service,
        ai_bridge,
        behavior_action_server,
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        rviz2
    ])
```

## Example 6: Running the Complete System

### Building and Running

1. **Build the packages:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select humanoid_control ai_bridge
   source install/setup.bash
   ```

2. **Run the complete system:**
   ```bash
   ros2 launch humanoid_control complete_humanoid_system.launch.py
   ```

3. **Test individual components:**
   ```bash
   # In separate terminals, test specific functionality
   ros2 topic pub /behavior_command std_msgs/String "data: 'standing'"
   ros2 service call /configure_ai_control std_srvs/srv/SetBool "{data: true}"
   ros2 topic echo /joint_commands
   ```

## Example 7: Testing and Validation

### Unit Testing Example

```python
# test/test_ai_bridge.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from ai_bridge.complete_ai_bridge import CompleteAIBridge
from std_msgs.msg import Float64MultiArray


class TestCompleteAIBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = CompleteAIBridge()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_node_initialization(self):
        """Test that the node initializes properly."""
        self.assertIsNotNone(self.node.joint_command_publisher)
        self.assertIsNotNone(self.node.behavior_command_publisher)
        self.assertIsNotNone(self.node.timer)

    def test_process_sensor_data(self):
        """Test the sensor data processing function."""
        # Set up mock joint states
        from sensor_msgs.msg import JointState
        mock_joint_state = JointState()
        mock_joint_state.position = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]

        self.node.latest_joint_states = mock_joint_state

        # Process the data
        commands = self.node.process_sensor_data()

        # Check that we got the right number of commands
        self.assertEqual(len(commands), len(mock_joint_state.position))

        # Check that commands are within expected range
        for cmd in commands:
            self.assertGreaterEqual(cmd, -1.0)
            self.assertLessEqual(cmd, 1.0)


if __name__ == '__main__':
    unittest.main()
```

## Best Practices Demonstrated

1. **Proper Node Structure**: All nodes inherit from `rclpy.node.Node`
2. **Resource Management**: Proper cleanup with `destroy_node()`
3. **Error Handling**: Check for service availability before calling
4. **Logging**: Use `self.get_logger()` for informative messages
5. **Parameter Usage**: Use parameters for configuration
6. **QoS Considerations**: Appropriate QoS settings for different data types
7. **Modular Design**: Separate concerns into different nodes
8. **Launch Files**: Organize system startup with launch files

## Troubleshooting Common Issues

### 1. Node Not Connecting to ROS 2 Network
- Ensure `ros2 daemon` is running: `ros2 daemon start`
- Check ROS_DOMAIN_ID environment variable
- Verify network configuration

### 2. Topic Communication Issues
- Use `ros2 topic list` to verify topics exist
- Use `ros2 topic info /topic_name` to check publishers/subscribers
- Check message types match between publishers and subscribers

### 3. Service/Action Timeout
- Verify service/action servers are running
- Check for typos in service/action names
- Ensure proper timing between client and server

## Next Steps

After mastering these examples:
1. Implement more sophisticated AI algorithms
2. Add sensor fusion techniques
3. Implement advanced control algorithms
4. Add simulation with Gazebo
5. Test with real hardware
6. Optimize performance for real-time operation

These practical examples provide a solid foundation for implementing ROS 2 systems for humanoid robotics, demonstrating all major communication patterns and system organization principles.