import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
import numpy as np
import time


class AIBridge(Node):
    """
    AI Bridge Node: Connects AI agents to ROS 2 controllers.
    This node processes sensor data and generates control commands for humanoid robots.
    """

    def __init__(self):
        super().__init__('ai_bridge')

        # Store the latest sensor data
        self.latest_joint_states = None
        self.latest_processed_sensor_data = None
        self.latest_robot_state = None

        # Create subscribers for different sensor types
        self.joint_state_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Subscribe to processed sensor data from sensor processing node
        self.processed_sensor_subscription = self.create_subscription(
            Float64MultiArray,
            'processed_sensor_data',
            self.processed_sensor_callback,
            10
        )

        # Subscribe to estimated robot state from state estimation node
        self.robot_state_subscription = self.create_subscription(
            Float64MultiArray,
            'estimated_state',
            self.robot_state_callback,
            10
        )

        # Create publisher for joint commands
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray,
            'joint_commands',
            10
        )

        # Create publisher for behavior commands to behavior manager
        self.behavior_command_publisher = self.create_publisher(
            String,
            'behavior_command',
            10
        )

        # Create publisher for robot movement commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Timer for AI processing loop
        self.timer = self.create_timer(0.1, self.ai_processing_callback)  # 10 Hz

        self.get_logger().info('AI Bridge node initialized')

    def joint_state_callback(self, msg):
        """
        Callback for joint state messages.
        """
        self.latest_joint_states = msg
        self.get_logger().debug(f'Received joint states for {len(msg.name)} joints')

    def processed_sensor_callback(self, msg):
        """
        Callback for processed sensor data from sensor processing node.
        """
        self.latest_processed_sensor_data = msg.data
        self.get_logger().debug(f'Received processed sensor data with {len(msg.data)} values')

    def robot_state_callback(self, msg):
        """
        Callback for estimated robot state from state estimation node.
        """
        self.latest_robot_state = msg.data
        self.get_logger().debug(f'Received robot state with {len(msg.data)} values')

    def ai_processing_callback(self):
        """
        Main AI processing callback that runs at a fixed frequency.
        This simulates the AI decision-making process.
        """
        # Process the sensor data through a simple AI algorithm
        # Use the most recent data from all available sources
        control_commands = self.process_sensor_data()

        # Publish the control commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = control_commands
        self.joint_command_publisher.publish(cmd_msg)

        # Also potentially send behavior commands based on state
        if self.latest_robot_state is not None and len(self.latest_robot_state) > 6:
            # Check if robot is balanced (simplified check)
            balance_stable = self.latest_robot_state[6] if len(self.latest_robot_state) > 6 else 1.0
            if balance_stable < 0.5:  # Robot is not stable
                behavior_msg = String()
                behavior_msg.data = 'balance'
                self.behavior_command_publisher.publish(behavior_msg)
                self.get_logger().info('AI: Requesting balance behavior')
            else:
                # Default behavior when stable
                behavior_msg = String()
                behavior_msg.data = 'standing'
                self.behavior_command_publisher.publish(behavior_msg)

        self.get_logger().info(f'AI output: {control_commands[:3]}...')  # Show first 3 values

    def process_sensor_data(self):
        """
        Simple AI logic to process sensor data and generate control commands.
        In a real implementation, this would call a trained ML model.
        """
        # Use processed sensor data if available, otherwise use joint states
        if self.latest_processed_sensor_data is not None:
            # The processed sensor data contains joint positions, velocities, and IMU data
            # For this example, we'll use the first portion which should be joint-related
            sensor_data = self.latest_processed_sensor_data

            # Extract joint positions part (assuming first N values are joint positions)
            # This is a simplified approach - in reality, you'd know the exact structure
            num_joints = min(13, len(sensor_data) // 3)  # Assuming 3 values per joint (pos, vel, rel_pos)
            joint_positions = sensor_data[:num_joints] if len(sensor_data) >= num_joints else [0.0] * num_joints

        elif self.latest_joint_states is not None:
            joint_positions = list(self.latest_joint_states.position)
        else:
            # Default to zero positions if no data available
            joint_positions = [0.0] * 13  # Default humanoid joint count

        # Simple control logic: move to opposite position with some gain
        # This is a placeholder for actual AI processing
        control_commands = []
        for i, pos in enumerate(joint_positions):
            # Simple PD controller simulation
            desired_pos = 0.0  # Desired position (e.g., home position)
            error = desired_pos - pos
            control_cmd = 0.5 * error  # Simple proportional control

            # Add some variation based on joint index
            control_cmd += 0.1 * np.sin(time.time() + i)

            # Limit the control command
            control_cmd = max(-1.0, min(1.0, control_cmd))
            control_commands.append(control_cmd)

        # Ensure we have the right number of commands for all joints
        expected_joint_count = 13  # Adjust based on your humanoid model
        while len(control_commands) < expected_joint_count:
            control_commands.append(0.0)

        return control_commands


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the AI bridge node
    ai_bridge = AIBridge()

    try:
        # Start spinning the node to process callbacks
        rclpy.spin(ai_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        ai_bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()