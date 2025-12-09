import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import time


class JointCommandPublisher(Node):
    """
    A simple publisher node that publishes joint commands for a humanoid robot.
    This serves as a basic example for learning ROS 2 concepts.
    """

    def __init__(self):
        super().__init__('joint_command_publisher')

        # Create a publisher for joint commands
        self.publisher = self.create_publisher(
            Float64MultiArray,
            'joint_commands',
            10
        )

        # Create a timer to publish messages at regular intervals
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter for generating different values
        self.i = 0

        self.get_logger().info('Joint Command Publisher node initialized')

    def timer_callback(self):
        # Create a message with joint command values
        msg = Float64MultiArray()

        # Generate example joint positions (in radians)
        # For a simple humanoid: [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]
        msg.data = [
            0.1 * math.sin(self.i * 0.1),      # left_hip
            0.2 * math.sin(self.i * 0.1),      # left_knee
            0.1 * math.sin(self.i * 0.1),      # left_ankle
            0.1 * math.sin(self.i * 0.1),      # right_hip
            0.2 * math.sin(self.i * 0.1),      # right_knee
            0.1 * math.sin(self.i * 0.1)       # right_ankle
        ]

        # Publish the message
        self.publisher.publish(msg)

        # Log the published data
        self.get_logger().info(f'Publishing joint commands: {msg.data}')

        # Increment counter
        self.i += 1


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the publisher node
    joint_command_publisher = JointCommandPublisher()

    try:
        # Start spinning the node to process callbacks
        rclpy.spin(joint_command_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        joint_command_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()