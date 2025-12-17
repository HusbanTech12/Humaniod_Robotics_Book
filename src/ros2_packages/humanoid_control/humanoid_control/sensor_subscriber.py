import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class SensorSubscriber(Node):
    """
    A simple subscriber node that receives sensor data from a humanoid robot.
    This serves as a basic example for learning ROS 2 concepts.
    """

    def __init__(self):
        super().__init__('sensor_subscriber')

        # Create a subscription to the sensor data topic
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'sensor_data',
            self.listener_callback,
            10
        )

        # Make sure the subscription is properly created
        self.subscription  # prevent unused variable warning

        self.get_logger().info('Sensor Subscriber node initialized')

    def listener_callback(self, msg):
        # Process the received sensor data
        self.get_logger().info(f'Received sensor data: {msg.data}')

        # In a real application, you would process the sensor data here
        # For example, you might:
        # - Update the robot's state estimate
        # - Trigger specific behaviors based on sensor readings
        # - Log data for analysis
        # - Send the data to an AI algorithm for processing


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the subscriber node
    sensor_subscriber = SensorSubscriber()

    try:
        # Start spinning the node to process callbacks
        rclpy.spin(sensor_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        sensor_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()