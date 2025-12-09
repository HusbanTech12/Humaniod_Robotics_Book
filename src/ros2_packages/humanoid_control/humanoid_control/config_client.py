import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
import sys


class ConfigClient(Node):
    """
    A service client that calls the configuration service to change robot settings.
    This demonstrates the service client pattern in ROS 2.
    """

    def __init__(self):
        super().__init__('config_client')

        # Create a client for the AI control configuration service
        self.cli = self.create_client(SetBool, 'configure_ai_control')

        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Configuration service not available, waiting again...')

        self.req = SetBool.Request()

        self.get_logger().info('Configuration client node initialized')

    def send_request(self, enable_ai_control):
        """
        Send a request to enable or disable AI control.
        """
        self.req.data = enable_ai_control

        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

        if self.future.result() is not None:
            response = self.future.result()
            self.get_logger().info(f'Result: {response.success}, {response.message}')
            return response
        else:
            self.get_logger().error('Exception while calling service: %r' % (self.future.exception(),))
            return None


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the client node
    config_client = ConfigClient()

    # Check command line arguments
    if len(sys.argv) != 2:
        print('Usage: ros2 run humanoid_control config_client [enable|disable]')
        sys.exit(1)

    # Parse the argument
    if sys.argv[1].lower() == 'enable':
        enable_ai = True
    elif sys.argv[1].lower() == 'disable':
        enable_ai = False
    else:
        print('Usage: ros2 run humanoid_control config_client [enable|disable]')
        sys.exit(1)

    # Send the request
    response = config_client.send_request(enable_ai)

    if response is not None:
        print(f'Service call result: {response.success}')
        print(f'Message: {response.message}')
    else:
        print('Service call failed')

    # Clean up
    config_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()