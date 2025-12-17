import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue
from rcl_interfaces.msg import SetParametersResult


class ConfigService(Node):
    """
    A service server that handles configuration changes for the humanoid robot.
    This demonstrates the service communication pattern in ROS 2.
    """

    def __init__(self):
        super().__init__('config_service')

        # Create a service for enabling/disabling AI control
        self.srv = self.create_service(
            SetBool,
            'configure_ai_control',
            self.configure_ai_control_callback
        )

        # Create a service for setting parameters
        self.param_srv = self.create_service(
            SetParameters,
            'set_robot_parameters',
            self.set_parameters_callback
        )

        # Initialize some configuration parameters
        self.ai_control_enabled = False
        self.walking_speed = 0.5  # default walking speed
        self.joint_torque_limit = 10.0  # default torque limit

        self.get_logger().info('Configuration service node initialized')

    def configure_ai_control_callback(self, request, response):
        """
        Callback for the AI control configuration service.
        """
        # Set the AI control flag based on the request
        self.ai_control_enabled = request.data

        # Prepare response
        response.success = True
        if self.ai_control_enabled:
            response.message = "AI control has been enabled"
            self.get_logger().info("AI control enabled")
        else:
            response.message = "AI control has been disabled"
            self.get_logger().info("AI control disabled")

        return response

    def set_parameters_callback(self, request, response):
        """
        Callback for setting robot parameters.
        """
        result = SetParametersResult()

        for param in request.parameters:
            if param.name == 'walking_speed':
                if param.value.type == ParameterValue().TYPE_DOUBLE:
                    self.walking_speed = param.value.double_value
                    self.get_logger().info(f'Set walking speed to: {self.walking_speed}')
                else:
                    result.successful = False
                    result.reason = 'walking_speed must be a double value'
                    response.results.append(result)
                    return response
            elif param.name == 'joint_torque_limit':
                if param.value.type == ParameterValue().TYPE_DOUBLE:
                    self.joint_torque_limit = param.value.double_value
                    self.get_logger().info(f'Set joint torque limit to: {self.joint_torque_limit}')
                else:
                    result.successful = False
                    result.reason = 'joint_torque_limit must be a double value'
                    response.results.append(result)
                    return response
            else:
                result.successful = False
                result.reason = f'Unknown parameter: {param.name}'
                response.results.append(result)
                return response

        result.successful = True
        result.reason = 'Parameters set successfully'
        response.results.append(result)

        return response


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the service node
    config_service = ConfigService()

    try:
        # Start spinning the node to handle service requests
        rclpy.spin(config_service)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        config_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()