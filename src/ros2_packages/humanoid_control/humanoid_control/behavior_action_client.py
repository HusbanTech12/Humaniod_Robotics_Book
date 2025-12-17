import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class BehaviorActionClient(Node):
    """
    An action client that sends goals to the behavior action server.
    This demonstrates the action client pattern in ROS 2.
    """

    def __init__(self):
        super().__init__('behavior_action_client')

        # Create an action client for the behavior trajectory execution
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            'execute_behavior_trajectory'
        )

        self.get_logger().info('Behavior action client initialized')

    def send_goal(self):
        """
        Send a goal to the action server.
        """
        # Wait for the action server to be available
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        # Create a trajectory goal
        goal_msg = FollowJointTrajectory.Goal()

        # Define joint names for a simple humanoid (left leg)
        goal_msg.trajectory.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        # Create trajectory points
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Initial position
        point1.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point1.time_from_start = Duration(sec=1)  # Reach this point in 1 second

        point2 = JointTrajectoryPoint()
        point2.positions = [0.1, 0.2, 0.05, 0.1, 0.2, 0.05]  # Move to new position
        point2.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point2.time_from_start = Duration(sec=2)  # Reach this point in 2 seconds

        point3 = JointTrajectoryPoint()
        point3.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Return to initial
        point3.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point3.time_from_start = Duration(sec=3)  # Reach this point in 3 seconds

        goal_msg.trajectory.points = [point1, point2, point3]

        # Send the goal
        self.get_logger().info('Sending goal to execute walking pattern...')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handle the goal response from the server.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get the result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """
        Handle the result from the action server.
        """
        result = future.result().result
        self.get_logger().info(f'Result: {result.error_string}')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        """
        Handle feedback from the action server.
        """
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback - Progress: {feedback.progress:.2f}%')


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the action client node
    behavior_action_client = BehaviorActionClient()

    # Send a goal
    behavior_action_client.send_goal()

    try:
        # Start spinning the node to handle responses
        rclpy.spin(behavior_action_client)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        behavior_action_client.destroy_node()


if __name__ == '__main__':
    main()