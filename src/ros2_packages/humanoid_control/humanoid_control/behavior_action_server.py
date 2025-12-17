import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class BehaviorActionServer(Node):
    """
    An action server that executes complex behaviors for the humanoid robot.
    This demonstrates the action communication pattern in ROS 2, which is ideal
    for long-running tasks with feedback and status reporting.
    """

    def __init__(self):
        super().__init__('behavior_action_server')

        # Create an action server for executing joint trajectories
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'execute_behavior_trajectory',
            execute_callback=self.execute_callback,
            callback_group=None,
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback
        )

        self.get_logger().info('Behavior action server initialized')

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """
        Accept or reject a goal.
        """
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle):
        """
        Handle an accepted goal.
        """
        self.get_logger().info('Goal accepted, executing...')
        goal_handle.execute()

    def cancel_callback(self, goal_handle):
        """
        Accept or reject a cancel request.
        """
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """
        Execute the requested behavior.
        """
        self.get_logger().info('Executing behavior...')

        # Get the trajectory from the goal
        trajectory = goal_handle.request.trajectory
        n_points = len(trajectory.points)

        # Execute each point in the trajectory
        for i, point in enumerate(trajectory.points):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return FollowJointTrajectory.Result()

            # Simulate executing the trajectory point
            # In a real robot, this would involve commanding the joints
            time.sleep(0.1)  # Simulate time to reach the position

            # Publish feedback
            feedback_msg = FollowJointTrajectory.Feedback()
            feedback_msg.joint_names = trajectory.joint_names
            feedback_msg.actual.positions = point.positions
            feedback_msg.actual.velocities = point.velocities
            feedback_msg.desired = point
            feedback_msg.error.positions = [0.0] * len(point.positions)  # Simulated perfect execution

            # Calculate progress percentage
            progress = float(i + 1) / float(n_points) * 100.0
            feedback_msg.progress = progress

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Progress: {progress:.2f}%')

        # Check if there was a cancel request during execution
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self.get_logger().info('Goal canceled during execution')
            return FollowJointTrajectory.Result()

        # Return success result
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = 'Behavior executed successfully'

        self.get_logger().info('Behavior execution completed successfully')
        return result


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the action server node
    behavior_action_server = BehaviorActionServer()

    try:
        # Start spinning the node to handle action requests
        rclpy.spin(behavior_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        behavior_action_server.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()