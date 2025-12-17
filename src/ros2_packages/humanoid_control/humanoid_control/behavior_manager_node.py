import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
from enum import Enum


class BehaviorState(Enum):
    IDLE = 1
    WALKING = 2
    STANDING = 3
    SITTING = 4
    GESTURING = 5
    BALANCING = 6


class BehaviorManagerNode(Node):
    """
    Behavior Manager Node: Coordinates high-level behaviors for the humanoid robot.
    This node manages different robot behaviors and orchestrates the necessary
    actions to execute them safely and effectively.
    """

    def __init__(self):
        super().__init__('behavior_manager_node')

        # Current behavior state
        self.current_behavior = BehaviorState.IDLE
        self.previous_behavior = None

        # Create subscribers for sensor data and commands
        self.sensor_subscription = self.create_subscription(
            Float64MultiArray,
            'processed_sensor_data',
            self.sensor_callback,
            10
        )

        self.command_subscription = self.create_subscription(
            String,
            'behavior_command',
            self.command_callback,
            10
        )

        # Create publishers for control commands
        self.joint_trajectory_publisher = self.create_publisher(
            JointTrajectory,
            'joint_trajectory',
            10
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Create action client for joint trajectory execution
        self.joint_trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            'joint_trajectory_controller/follow_joint_trajectory'
        )

        # Timer for behavior execution
        self.behavior_timer = self.create_timer(0.1, self.execute_behavior)  # 10 Hz

        # Store robot state
        self.current_sensor_data = None
        self.joint_names = [
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'neck_joint'
        ]

        self.get_logger().info('Behavior Manager node initialized')

    def sensor_callback(self, msg):
        """
        Callback for processed sensor data.
        """
        self.current_sensor_data = msg.data

    def command_callback(self, msg):
        """
        Callback for behavior commands.
        """
        command = msg.data.lower()

        if command == 'stand':
            self.request_behavior_change(BehaviorState.STANDING)
        elif command == 'walk':
            self.request_behavior_change(BehaviorState.WALKING)
        elif command == 'sit':
            self.request_behavior_change(BehaviorState.SITTING)
        elif command == 'gesture':
            self.request_behavior_change(BehaviorState.GESTURING)
        elif command == 'balance':
            self.request_behavior_change(BehaviorState.BALANCING)
        elif command == 'idle' or command == 'stop':
            self.request_behavior_change(BehaviorState.IDLE)
        else:
            self.get_logger().warn(f'Unknown behavior command: {command}')

    def request_behavior_change(self, new_behavior):
        """
        Request a change to a new behavior.
        """
        if self.current_behavior != new_behavior:
            self.get_logger().info(f'Requesting behavior change from {self.current_behavior} to {new_behavior}')

            # Check if transition is valid
            if self.is_valid_transition(self.current_behavior, new_behavior):
                self.previous_behavior = self.current_behavior
                self.current_behavior = new_behavior
                self.get_logger().info(f'Behavior changed to: {self.current_behavior}')
            else:
                self.get_logger().warn(f'Invalid behavior transition from {self.current_behavior} to {new_behavior}')

    def is_valid_transition(self, from_behavior, to_behavior):
        """
        Check if a behavior transition is valid.
        """
        # Define valid transitions
        valid_transitions = {
            BehaviorState.IDLE: [BehaviorState.STANDING, BehaviorState.GESTURING],
            BehaviorState.STANDING: [BehaviorState.IDLE, BehaviorState.WALKING, BehaviorState.SITTING, BehaviorState.GESTURING, BehaviorState.BALANCING],
            BehaviorState.WALKING: [BehaviorState.STANDING, BehaviorState.BALANCING],
            BehaviorState.SITTING: [BehaviorState.STANDING],
            BehaviorState.GESTURING: [BehaviorState.STANDING, BehaviorState.IDLE],
            BehaviorState.BALANCING: [BehaviorState.STANDING, BehaviorState.WALKING]
        }

        if from_behavior in valid_transitions:
            return to_behavior in valid_transitions[from_behavior]
        else:
            # If from_behavior is not in the dictionary, only allow transition to IDLE
            return to_behavior == BehaviorState.IDLE

    def execute_behavior(self):
        """
        Main behavior execution function that runs at a fixed frequency.
        """
        if self.current_sensor_data is None:
            return

        # Execute the current behavior
        if self.current_behavior == BehaviorState.IDLE:
            self.execute_idle_behavior()
        elif self.current_behavior == BehaviorState.STANDING:
            self.execute_standing_behavior()
        elif self.current_behavior == BehaviorState.WALKING:
            self.execute_walking_behavior()
        elif self.current_behavior == BehaviorState.SITTING:
            self.execute_sitting_behavior()
        elif self.current_behavior == BehaviorState.GESTURING:
            self.execute_gesturing_behavior()
        elif self.current_behavior == BehaviorState.BALANCING:
            self.execute_balancing_behavior()

    def execute_idle_behavior(self):
        """
        Execute idle behavior - minimal movement, ready to receive commands.
        """
        # In idle state, maintain a neutral position
        neutral_positions = [0.0] * len(self.joint_names)
        self.publish_joint_trajectory(self.joint_names, [neutral_positions], [2.0])  # Hold for 2 seconds

    def execute_standing_behavior(self):
        """
        Execute standing behavior - maintain balance in standing position.
        """
        # Standing position - more natural stance
        standing_positions = [
            0.0,    # left_shoulder_joint
            0.0,    # left_elbow_joint
            0.0,    # left_wrist_joint
            0.0,    # right_shoulder_joint
            0.0,    # right_elbow_joint
            0.0,    # right_wrist_joint
            0.0,    # left_hip_joint
            0.0,    # left_knee_joint
            0.0,    # left_ankle_joint
            0.0,    # right_hip_joint
            0.0,    # right_knee_joint
            0.0,    # right_ankle_joint
            0.0     # neck_joint
        ]
        self.publish_joint_trajectory(self.joint_names, [standing_positions], [1.0])

    def execute_walking_behavior(self):
        """
        Execute walking behavior - generate walking pattern.
        """
        # This is a simplified walking pattern
        # In a real implementation, you would use more sophisticated gait generation

        # Define a simple walking trajectory with multiple points
        trajectory_points = []
        time_points = []

        # Create a simple walking pattern (this is highly simplified)
        for i in range(5):  # 5 steps
            # Left leg forward
            positions1 = [
                0.1,   # left_shoulder_joint
                0.2,   # left_elbow_joint
                0.0,   # left_wrist_joint
                -0.1,  # right_shoulder_joint
                -0.2,  # right_elbow_joint
                0.0,   # right_wrist_joint
                0.1,   # left_hip_joint
                0.3,   # left_knee_joint
                -0.2,  # left_ankle_joint
                -0.1,  # right_hip_joint
                -0.1,  # right_knee_joint
                0.0,   # right_ankle_joint
                0.0    # neck_joint
            ]
            trajectory_points.append(positions1)
            time_points.append(0.5 * (i + 1))

            # Right leg forward
            positions2 = [
                -0.1,  # left_shoulder_joint
                -0.2,  # left_elbow_joint
                0.0,   # left_wrist_joint
                0.1,   # right_shoulder_joint
                0.2,   # right_elbow_joint
                0.0,   # right_wrist_joint
                -0.1,  # left_hip_joint
                -0.1,  # left_knee_joint
                0.0,   # left_ankle_joint
                0.1,   # right_hip_joint
                0.3,   # right_knee_joint
                -0.2,  # right_ankle_joint
                0.0    # neck_joint
            ]
            trajectory_points.append(positions2)
            time_points.append(0.5 * (i + 1) + 0.5)

        self.publish_joint_trajectory(self.joint_names, trajectory_points, time_points)

    def execute_sitting_behavior(self):
        """
        Execute sitting behavior - move to sitting position.
        """
        sitting_positions = [
            0.5,    # left_shoulder_joint
            1.0,    # left_elbow_joint
            0.2,    # left_wrist_joint
            -0.5,   # right_shoulder_joint
            -1.0,   # right_elbow_joint
            -0.2,   # right_wrist_joint
            -1.0,   # left_hip_joint
            1.5,    # left_knee_joint
            -0.5,   # left_ankle_joint
            -1.0,   # right_hip_joint
            1.5,    # right_knee_joint
            -0.5,   # right_ankle_joint
            0.1     # neck_joint
        ]
        self.publish_joint_trajectory(self.joint_names, [sitting_positions], [2.0])

    def execute_gesturing_behavior(self):
        """
        Execute gesturing behavior - perform simple gestures.
        """
        # Define a simple waving gesture
        gesture_positions = [
            0.5,    # left_shoulder_joint - raise arm
            1.0,    # left_elbow_joint - bend elbow
            0.0,    # left_wrist_joint
            0.0,    # right_shoulder_joint
            0.0,    # right_elbow_joint
            0.0,    # right_wrist_joint
            0.0,    # left_hip_joint
            0.0,    # left_knee_joint
            0.0,    # left_ankle_joint
            0.0,    # right_hip_joint
            0.0,    # right_knee_joint
            0.0,    # right_ankle_joint
            0.2     # neck_joint - look slightly up
        ]
        self.publish_joint_trajectory(self.joint_names, [gesture_positions], [1.0])

    def execute_balancing_behavior(self):
        """
        Execute balancing behavior - adjust posture to maintain balance.
        """
        # This would use sensor feedback to maintain balance
        # For now, just maintain a stable standing position
        balance_positions = [
            0.05,   # left_shoulder_joint
            0.0,    # left_elbow_joint
            0.0,    # left_wrist_joint
            -0.05,  # right_shoulder_joint
            0.0,    # right_elbow_joint
            0.0,    # right_wrist_joint
            0.05,   # left_hip_joint
            0.0,    # left_knee_joint
            -0.05,  # left_ankle_joint
            -0.05,  # right_hip_joint
            0.0,    # right_knee_joint
            0.05,   # right_ankle_joint
            0.0     # neck_joint
        ]
        self.publish_joint_trajectory(self.joint_names, [balance_positions], [0.5])

    def publish_joint_trajectory(self, joint_names, positions_list, time_from_start_list):
        """
        Publish a joint trajectory message.
        """
        msg = JointTrajectory()
        msg.joint_names = joint_names

        for positions, time_from_start in zip(positions_list, time_from_start_list):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start = Duration(sec=int(time_from_start), nanosec=int((time_from_start % 1) * 1e9))
            msg.points.append(point)

        self.joint_trajectory_publisher.publish(msg)

    def get_behavior_status(self):
        """
        Get the current behavior status as a string.
        """
        return self.current_behavior.name.lower()


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the behavior manager node
    behavior_manager_node = BehaviorManagerNode()

    try:
        # Start spinning the node to process callbacks
        rclpy.spin(behavior_manager_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        behavior_manager_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()