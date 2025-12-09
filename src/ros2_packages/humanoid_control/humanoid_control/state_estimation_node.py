import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from std_msgs.msg import Float64MultiArray
import numpy as np
from collections import deque
import time
from scipy.spatial.transform import Rotation as R


class StateEstimationNode(Node):
    """
    State Estimation Node: Fuses sensor data to estimate the robot's state.
    This node combines joint states, IMU data, and other sensors to provide
    an accurate estimate of the robot's position, orientation, velocity, and other state variables.
    """

    def __init__(self):
        super().__init__('state_estimation_node')

        # Robot state variables
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion (x, y, z, w)
        self.linear_velocity = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.joint_positions = []
        self.joint_velocities = []

        # Time tracking
        self.last_update_time = self.get_clock().now()

        # Create subscribers for sensor data
        self.joint_state_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.imu_subscription = self.create_subscription(
            Imu,
            'imu_sensor',
            self.imu_callback,
            10
        )

        # Create publishers for estimated state
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            'estimated_pose',
            10
        )

        self.twist_publisher = self.create_publisher(
            TwistStamped,
            'estimated_twist',
            10
        )

        self.state_publisher = self.create_publisher(
            Float64MultiArray,
            'estimated_state',
            10
        )

        # Timer for state estimation loop
        self.estimation_timer = self.create_timer(0.02, self.estimate_state)  # 50 Hz

        # Store historical data for filtering
        self.imu_history = deque(maxlen=20)
        self.joint_history = deque(maxlen=20)

        self.get_logger().info('State Estimation node initialized')

    def joint_state_callback(self, msg):
        """
        Callback for joint state messages.
        """
        self.joint_positions = list(msg.position)
        self.joint_velocities = list(msg.velocity)

        # Store in history for filtering
        self.joint_history.append({
            'position': list(msg.position),
            'velocity': list(msg.velocity),
            'timestamp': self.get_clock().now().nanoseconds
        })

    def imu_callback(self, msg):
        """
        Callback for IMU messages.
        """
        # Update orientation from IMU (this is a simplified approach)
        self.orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Update angular velocity from IMU
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Store in history for filtering
        self.imu_history.append({
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            'timestamp': self.get_clock().now().nanoseconds
        })

    def estimate_state(self):
        """
        Main state estimation function that runs at a fixed frequency.
        Fuses sensor data to estimate the robot's state.
        """
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9  # Convert to seconds
        self.last_update_time = current_time

        if dt > 0 and dt < 1.0:  # Valid time difference
            # Update position based on velocity and acceleration
            # This is a simplified integration - in practice, you'd use more sophisticated filtering
            imu_linear_acc = self.get_latest_imu_linear_acc()
            if imu_linear_acc is not None:
                # Transform acceleration to world frame and integrate
                world_acc = self.transform_vector_to_world_frame(imu_linear_acc, self.orientation)

                # Update velocity (simple integration)
                self.linear_velocity += world_acc * dt

                # Update position (simple integration)
                self.position += self.linear_velocity * dt

        # Publish estimated state
        self.publish_estimated_state()

    def get_latest_imu_linear_acc(self):
        """
        Get the latest linear acceleration from IMU history.
        """
        if self.imu_history:
            latest = self.imu_history[-1]
            return np.array(latest['linear_acceleration'])
        return None

    def transform_vector_to_world_frame(self, vector, orientation_quat):
        """
        Transform a vector from the IMU frame to the world frame using the orientation quaternion.
        """
        # Convert quaternion to rotation matrix
        r = R.from_quat(orientation_quat)  # Note: scipy expects [x, y, z, w] format
        # Apply rotation to the vector
        world_vector = r.apply(vector)
        return world_vector

    def publish_estimated_state(self):
        """
        Publish the estimated state to various topics.
        """
        # Publish estimated pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'  # or 'odom' depending on your setup
        pose_msg.pose.position.x = float(self.position[0])
        pose_msg.pose.position.y = float(self.position[1])
        pose_msg.pose.position.z = float(self.position[2])
        pose_msg.pose.orientation.x = float(self.orientation[0])
        pose_msg.pose.orientation.y = float(self.orientation[1])
        pose_msg.pose.orientation.z = float(self.orientation[2])
        pose_msg.pose.orientation.w = float(self.orientation[3])

        self.pose_publisher.publish(pose_msg)

        # Publish estimated twist
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'  # or appropriate frame
        twist_msg.twist.linear.x = float(self.linear_velocity[0])
        twist_msg.twist.linear.y = float(self.linear_velocity[1])
        twist_msg.twist.linear.z = float(self.linear_velocity[2])
        twist_msg.twist.angular.x = float(self.angular_velocity[0])
        twist_msg.twist.angular.y = float(self.angular_velocity[1])
        twist_msg.twist.angular.z = float(self.angular_velocity[2])

        self.twist_publisher.publish(twist_msg)

        # Publish comprehensive state as Float64MultiArray
        state_msg = Float64MultiArray()
        state_data = []

        # Position (3 values)
        state_data.extend([float(x) for x in self.position])

        # Orientation quaternion (4 values)
        state_data.extend([float(x) for x in self.orientation])

        # Linear velocity (3 values)
        state_data.extend([float(x) for x in self.linear_velocity])

        # Angular velocity (3 values)
        state_data.extend([float(x) for x in self.angular_velocity])

        # Joint positions (variable number)
        state_data.extend([float(x) for x in self.joint_positions])

        # Joint velocities (variable number)
        state_data.extend([float(x) for x in self.joint_velocities])

        # Add a timestamp
        state_data.append(float(self.get_clock().now().nanoseconds / 1e9))

        state_msg.data = state_data
        self.state_publisher.publish(state_msg)

        self.get_logger().debug(f'Published estimated state with {len(state_data)} values')


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the state estimation node
    state_estimation_node = StateEstimationNode()

    try:
        # Start spinning the node to process callbacks
        rclpy.spin(state_estimation_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        state_estimation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()