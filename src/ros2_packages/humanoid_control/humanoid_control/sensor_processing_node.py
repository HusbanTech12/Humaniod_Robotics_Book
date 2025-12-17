import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Vector3
import numpy as np
from collections import deque
import time


class SensorProcessingNode(Node):
    """
    Sensor Processing Node: Handles IMU and joint data for the humanoid robot.
    This node processes raw sensor data and prepares it for AI algorithms.
    """

    def __init__(self):
        super().__init__('sensor_processing_node')

        # Store the latest sensor data
        self.latest_joint_states = None
        self.latest_imu_data = None

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

        # Create publishers for processed sensor data
        self.processed_sensor_publisher = self.create_publisher(
            Float64MultiArray,
            'processed_sensor_data',
            10
        )

        # Publisher for robot state information
        self.robot_state_publisher = self.create_publisher(
            Float64MultiArray,
            'robot_state',
            10
        )

        # Timer for processing loop
        self.processing_timer = self.create_timer(0.05, self.process_sensors)  # 20 Hz

        # Store historical data for filtering
        self.joint_history = deque(maxlen=10)
        self.imu_history = deque(maxlen=10)

        self.get_logger().info('Sensor Processing node initialized')

    def joint_state_callback(self, msg):
        """
        Callback for joint state messages.
        """
        self.latest_joint_states = msg
        self.joint_history.append({
            'position': list(msg.position),
            'velocity': list(msg.velocity),
            'effort': list(msg.effort),
            'timestamp': time.time()
        })

    def imu_callback(self, msg):
        """
        Callback for IMU messages.
        """
        self.latest_imu_data = msg
        self.imu_history.append({
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            'timestamp': time.time()
        })

    def process_sensors(self):
        """
        Main processing function that runs at a fixed frequency.
        Processes sensor data and publishes processed information.
        """
        if self.latest_joint_states is not None and self.latest_imu_data is not None:
            # Process joint data
            processed_joint_data = self.process_joint_data(self.latest_joint_states)

            # Process IMU data
            processed_imu_data = self.process_imu_data(self.latest_imu_data)

            # Combine all processed data
            combined_data = processed_joint_data + processed_imu_data

            # Create and publish processed sensor data message
            sensor_msg = Float64MultiArray()
            sensor_msg.data = combined_data
            self.processed_sensor_publisher.publish(sensor_msg)

            # Create and publish robot state message
            state_msg = Float64MultiArray()
            state_msg.data = self.calculate_robot_state(processed_joint_data, processed_imu_data)
            self.robot_state_publisher.publish(state_msg)

            self.get_logger().debug(f'Published processed sensor data with {len(combined_data)} values')

    def process_joint_data(self, joint_state):
        """
        Process joint state data to extract relevant information.
        """
        processed_data = []

        # Process joint positions
        for pos in joint_state.position:
            processed_data.append(pos)

        # Process joint velocities
        for vel in joint_state.velocity:
            processed_data.append(vel)

        # Calculate derived information (e.g., joint angles relative to neutral position)
        neutral_positions = [0.0] * len(joint_state.position)  # Assuming neutral position is 0
        for i, pos in enumerate(joint_state.position):
            processed_data.append(pos - neutral_positions[i])  # Relative position

        return processed_data

    def process_imu_data(self, imu_msg):
        """
        Process IMU data to extract relevant information.
        """
        processed_data = []

        # Extract orientation (as Euler angles from quaternion)
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        quat = [imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w]
        euler = self.quaternion_to_euler(quat)
        processed_data.extend(euler)

        # Extract angular velocity
        processed_data.append(imu_msg.angular_velocity.x)
        processed_data.append(imu_msg.angular_velocity.y)
        processed_data.append(imu_msg.angular_velocity.z)

        # Extract linear acceleration
        processed_data.append(imu_msg.linear_acceleration.x)
        processed_data.append(imu_msg.linear_acceleration.y)
        processed_data.append(imu_msg.linear_acceleration.z)

        return processed_data

    def calculate_robot_state(self, joint_data, imu_data):
        """
        Calculate higher-level robot state information.
        """
        state_data = []

        # Calculate center of mass estimate (simplified)
        # This is a placeholder - in a real system, you'd use the URDF and joint positions
        com_x = 0.0  # Placeholder
        com_y = 0.0  # Placeholder
        com_z = 0.5  # Placeholder (assuming robot is standing)
        state_data.extend([com_x, com_y, com_z])

        # Calculate balance stability (simplified)
        # Check if the center of mass is within the support polygon
        # This is a simplified check based on IMU data
        pitch = imu_data[1] if len(imu_data) > 1 else 0.0
        roll = imu_data[0] if len(imu_data) > 0 else 0.0
        balance_stable = abs(pitch) < 0.5 and abs(roll) < 0.5
        state_data.append(1.0 if balance_stable else 0.0)

        # Add other state information as needed
        state_data.append(len(joint_data))  # Number of joint values
        state_data.append(len(imu_data))    # Number of IMU values

        return state_data

    def quaternion_to_euler(self, quat):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        """
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the sensor processing node
    sensor_processing_node = SensorProcessingNode()

    try:
        # Start spinning the node to process callbacks
        rclpy.spin(sensor_processing_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        sensor_processing_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()