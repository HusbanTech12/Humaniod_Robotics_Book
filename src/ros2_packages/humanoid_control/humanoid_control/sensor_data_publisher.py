#!/usr/bin/env python3
"""
Sensor Data Publisher Node

This node subscribes to raw sensor data from Gazebo simulation and processes it
into the custom SensorData message format for the digital twin system.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, Imu, Image, CameraInfo, PointCloud2
from geometry_msgs.msg import WrenchStamped
from humanoid_control.msg import SensorData
import numpy as np
import time
from cv_bridge import CvBridge
import message_filters


class SensorDataPublisher(Node):
    def __init__(self):
        super().__init__('sensor_data_publisher')

        # Declare parameters
        self.declare_parameter('publish_rate', 30)
        self.declare_parameter('use_sim_time', True)

        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value

        # Create publisher for processed sensor data
        self.sensor_data_pub = self.create_publisher(
            SensorData,
            'sensor_data_stream',
            10
        )

        # Create QoS profile for sensor data
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribe to various sensor topics from Gazebo
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/humanoid/scan',  # This matches the remapping in URDF
            self.lidar_callback,
            sensor_qos
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid/imu_sensor',  # This matches the sensor name in URDF
            self.imu_callback,
            sensor_qos
        )

        self.camera_sub = self.create_subscription(
            Image,
            '/humanoid/camera/rgb/image_raw',
            self.camera_callback,
            sensor_qos
        )

        self.depth_camera_sub = self.create_subscription(
            Image,
            '/humanoid/camera/depth/image_raw',
            self.depth_camera_callback,
            sensor_qos
        )

        self.left_ankle_ft_sub = self.create_subscription(
            WrenchStamped,
            '/humanoid/left_ankle/ft_sensor',
            self.left_ankle_ft_callback,
            sensor_qos
        )

        self.right_ankle_ft_sub = self.create_subscription(
            WrenchStamped,
            '/humanoid/right_ankle/ft_sensor',
            self.right_ankle_ft_callback,
            sensor_qos
        )

        self.right_wrist_ft_sub = self.create_subscription(
            WrenchStamped,
            '/humanoid/right_wrist/ft_sensor',
            self.right_wrist_ft_callback,
            sensor_qos
        )

        # Timer for publishing processed data
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_processed_data)

        # Storage for latest sensor data
        self.latest_sensor_data = {}
        self.cv_bridge = CvBridge()

        self.get_logger().info('Sensor Data Publisher node initialized')

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        try:
            sensor_data = SensorData()
            sensor_data.header = msg.header
            sensor_data.sensor_type = SensorData.LIDAR
            sensor_data.sensor_name = 'lidar_sensor'
            sensor_data.data_format = 'FLOAT32_ARRAY'

            # Convert laser scan ranges to bytes
            ranges_array = np.array(msg.ranges, dtype=np.float32)
            sensor_data.raw_data = ranges_array.tobytes()
            sensor_data.timestamp = float(len(msg.ranges))  # Store array length as a simple metadata

            self.latest_sensor_data['lidar'] = sensor_data
        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR data: {e}')

    def imu_callback(self, msg):
        """Process IMU data"""
        try:
            sensor_data = SensorData()
            sensor_data.header = msg.header
            sensor_data.sensor_type = SensorData.IMU
            sensor_data.sensor_name = 'imu_sensor'
            sensor_data.data_format = 'IMU_MESSAGE'

            # Store IMU data as a packed format
            imu_data = [
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
            ]
            imu_array = np.array(imu_data, dtype=np.float64)
            sensor_data.raw_data = imu_array.tobytes()
            sensor_data.timestamp = time.time()

            self.latest_sensor_data['imu'] = sensor_data
        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')

    def camera_callback(self, msg):
        """Process camera data"""
        try:
            sensor_data = SensorData()
            sensor_data.header = msg.header
            sensor_data.sensor_type = SensorData.CAMERA_RGB
            sensor_data.sensor_name = 'camera_rgb'
            sensor_data.data_format = 'IMAGE_MESSAGE'

            # Store image data
            sensor_data.raw_data = bytes(msg.data)
            sensor_data.timestamp = float(msg.width * msg.height)  # Store dimensions as metadata

            self.latest_sensor_data['camera_rgb'] = sensor_data
        except Exception as e:
            self.get_logger().error(f'Error processing camera data: {e}')

    def depth_camera_callback(self, msg):
        """Process depth camera data"""
        try:
            sensor_data = SensorData()
            sensor_data.header = msg.header
            sensor_data.sensor_type = SensorData.CAMERA_DEPTH
            sensor_data.sensor_name = 'camera_depth'
            sensor_data.data_format = 'IMAGE_MESSAGE'

            # Store depth image data
            sensor_data.raw_data = bytes(msg.data)
            sensor_data.timestamp = float(msg.width * msg.height)  # Store dimensions as metadata

            self.latest_sensor_data['camera_depth'] = sensor_data
        except Exception as e:
            self.get_logger().error(f'Error processing depth camera data: {e}')

    def left_ankle_ft_callback(self, msg):
        """Process left ankle force/torque data"""
        try:
            sensor_data = SensorData()
            sensor_data.header = msg.header
            sensor_data.sensor_type = SensorData.FORCE_TORQUE
            sensor_data.sensor_name = 'left_ankle_ft'
            sensor_data.data_format = 'WRENCH_MESSAGE'

            # Pack force and torque data
            ft_data = [
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
            ]
            ft_array = np.array(ft_data, dtype=np.float64)
            sensor_data.raw_data = ft_array.tobytes()
            sensor_data.timestamp = time.time()

            self.latest_sensor_data['left_ankle_ft'] = sensor_data
        except Exception as e:
            self.get_logger().error(f'Error processing left ankle FT data: {e}')

    def right_ankle_ft_callback(self, msg):
        """Process right ankle force/torque data"""
        try:
            sensor_data = SensorData()
            sensor_data.header = msg.header
            sensor_data.sensor_type = SensorData.FORCE_TORQUE
            sensor_data.sensor_name = 'right_ankle_ft'
            sensor_data.data_format = 'WRENCH_MESSAGE'

            # Pack force and torque data
            ft_data = [
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
            ]
            ft_array = np.array(ft_data, dtype=np.float64)
            sensor_data.raw_data = ft_array.tobytes()
            sensor_data.timestamp = time.time()

            self.latest_sensor_data['right_ankle_ft'] = sensor_data
        except Exception as e:
            self.get_logger().error(f'Error processing right ankle FT data: {e}')

    def right_wrist_ft_callback(self, msg):
        """Process right wrist force/torque data"""
        try:
            sensor_data = SensorData()
            sensor_data.header = msg.header
            sensor_data.sensor_type = SensorData.FORCE_TORQUE
            sensor_data.sensor_name = 'right_wrist_ft'
            sensor_data.data_format = 'WRENCH_MESSAGE'

            # Pack force and torque data
            ft_data = [
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
            ]
            ft_array = np.array(ft_data, dtype=np.float64)
            sensor_data.raw_data = ft_array.tobytes()
            sensor_data.timestamp = time.time()

            self.latest_sensor_data['right_wrist_ft'] = sensor_data
        except Exception as e:
            self.get_logger().error(f'Error processing right wrist FT data: {e}')

    def publish_processed_data(self):
        """Publish the latest sensor data"""
        for sensor_name, sensor_data in self.latest_sensor_data.items():
            try:
                # Add a small delay to avoid overwhelming the system
                sensor_data.header.frame_id = f"{sensor_name}_frame"
                self.sensor_data_pub.publish(sensor_data)
            except Exception as e:
                self.get_logger().error(f'Error publishing {sensor_name} data: {e}')


def main(args=None):
    rclpy.init(args=args)

    sensor_publisher = SensorDataPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        sensor_publisher.get_logger().info('Sensor Data Publisher interrupted by user')
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()