#!/usr/bin/env python3
"""
Sensor Data Verification Script

This script verifies that sensor data is being published to the correct ROS 2 topics
and that the data format is as expected.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, Imu, Image, PointCloud2
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String
from builtin_interfaces.msg import Time
import time


class SensorDataVerifier(Node):
    def __init__(self):
        super().__init__('sensor_data_verifier')

        # Create QoS profile for sensor data
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Topics to verify
        self.topics_to_verify = {
            '/humanoid/scan': {'type': LaserScan, 'received': False, 'count': 0},
            '/humanoid/imu_sensor': {'type': Imu, 'received': False, 'count': 0},
            '/humanoid/camera/rgb/image_raw': {'type': Image, 'received': False, 'count': 0},
            '/humanoid/camera/depth/image_raw': {'type': Image, 'received': False, 'count': 0},
            '/humanoid/left_ankle/ft_sensor': {'type': WrenchStamped, 'received': False, 'count': 0},
            '/humanoid/right_ankle/ft_sensor': {'type': WrenchStamped, 'received': False, 'count': 0},
            '/humanoid/right_wrist/ft_sensor': {'type': WrenchStamped, 'received': False, 'count': 0},
        }

        # Subscribe to all sensor topics
        self.subscribers = {}
        for topic_name, topic_info in self.topics_to_verify.items():
            self.subscribers[topic_name] = self.create_subscription(
                topic_info['type'],
                topic_name,
                lambda msg, t=topic_name: self.generic_callback(msg, t),
                sensor_qos
            )

        # Timer to periodically check and report
        self.timer = self.create_timer(2.0, self.check_status)

        # Timer to stop after a certain period
        self.stop_timer = self.create_timer(15.0, self.stop_verification)

        self.start_time = time.time()
        self.verification_complete = False

        self.get_logger().info('Sensor Data Verifier started - checking for data on expected topics...')

    def generic_callback(self, msg, topic_name):
        """Generic callback to handle messages from any sensor topic"""
        if topic_name in self.topics_to_verify:
            self.topics_to_verify[topic_name]['received'] = True
            self.topics_to_verify[topic_name]['count'] += 1
            if self.topics_to_verify[topic_name]['count'] == 1:  # Log first reception
                self.get_logger().info(f'Data received on topic: {topic_name}')

    def check_status(self):
        """Check and report the status of sensor data reception"""
        if self.verification_complete:
            return

        all_received = all(info['received'] for info in self.topics_to_verify.values())
        total_received = sum(info['count'] for info in self.topics_to_verify.values())

        self.get_logger().info(f'Verification status after {time.time() - self.start_time:.1f}s: '
                              f'{sum(1 for info in self.topics_to_verify.values() if info["received"])}/'
                              f'{len(self.topics_to_verify)} topics receiving data, '
                              f'total messages: {total_received}')

        if all_received:
            self.get_logger().info('SUCCESS: All sensor topics are receiving data!')
            self.verification_complete = True
            self.print_topic_summary()

    def print_topic_summary(self):
        """Print a summary of all verified topics"""
        self.get_logger().info('\n=== Sensor Data Verification Summary ===')
        for topic_name, info in self.topics_to_verify.items():
            status = 'RECEIVED' if info['received'] else 'MISSING'
            self.get_logger().info(f'{topic_name}: {status} ({info["count"]} messages)')

    def stop_verification(self):
        """Stop the verification and print final results"""
        self.get_logger().info('\n=== Final Verification Results ===')
        all_received = all(info['received'] for info in self.topics_to_verify.values())

        for topic_name, info in self.topics_to_verify.items():
            status = 'RECEIVED' if info['received'] else 'MISSING'
            self.get_logger().info(f'{topic_name}: {status} ({info["count"]} messages)')

        if all_received:
            self.get_logger().info('\nVERIFICATION SUCCESS: All expected sensor topics are publishing data correctly!')
        else:
            missing_topics = [name for name, info in self.topics_to_verify.items() if not info['received']]
            self.get_logger().error(f'\nVERIFICATION FAILED: Missing data on topics: {missing_topics}')

        # Shutdown after verification
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    verifier = SensorDataVerifier()

    try:
        rclpy.spin(verifier)
    except KeyboardInterrupt:
        verifier.get_logger().info('Sensor Data Verifier interrupted by user')
        verifier.print_topic_summary()
        verifier.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()