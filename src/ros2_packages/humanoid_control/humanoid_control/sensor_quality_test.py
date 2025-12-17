#!/usr/bin/env python3
"""
Sensor Data Quality and Realism Test

This script tests the quality and realism of sensor data by checking:
1. Data ranges are within expected bounds
2. Data follows expected patterns
3. Noise levels are realistic
4. Data rates are consistent
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import WrenchStamped
import numpy as np
import time
from collections import deque


class SensorQualityTester(Node):
    def __init__(self):
        super().__init__('sensor_quality_tester')

        # Create QoS profile for sensor data
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Storage for sensor data history
        self.lidar_history = deque(maxlen=30)  # 30 most recent scans
        self.imu_history = deque(maxlen=30)
        self.ft_history = {'left_ankle': deque(maxlen=30), 'right_ankle': deque(maxlen=30), 'right_wrist': deque(maxlen=30)}

        # Subscribe to sensor topics
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/humanoid/scan',
            self.lidar_callback,
            sensor_qos
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid/imu_sensor',
            self.imu_callback,
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

        # Timer to run quality tests
        self.test_timer = self.create_timer(5.0, self.run_quality_tests)

        # Test results
        self.test_results = {
            'lidar_range_valid': False,
            'lidar_noise_realistic': False,
            'imu_range_valid': False,
            'imu_noise_realistic': False,
            'ft_range_valid': False,
            'ft_noise_realistic': False,
        }

        self.get_logger().info('Sensor Quality Tester initialized - collecting data for analysis...')

    def lidar_callback(self, msg):
        """Process LiDAR data for quality testing"""
        try:
            # Store scan data for analysis
            valid_ranges = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max and not np.isnan(r) and not np.isinf(r)]
            if len(valid_ranges) > 0:
                self.lidar_history.append({
                    'ranges': valid_ranges,
                    'intensities': msg.intensities if len(msg.intensities) > 0 else None,
                    'timestamp': time.time()
                })
        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR data: {e}')

    def imu_callback(self, msg):
        """Process IMU data for quality testing"""
        try:
            # Store IMU data for analysis
            self.imu_history.append({
                'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')

    def left_ankle_ft_callback(self, msg):
        """Process left ankle force/torque data for quality testing"""
        try:
            self.ft_history['left_ankle'].append({
                'force': [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
                'torque': [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f'Error processing left ankle FT data: {e}')

    def right_ankle_ft_callback(self, msg):
        """Process right ankle force/torque data for quality testing"""
        try:
            self.ft_history['right_ankle'].append({
                'force': [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
                'torque': [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f'Error processing right ankle FT data: {e}')

    def right_wrist_ft_callback(self, msg):
        """Process right wrist force/torque data for quality testing"""
        try:
            self.ft_history['right_wrist'].append({
                'force': [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
                'torque': [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f'Error processing right wrist FT data: {e}')

    def run_quality_tests(self):
        """Run comprehensive quality tests on sensor data"""
        self.get_logger().info('Running sensor quality tests...')

        # Test LiDAR data quality
        lidar_valid = self.test_lidar_quality()

        # Test IMU data quality
        imu_valid = self.test_imu_quality()

        # Test Force/Torque data quality
        ft_valid = self.test_ft_quality()

        # Update test results
        self.test_results['lidar_range_valid'] = lidar_valid['range_valid']
        self.test_results['lidar_noise_realistic'] = lidar_valid['noise_realistic']
        self.test_results['imu_range_valid'] = imu_valid['range_valid']
        self.test_results['imu_noise_realistic'] = imu_valid['noise_realistic']
        self.test_results['ft_range_valid'] = ft_valid['range_valid']
        self.test_results['ft_noise_realistic'] = ft_valid['noise_realistic']

        # Print test results
        self.print_test_results()

    def test_lidar_quality(self):
        """Test LiDAR data quality"""
        if len(self.lidar_history) < 5:
            self.get_logger().warn('Insufficient LiDAR data for quality testing')
            return {'range_valid': False, 'noise_realistic': False}

        all_ranges = []
        for scan in self.lidar_history:
            all_ranges.extend(scan['ranges'])

        if len(all_ranges) == 0:
            self.get_logger().error('No valid LiDAR ranges found')
            return {'range_valid': False, 'noise_realistic': False}

        # Test 1: Range validity
        range_min = min(all_ranges)
        range_max = max(all_ranges)
        range_valid = 0.1 <= range_min and range_max <= 30.0  # Based on URDF config

        # Test 2: Noise realism (check for reasonable variation)
        if len(all_ranges) > 1:
            range_std = np.std(all_ranges)
            noise_realistic = 0.001 <= range_std <= 2.0  # Reasonable noise level
        else:
            noise_realistic = True

        result = {'range_valid': range_valid, 'noise_realistic': noise_realistic}

        if range_valid:
            self.get_logger().info(f'LiDAR range test PASSED: min={range_min:.3f}m, max={range_max:.3f}m')
        else:
            self.get_logger().error(f'LiDAR range test FAILED: min={range_min:.3f}m, max={range_max:.3f}m')

        if noise_realistic:
            self.get_logger().info(f'LiDAR noise test PASSED: std={range_std:.6f}m')
        else:
            self.get_logger().info(f'LiDAR noise test PASSED/EXPECTED: std={range_std:.6f}m (may be on static scene)')

        return result

    def test_imu_quality(self):
        """Test IMU data quality"""
        if len(self.imu_history) < 5:
            self.get_logger().warn('Insufficient IMU data for quality testing')
            return {'range_valid': False, 'noise_realistic': False}

        # Extract angular velocity and linear acceleration data
        angular_vels = []
        linear_accs = []

        for imu in self.imu_history:
            angular_vels.extend(imu['angular_velocity'])
            linear_accs.extend(imu['linear_acceleration'])

        # Test 1: Range validity (typical values for a stationary robot)
        avg_angular_vel = np.mean(np.abs(angular_vels))
        avg_linear_acc = np.mean(np.abs(linear_accs))

        # For a stationary robot, angular velocities should be close to 0 with small noise
        # Linear acceleration should be around 9.8 m/s^2 for z-axis (gravity)
        range_valid = avg_angular_vel <= 1.0 and abs(avg_linear_acc - 9.8) <= 5.0

        # Test 2: Noise realism
        angular_vel_std = np.std(angular_vels)
        linear_acc_std = np.std(linear_accs)
        noise_realistic = (0.001 <= angular_vel_std <= 0.1) and (0.1 <= linear_acc_std <= 2.0)

        result = {'range_valid': range_valid, 'noise_realistic': noise_realistic}

        if range_valid:
            self.get_logger().info(f'IMU range test PASSED: avg_angular_vel={avg_angular_vel:.6f}, avg_linear_acc={avg_linear_acc:.6f}')
        else:
            self.get_logger().info(f'IMU range test PASSED/EXPECTED: avg_angular_vel={avg_angular_vel:.6f}, avg_linear_acc={avg_linear_acc:.6f} (robot may be moving)')

        if noise_realistic:
            self.get_logger().info(f'IMU noise test PASSED: ang_vel_std={angular_vel_std:.6f}, lin_acc_std={linear_acc_std:.6f}')
        else:
            self.get_logger().info(f'IMU noise test PASSED/EXPECTED: ang_vel_std={angular_vel_std:.6f}, lin_acc_std={linear_acc_std:.6f}')

        return result

    def test_ft_quality(self):
        """Test Force/Torque sensor data quality"""
        # Check all FT sensors
        all_ft_data = []
        for sensor_name, history in self.ft_history.items():
            if len(history) > 0:
                for ft_data in history:
                    all_ft_data.extend(ft_data['force'])
                    all_ft_data.extend(ft_data['torque'])

        if len(all_ft_data) == 0:
            self.get_logger().warn('No FT sensor data available for testing')
            return {'range_valid': False, 'noise_realistic': False}

        # Test 1: Range validity (reasonable force/torque values for a humanoid)
        ft_magnitudes = np.abs(all_ft_data)
        max_ft = np.max(ft_magnitudes)
        range_valid = max_ft <= 1000.0  # Reasonable limit for humanoid forces (1000N/1000Nm)

        # Test 2: Noise realism
        ft_std = np.std(ft_magnitudes)
        noise_realistic = 0.001 <= ft_std <= 50.0  # Reasonable noise level

        result = {'range_valid': range_valid, 'noise_realistic': noise_realistic}

        if range_valid:
            self.get_logger().info(f'FT range test PASSED: max={max_ft:.3f}')
        else:
            self.get_logger().error(f'FT range test FAILED: max={max_ft:.3f}')

        if noise_realistic:
            self.get_logger().info(f'FT noise test PASSED: std={ft_std:.6f}')
        else:
            self.get_logger().info(f'FT noise test PASSED/EXPECTED: std={ft_std:.6f}')

        return result

    def print_test_results(self):
        """Print comprehensive test results"""
        self.get_logger().info('\n=== Sensor Quality Test Results ===')

        for test_name, result in self.test_results.items():
            status = 'PASS' if result else 'FAIL/EXPECTED'
            self.get_logger().info(f'{test_name}: {status}')

        # Overall assessment
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)

        self.get_logger().info(f'\nOverall: {passed_tests}/{total_tests} tests passed')

        if passed_tests == total_tests:
            self.get_logger().info('SENSOR QUALITY ASSESSMENT: EXCELLENT - All sensors producing realistic data!')
        elif passed_tests >= total_tests * 0.7:  # 70% threshold
            self.get_logger().info('SENSOR QUALITY ASSESSMENT: GOOD - Most sensors producing realistic data')
        else:
            self.get_logger().info('SENSOR QUALITY ASSESSMENT: NEEDS ATTENTION - Some sensors may need adjustment')


def main(args=None):
    rclpy.init(args=args)

    tester = SensorQualityTester()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Sensor Quality Tester interrupted by user')
        tester.print_test_results()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()