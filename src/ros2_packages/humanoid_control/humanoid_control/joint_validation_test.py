#!/usr/bin/env python3
"""
Test script to validate joint limits and range of motion in simulation.
This script checks that joints stay within their defined limits and can
achieve their expected range of motion.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
import math


class JointValidationTest(Node):
    def __init__(self):
        super().__init__('joint_validation_test')

        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publisher for joint commands to test range of motion
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.joint_states = None
        self.joint_limits = self.define_joint_limits()
        self.get_logger().info('Joint validation test node initialized')

    def define_joint_limits(self):
        """
        Define expected joint limits based on the URDF model.
        These should match the limits defined in the URDF file.
        """
        limits = {
            # Hip joints
            'hip_joint_r_x': {'lower': -1.57, 'upper': 1.57},  # -90 to 90 degrees
            'hip_joint_r_y': {'lower': -2.0, 'upper': 0.5},    # -114 to 28 degrees
            'hip_joint_r_z': {'lower': -0.78, 'upper': 0.78},  # -45 to 45 degrees
            'hip_joint_l_x': {'lower': -1.57, 'upper': 1.57},  # -90 to 90 degrees
            'hip_joint_l_y': {'lower': -2.0, 'upper': 0.5},    # -114 to 28 degrees
            'hip_joint_l_z': {'lower': -0.78, 'upper': 0.78},  # -45 to 45 degrees

            # Knee joints
            'knee_joint_r': {'lower': 0.0, 'upper': 2.35},     # 0 to 135 degrees
            'knee_joint_l': {'lower': 0.0, 'upper': 2.35},     # 0 to 135 degrees

            # Ankle joints
            'ankle_joint_r_x': {'lower': -0.78, 'upper': 0.78},  # -45 to 45 degrees
            'ankle_joint_r_y': {'lower': -0.52, 'upper': 0.52},  # -30 to 30 degrees
            'ankle_joint_l_x': {'lower': -0.78, 'upper': 0.78},  # -45 to 45 degrees
            'ankle_joint_l_y': {'lower': -0.52, 'upper': 0.52},  # -30 to 30 degrees

            # Shoulder joints
            'shoulder_joint_r_x': {'lower': -2.09, 'upper': 1.57},  # -120 to 90 degrees
            'shoulder_joint_r_y': {'lower': -2.09, 'upper': 1.57},  # -120 to 90 degrees
            'shoulder_joint_l_x': {'lower': -2.09, 'upper': 1.57},  # -120 to 90 degrees
            'shoulder_joint_l_y': {'lower': -2.09, 'upper': 1.57},  # -120 to 90 degrees

            # Elbow joints
            'elbow_joint_r': {'lower': 0.0, 'upper': 2.62},     # 0 to 150 degrees
            'elbow_joint_l': {'lower': 0.0, 'upper': 2.62},     # 0 to 150 degrees

            # Neck joint
            'neck_joint': {'lower': -0.78, 'upper': 0.78},      # -45 to 45 degrees
        }

        return limits

    def joint_state_callback(self, msg):
        """Callback to store joint states"""
        self.joint_states = msg

    def check_joint_limits(self):
        """Check if current joint positions are within defined limits"""
        if self.joint_states is None:
            self.get_logger().warn('No joint states received yet')
            return False

        all_within_limits = True
        violations = []

        for i, joint_name in enumerate(self.joint_states.name):
            if i < len(self.joint_states.position):
                position = self.joint_states.position[i]

                if joint_name in self.joint_limits:
                    limits = self.joint_limits[joint_name]
                    lower_limit = limits['lower']
                    upper_limit = limits['upper']

                    if position < lower_limit or position > upper_limit:
                        violations.append(f'{joint_name}: {position:.3f} (limits: {lower_limit:.3f} to {upper_limit:.3f})')
                        all_within_limits = False
                    else:
                        self.get_logger().debug(f'{joint_name}: {position:.3f} (OK)')
                else:
                    # Not all joints may have explicit limits defined
                    self.get_logger().debug(f'{joint_name}: {position:.3f} (no limits defined)')

        if violations:
            for violation in violations:
                self.get_logger().error(f'JOINT LIMIT VIOLATION: {violation}')
        else:
            self.get_logger().info('SUCCESS: All joints within defined limits')

        return all_within_limits

    def test_range_of_motion(self, duration=10.0):
        """Test range of motion by commanding joints to move to limit positions"""
        self.get_logger().info('Testing range of motion...')

        # Example trajectory for a few key joints
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['hip_joint_r_y', 'knee_joint_r', 'shoulder_joint_r_y']

        # Create trajectory points to test range of motion
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, 0.0, 0.0]  # Neutral position
        point1.time_from_start = Duration(sec=1, nanosec=0)

        point2 = JointTrajectoryPoint()
        point2.positions = [-1.0, 1.5, 1.0]  # Move toward limits
        point2.time_from_start = Duration(sec=3, nanosec=0)

        point3 = JointTrajectoryPoint()
        point3.positions = [0.0, 0.0, 0.0]  # Return to neutral
        point3.time_from_start = Duration(sec=5, nanosec=0)

        trajectory_msg.points = [point1, point2, point3]

        # Publish the trajectory
        self.joint_cmd_pub.publish(trajectory_msg)
        self.get_logger().info('Published trajectory command to test range of motion')

        # Wait and monitor joint positions
        start_time = self.get_clock().now()
        while self.get_clock().now() - start_time < rclpy.time.Duration(seconds=duration):
            self.check_joint_limits()
            time.sleep(0.5)

        return True

    def run_validation(self):
        """Run comprehensive joint validation tests"""
        self.get_logger().info('Starting joint validation tests...')

        # Wait for joint states to be available
        timeout = time.time() + 60*2  # 2 minutes timeout
        while self.joint_states is None and time.time() < timeout:
            self.get_logger().info('Waiting for joint states...')
            time.sleep(1.0)

        if self.joint_states is None:
            self.get_logger().error('Could not get joint states after timeout')
            return False

        # Check current joint positions against limits
        limits_ok = self.check_joint_limits()

        # Test range of motion
        motion_ok = self.test_range_of_motion(duration=15.0)

        if limits_ok and motion_ok:
            self.get_logger().info('JOINT VALIDATION: All tests passed!')
            return True
        else:
            self.get_logger().error('JOINT VALIDATION: Some tests failed!')
            return False


def main(args=None):
    rclpy.init(args=args)

    test_node = JointValidationTest()

    # Run validation tests
    success = test_node.run_validation()

    if success:
        test_node.get_logger().info('JOINT VALIDATION COMPLETED SUCCESSFULLY')
    else:
        test_node.get_logger().error('JOINT VALIDATION FAILED')

    test_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()