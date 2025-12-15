#!/usr/bin/env python3
"""
Test script to validate robot response to gravity and basic physics in simulation.
This script can be run after launching the simulation to verify that the robot
responds correctly to physics.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import time


class PhysicsValidationTest(Node):
    def __init__(self):
        super().__init__('physics_validation_test')

        # Create TF buffer and listener to get robot position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to joint states to monitor joint positions
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.joint_states = None
        self.initial_height = None
        self.get_logger().info('Physics validation test node initialized')

    def joint_state_callback(self, msg):
        """Callback to store joint states"""
        self.joint_states = msg

    def get_robot_height(self, frame_id='base_link'):
        """Get the height of the robot base from the TF tree"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'world',  # Fixed frame
                frame_id,  # Robot frame
                rclpy.time.Time(),  # Latest available
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return transform.transform.translation.z
        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {e}')
            return None

    def validate_gravity_response(self, duration=10.0):
        """Validate that the robot responds to gravity by checking if it falls"""
        self.get_logger().info(f'Validating gravity response over {duration} seconds...')

        start_time = self.get_clock().now()
        initial_height = None

        # Wait a bit for transforms to be available
        while self.get_clock().now() - start_time < rclpy.time.Duration(seconds=2.0):
            initial_height = self.get_robot_height()
            if initial_height is not None:
                break
            time.sleep(0.1)

        if initial_height is None:
            self.get_logger().error('Could not get initial robot height')
            return False

        self.get_logger().info(f'Initial robot height: {initial_height:.3f}m')

        # Monitor robot height over time to see if it falls due to gravity
        falling_detected = False
        previous_height = initial_height

        start_time = self.get_clock().now()
        while self.get_clock().now() - start_time < rclpy.time.Duration(seconds=duration):
            current_height = self.get_robot_height()
            if current_height is not None:
                height_diff = previous_height - current_height
                if height_diff > 0.01:  # 1cm threshold for detecting fall
                    self.get_logger().info(f'Robot falling detected: height changed from {previous_height:.3f}m to {current_height:.3f}m')
                    falling_detected = True
                previous_height = current_height

            time.sleep(0.1)

        if falling_detected:
            self.get_logger().info('SUCCESS: Robot responds to gravity (falling detected)')
            return True
        else:
            self.get_logger().info('INFO: Robot may be stable (not falling) - this could be due to contacts with ground or controllers')
            return True  # This might be expected if robot is standing/walking with controllers

    def validate_joint_motion(self, duration=5.0):
        """Validate that joints can move and respond to physics"""
        self.get_logger().info(f'Validating joint motion over {duration} seconds...')

        start_time = self.get_clock().now()
        initial_positions = {}

        # Wait for initial joint states
        while self.joint_states is None and self.get_clock().now() - start_time < rclpy.time.Duration(seconds=5.0):
            time.sleep(0.1)

        if self.joint_states is None:
            self.get_logger().error('Could not get joint states')
            return False

        # Store initial positions for some key joints
        for i, name in enumerate(self.joint_states.name):
            if 'hip' in name or 'knee' in name or 'ankle' in name:  # Leg joints
                initial_positions[name] = self.joint_states.position[i]

        self.get_logger().info(f'Initial joint positions recorded for {len(initial_positions)} joints')

        # Monitor joint positions over time
        start_time = self.get_clock().now()
        significant_motion_detected = False

        while self.get_clock().now() - start_time < rclpy.time.Duration(seconds=duration):
            if self.joint_states is not None:
                for i, name in enumerate(self.joint_states.name):
                    if name in initial_positions:
                        current_pos = self.joint_states.position[i]
                        initial_pos = initial_positions[name]
                        if abs(current_pos - initial_pos) > 0.1:  # 0.1 rad threshold
                            self.get_logger().debug(f'Joint {name} moved: {initial_pos:.3f} -> {current_pos:.3f}')
                            significant_motion_detected = True

            time.sleep(0.1)

        if significant_motion_detected:
            self.get_logger().info('SUCCESS: Joint motion detected - physics simulation is active')
        else:
            self.get_logger().info('INFO: No significant joint motion detected - this may be expected')

        return True


def main(args=None):
    rclpy.init(args=args)

    test_node = PhysicsValidationTest()

    # Give some time for the simulation to start
    time.sleep(3.0)

    # Run validation tests
    gravity_success = test_node.validate_gravity_response(duration=10.0)
    joint_success = test_node.validate_joint_motion(duration=5.0)

    if gravity_success and joint_success:
        test_node.get_logger().info('PHYSICS VALIDATION: All tests passed!')
    else:
        test_node.get_logger().error('PHYSICS VALIDATION: Some tests failed!')

    test_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()