#!/usr/bin/env python3

"""
Robot State Monitor Node for Vision-Language-Action (VLA) Module
Monitors robot state and provides state information for planning and execution
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
import time
from typing import Dict, List, Any, Optional
import json
import threading

# Import custom message types
from vla_integration.msg import ExecutionStatus


class RobotStateMonitorNode(Node):
    """
    ROS 2 Node for monitoring robot state and providing state information
    """

    def __init__(self):
        super().__init__('robot_state_monitor')

        # Declare parameters
        self.declare_parameter('state_update_rate', 10.0)  # Hz
        self.declare_parameter('state_history_size', 100)
        self.declare_parameter('critical_state_timeout', 5.0)  # seconds
        self.declare_parameter('monitoring_enabled', True)

        # Get parameters
        self.state_update_rate = self.get_parameter('state_update_rate').value
        self.state_history_size = self.get_parameter('state_history_size').value
        self.critical_state_timeout = self.get_parameter('critical_state_timeout').value
        self.monitoring_enabled = self.get_parameter('monitoring_enabled').value

        # Create subscribers for robot state topics
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        # Create publisher for robot state updates
        self.state_status_pub = self.create_publisher(ExecutionStatus, 'vla/robot_state', 10)

        # Initialize robot state storage
        self.current_state = {
            'joint_states': {},
            'pose': Pose(),
            'twist': Twist(),
            'timestamp': time.time(),
            'robot_id': 'default_robot',
            'is_connected': False,
            'is_operational': False,
            'is_emergency_stopped': False,
            'battery_level': 100.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0
        }

        self.state_history: List[Dict[str, Any]] = []
        self.state_lock = threading.Lock()

        # Create timer for state monitoring
        self.state_timer = self.create_timer(1.0/self.state_update_rate, self.state_monitor_callback)

        self.get_logger().info('Robot State Monitor Node initialized')

    def joint_state_callback(self, msg: JointState):
        """
        Callback for joint state messages
        """
        try:
            with self.state_lock:
                # Update joint states
                for i, name in enumerate(msg.name):
                    if i < len(msg.position):
                        position = msg.position[i] if i < len(msg.position) else 0.0
                        velocity = msg.velocity[i] if i < len(msg.velocity) else 0.0
                        effort = msg.effort[i] if i < len(msg.effort) else 0.0

                        self.current_state['joint_states'][name] = {
                            'position': position,
                            'velocity': velocity,
                            'effort': effort,
                            'timestamp': time.time()
                        }

                self.current_state['timestamp'] = time.time()
                self.current_state['is_connected'] = True
                self.current_state['is_operational'] = True

                # Update state history
                self.update_state_history()

                self.get_logger().debug(f'Updated joint states for {len(msg.name)} joints')

        except Exception as e:
            self.get_logger().error(f'Error in joint state callback: {str(e)}')

    def odom_callback(self, msg: Odometry):
        """
        Callback for odometry messages
        """
        try:
            with self.state_lock:
                # Update pose and twist
                self.current_state['pose'] = msg.pose.pose
                self.current_state['twist'] = msg.twist.twist
                self.current_state['timestamp'] = time.time()

                # Update state history
                self.update_state_history()

                self.get_logger().debug(f'Updated robot pose and twist')

        except Exception as e:
            self.get_logger().error(f'Error in odometry callback: {str(e)}')

    def state_monitor_callback(self):
        """
        Main state monitoring callback
        """
        try:
            if not self.monitoring_enabled:
                return

            with self.state_lock:
                # Check for stale state
                time_since_update = time.time() - self.current_state['timestamp']
                if time_since_update > self.critical_state_timeout:
                    self.current_state['is_connected'] = False
                    self.current_state['is_operational'] = False

                # Publish state status
                self.publish_robot_state()

        except Exception as e:
            self.get_logger().error(f'Error in state monitor callback: {str(e)}')

    def update_state_history(self):
        """
        Update the state history with current state
        """
        try:
            state_snapshot = self.get_current_state()
            self.state_history.append(state_snapshot)

            # Limit history size
            if len(self.state_history) > self.state_history_size:
                self.state_history = self.state_history[-self.state_history_size:]

        except Exception as e:
            self.get_logger().error(f'Error updating state history: {str(e)}')

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get a copy of the current robot state
        """
        try:
            with self.state_lock:
                return self.current_state.copy()
        except Exception as e:
            self.get_logger().error(f'Error getting current state: {str(e)}')
            return {}

    def get_joint_state(self, joint_name: str) -> Optional[Dict[str, float]]:
        """
        Get state for a specific joint
        """
        try:
            with self.state_lock:
                return self.current_state['joint_states'].get(joint_name, None)
        except Exception as e:
            self.get_logger().error(f'Error getting joint state: {str(e)}')
            return None

    def get_robot_pose(self) -> Pose:
        """
        Get the current robot pose
        """
        try:
            with self.state_lock:
                return self.current_state['pose']
        except Exception as e:
            self.get_logger().error(f'Error getting robot pose: {str(e)}')
            return Pose()

    def get_robot_twist(self) -> Twist:
        """
        Get the current robot twist (velocity)
        """
        try:
            with self.state_lock:
                return self.current_state['twist']
        except Exception as e:
            self.get_logger().error(f'Error getting robot twist: {str(e)}')
            return Twist()

    def is_robot_operational(self) -> bool:
        """
        Check if the robot is operational
        """
        try:
            with self.state_lock:
                return self.current_state['is_operational']
        except Exception as e:
            self.get_logger().error(f'Error checking robot operational status: {str(e)}')
            return False

    def is_robot_connected(self) -> bool:
        """
        Check if the robot is connected
        """
        try:
            with self.state_lock:
                return self.current_state['is_connected']
        except Exception as e:
            self.get_logger().error(f'Error checking robot connection status: {str(e)}')
            return False

    def get_battery_level(self) -> float:
        """
        Get the current battery level
        """
        try:
            with self.state_lock:
                return self.current_state['battery_level']
        except Exception as e:
            self.get_logger().error(f'Error getting battery level: {str(e)}')
            return 100.0

    def is_emergency_stopped(self) -> bool:
        """
        Check if the robot is emergency stopped
        """
        try:
            with self.state_lock:
                return self.current_state['is_emergency_stopped']
        except Exception as e:
            self.get_logger().error(f'Error checking emergency stop status: {str(e)}')
            return False

    def get_reachable_joints(self) -> List[str]:
        """
        Get list of joints that are currently reachable (not in error state)
        """
        try:
            with self.state_lock:
                reachable = []
                current_time = time.time()

                for joint_name, joint_data in self.current_state['joint_states'].items():
                    # Check if joint state is recent (not stale)
                    if current_time - joint_data['timestamp'] < self.critical_state_timeout:
                        reachable.append(joint_name)

                return reachable
        except Exception as e:
            self.get_logger().error(f'Error getting reachable joints: {str(e)}')
            return []

    def get_state_at_time(self, target_time: float) -> Optional[Dict[str, Any]]:
        """
        Get robot state at a specific time from history
        """
        try:
            for state in reversed(self.state_history):
                if state['timestamp'] <= target_time:
                    return state
            return None
        except Exception as e:
            self.get_logger().error(f'Error getting state at time: {str(e)}')
            return None

    def get_state_history(self, start_time: float = None, end_time: float = None) -> List[Dict[str, Any]]:
        """
        Get state history within a time range
        """
        try:
            if not start_time and not end_time:
                return self.state_history.copy()

            filtered_history = []
            for state in self.state_history:
                timestamp = state['timestamp']
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_history.append(state)

            return filtered_history
        except Exception as e:
            self.get_logger().error(f'Error getting state history: {str(e)}')
            return []

    def publish_robot_state(self):
        """
        Publish current robot state as ExecutionStatus message
        """
        try:
            status_msg = ExecutionStatus()
            status_msg.header.stamp = self.get_clock().now().to_msg()
            status_msg.header.frame_id = 'robot_state_monitor'
            status_msg.execution_id = f'state_{int(time.time())}'
            status_msg.plan_id = 'robot_state_monitor'

            with self.state_lock:
                status_msg.overall_status = 'operational' if self.current_state['is_operational'] else 'not_operational'
                status_msg.completed_tasks = len(self.current_state['joint_states'])
                status_msg.total_tasks = 1  # For state monitoring
                status_msg.progress = 1.0 if self.current_state['is_operational'] else 0.0
                status_msg.error = 'Emergency stop' if self.current_state['is_emergency_stopped'] else ''

            self.state_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing robot state: {str(e)}')

    def get_robot_reachability(self, target_position: Dict[str, float]) -> bool:
        """
        Check if a target position is reachable by the robot
        """
        try:
            # In a real implementation, this would perform inverse kinematics
            # For simulation, we'll return True as a placeholder
            return True
        except Exception as e:
            self.get_logger().error(f'Error checking reachability: {str(e)}')
            return False

    def get_robot_workspace_bounds(self) -> Dict[str, float]:
        """
        Get the robot's workspace bounds
        """
        try:
            # Return default workspace bounds
            return {
                'x_min': -2.0, 'x_max': 2.0,
                'y_min': -2.0, 'y_max': 2.0,
                'z_min': 0.0, 'z_max': 1.5
            }
        except Exception as e:
            self.get_logger().error(f'Error getting workspace bounds: {str(e)}')
            return {}

    def reset_state_monitor(self):
        """
        Reset the state monitor to initial state
        """
        try:
            with self.state_lock:
                self.current_state = {
                    'joint_states': {},
                    'pose': Pose(),
                    'twist': Twist(),
                    'timestamp': time.time(),
                    'robot_id': 'default_robot',
                    'is_connected': False,
                    'is_operational': False,
                    'is_emergency_stopped': False,
                    'battery_level': 100.0,
                    'cpu_usage': 0.0,
                    'memory_usage': 0.0
                }
                self.state_history.clear()

            self.get_logger().info('Robot state monitor reset')

        except Exception as e:
            self.get_logger().error(f'Error resetting state monitor: {str(e)}')


def main(args=None):
    rclpy.init(args=args)

    state_monitor = RobotStateMonitorNode()

    try:
        rclpy.spin(state_monitor)
    except KeyboardInterrupt:
        pass
    finally:
        state_monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()