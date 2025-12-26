#!/usr/bin/env python3

"""
Error Recovery Node for Vision-Language-Action (VLA) Module
Handles error detection, recovery strategies, and fault tolerance
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
from typing import Dict, List, Any, Optional
import json
from enum import Enum

# Import custom message types
from vla_integration.msg import ActionPlan, Task, ExecutionStatus
from vla_integration.srv import ExecutePlan


class RecoveryStrategy(Enum):
    RETRY = "retry"
    SKIP = "skip"
    REPLAN = "replan"
    ABORT = "abort"
    FALLBACK = "fallback"


class ErrorRecoveryNode(Node):
    """
    ROS 2 Node for handling error detection, recovery strategies, and fault tolerance
    """

    def __init__(self):
        super().__init__('error_recovery')

        # Declare parameters
        self.declare_parameter('max_recovery_attempts', 3)
        self.declare_parameter('recovery_retry_delay', 2.0)  # seconds
        self.declare_parameter('error_detection_timeout', 10.0)  # seconds
        self.declare_parameter('recovery_enabled', True)

        # Get parameters
        self.max_recovery_attempts = self.get_parameter('max_recovery_attempts').value
        self.recovery_retry_delay = self.get_parameter('recovery_retry_delay').value
        self.error_detection_timeout = self.get_parameter('error_detection_timeout').value
        self.recovery_enabled = self.get_parameter('recovery_enabled').value

        # Create subscriber for error notifications
        self.error_sub = self.create_subscription(
            ExecutionStatus,
            'vla/error_status',
            self.error_callback,
            10
        )

        # Create publisher for recovery actions
        self.recovery_pub = self.create_publisher(ExecutionStatus, 'vla/recovery_status', 10)

        # Store error and recovery state
        self.error_history: Dict[str, List[Dict[str, Any]]] = {}
        self.recovery_attempts: Dict[str, int] = {}
        self.active_recovery_operations: Dict[str, Dict[str, Any]] = {}

        # Create timer for monitoring and cleanup
        self.monitor_timer = self.create_timer(1.0, self.monitor_errors)

        self.get_logger().info('Error Recovery Node initialized')

    def error_callback(self, msg: ExecutionStatus):
        """
        Callback for receiving error notifications
        """
        try:
            self.get_logger().info(f'Received error notification for execution: {msg.execution_id}, status: {msg.overall_status}')

            # Store error in history
            if msg.execution_id not in self.error_history:
                self.error_history[msg.execution_id] = []

            error_entry = {
                'timestamp': time.time(),
                'status': msg.overall_status,
                'error_message': msg.error if hasattr(msg, 'error') else 'Unknown error',
                'task_id': msg.current_task_id if hasattr(msg, 'current_task_id') else 'Unknown'
            }

            self.error_history[msg.execution_id].append(error_entry)

            # Trigger recovery if enabled
            if self.recovery_enabled:
                self.attempt_recovery(msg.execution_id, error_entry)

        except Exception as e:
            self.get_logger().error(f'Error in error callback: {str(e)}')

    def attempt_recovery(self, execution_id: str, error_info: Dict[str, Any]) -> bool:
        """
        Attempt to recover from an error using appropriate strategy
        """
        try:
            self.get_logger().info(f'Attempting recovery for execution: {execution_id}, error: {error_info["error_message"]}')

            # Check if we've exceeded recovery attempts
            if execution_id in self.recovery_attempts:
                if self.recovery_attempts[execution_id] >= self.max_recovery_attempts:
                    self.get_logger().error(f'Max recovery attempts exceeded for execution: {execution_id}')
                    return False
            else:
                self.recovery_attempts[execution_id] = 0

            # Determine recovery strategy based on error type
            strategy = self.select_recovery_strategy(error_info['error_message'], error_info['task_id'])

            self.get_logger().info(f'Selected recovery strategy: {strategy.value} for execution: {execution_id}')

            recovery_success = False
            if strategy == RecoveryStrategy.RETRY:
                recovery_success = self.retry_task(execution_id, error_info['task_id'])
            elif strategy == RecoveryStrategy.SKIP:
                recovery_success = self.skip_task(execution_id, error_info['task_id'])
            elif strategy == RecoveryStrategy.REPLAN:
                recovery_success = self.replan_task(execution_id, error_info['task_id'])
            elif strategy == RecoveryStrategy.FALLBACK:
                recovery_success = self.execute_fallback(execution_id, error_info['task_id'])
            elif strategy == RecoveryStrategy.ABORT:
                recovery_success = self.abort_execution(execution_id)

            # Update recovery attempt count
            self.recovery_attempts[execution_id] += 1

            # Publish recovery status
            recovery_status = ExecutionStatus()
            recovery_status.header.stamp = self.get_clock().now().to_msg()
            recovery_status.header.frame_id = 'error_recovery'
            recovery_status.execution_id = execution_id
            recovery_status.overall_status = 'recovery_attempted' if recovery_success else 'recovery_failed'
            recovery_status.error = f'Recovery {strategy.value} {"succeeded" if recovery_success else "failed"}'

            self.recovery_pub.publish(recovery_status)

            return recovery_success

        except Exception as e:
            self.get_logger().error(f'Error in recovery attempt: {str(e)}')
            return False

    def select_recovery_strategy(self, error_message: str, task_id: str) -> RecoveryStrategy:
        """
        Select appropriate recovery strategy based on error type
        """
        try:
            error_lower = error_message.lower()

            # Map error types to recovery strategies
            if 'timeout' in error_lower or 'connection' in error_lower:
                return RecoveryStrategy.RETRY
            elif 'collision' in error_lower or 'obstacle' in error_lower:
                return RecoveryStrategy.REPLAN
            elif 'hardware' in error_lower or 'critical' in error_lower:
                return RecoveryStrategy.ABORT
            elif 'not_found' in error_lower or 'unreachable' in error_lower:
                return RecoveryStrategy.SKIP
            elif 'fallback' in error_lower:
                return RecoveryStrategy.FALLBACK
            else:
                # Default to retry for unknown errors
                return RecoveryStrategy.RETRY

        except Exception as e:
            self.get_logger().error(f'Error selecting recovery strategy: {str(e)}')
            return RecoveryStrategy.RETRY  # Default to retry

    def retry_task(self, execution_id: str, task_id: str) -> bool:
        """
        Retry the failed task
        """
        try:
            self.get_logger().info(f'Retrying task {task_id} for execution {execution_id}')

            # In a real implementation, this would send a retry command to the executor
            # For simulation, we'll just wait and return success
            time.sleep(self.recovery_retry_delay)

            self.get_logger().info(f'Task retry completed for {task_id}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in task retry: {str(e)}')
            return False

    def skip_task(self, execution_id: str, task_id: str) -> bool:
        """
        Skip the failed task and continue execution
        """
        try:
            self.get_logger().info(f'Skipping task {task_id} for execution {execution_id}')

            # In a real implementation, this would update the execution context to skip the task
            # For simulation, we'll just log and return success
            self.get_logger().info(f'Task {task_id} marked as skipped')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in task skip: {str(e)}')
            return False

    def replan_task(self, execution_id: str, task_id: str) -> bool:
        """
        Generate a new plan to work around the failed task
        """
        try:
            self.get_logger().info(f'Replanning around task {task_id} for execution {execution_id}')

            # In a real implementation, this would call the planner to generate a new plan
            # For simulation, we'll just wait and return success
            time.sleep(self.recovery_retry_delay * 0.5)

            self.get_logger().info(f'Replanning completed for execution {execution_id}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in task replan: {str(e)}')
            return False

    def execute_fallback(self, execution_id: str, task_id: str) -> bool:
        """
        Execute a fallback behavior
        """
        try:
            self.get_logger().info(f'Executing fallback for task {task_id} in execution {execution_id}')

            # In a real implementation, this would execute a predefined fallback behavior
            # For simulation, we'll just wait and return success
            time.sleep(self.recovery_retry_delay * 0.3)

            self.get_logger().info(f'Fallback executed for execution {execution_id}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in fallback execution: {str(e)}')
            return False

    def abort_execution(self, execution_id: str) -> bool:
        """
        Abort the entire execution
        """
        try:
            self.get_logger().info(f'Aborting execution: {execution_id}')

            # In a real implementation, this would send an abort command to the executor
            # For simulation, we'll just log
            self.get_logger().info(f'Execution {execution_id} aborted')

            return True

        except Exception as e:
            self.get_logger().error(f'Error in execution abort: {str(e)}')
            return False

    def monitor_errors(self):
        """
        Monitor for errors and trigger recovery when needed
        """
        try:
            # Clean up old error history
            current_time = time.time()
            cutoff_time = current_time - self.error_detection_timeout * 10  # Keep for longer than timeout

            for exec_id, errors in list(self.error_history.items()):
                self.error_history[exec_id] = [
                    error for error in errors if error['timestamp'] > cutoff_time
                ]
                if not self.error_history[exec_id]:
                    del self.error_history[exec_id]

            # Clean up recovery attempts
            for exec_id, attempts in list(self.recovery_attempts.items()):
                if exec_id not in self.error_history and attempts == 0:
                    del self.recovery_attempts[exec_id]

        except Exception as e:
            self.get_logger().error(f'Error in error monitoring: {str(e)}')

    def get_error_history(self, execution_id: str) -> List[Dict[str, Any]]:
        """
        Get error history for a specific execution
        """
        try:
            return self.error_history.get(execution_id, [])
        except Exception as e:
            self.get_logger().error(f'Error getting error history: {str(e)}')
            return []

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Get overall recovery statistics
        """
        try:
            stats = {
                'total_error_count': sum(len(errors) for errors in self.error_history.values()),
                'active_recovery_operations': len(self.active_recovery_operations),
                'total_recovery_attempts': sum(self.recovery_attempts.values()),
                'recovery_success_rate': 'N/A'  # Would need success tracking to calculate
            }

            return stats

        except Exception as e:
            self.get_logger().error(f'Error getting recovery statistics: {str(e)}')
            return {}

    def reset_recovery_state(self, execution_id: str = None):
        """
        Reset recovery state for specific execution or all executions
        """
        try:
            if execution_id:
                if execution_id in self.error_history:
                    del self.error_history[execution_id]
                if execution_id in self.recovery_attempts:
                    del self.recovery_attempts[execution_id]
                if execution_id in self.active_recovery_operations:
                    del self.active_recovery_operations[execution_id]
                self.get_logger().info(f'Reset recovery state for execution: {execution_id}')
            else:
                self.error_history.clear()
                self.recovery_attempts.clear()
                self.active_recovery_operations.clear()
                self.get_logger().info('Reset all recovery state')

        except Exception as e:
            self.get_logger().error(f'Error resetting recovery state: {str(e)}')

    def is_recovery_active(self, execution_id: str) -> bool:
        """
        Check if recovery is currently active for an execution
        """
        return execution_id in self.active_recovery_operations


def main(args=None):
    rclpy.init(args=args)

    error_recovery = ErrorRecoveryNode()

    try:
        rclpy.spin(error_recovery)
    except KeyboardInterrupt:
        pass
    finally:
        error_recovery.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()