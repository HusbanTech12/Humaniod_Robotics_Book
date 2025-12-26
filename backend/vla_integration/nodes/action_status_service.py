#!/usr/bin/env python3

"""
Action Status Service Node for Vision-Language-Action (VLA) Module
API contract implementation for /actions/execution_status
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
from typing import Dict, List, Any
import json

# Import custom message types
from vla_integration.msg import ExecutionStatus
from vla_integration.srv import ExecutePlan


class ActionStatusServiceNode(Node):
    """
    ROS 2 Node implementing the action execution status API service
    """

    def __init__(self):
        super().__init__('action_status_service')

        # Declare parameters
        self.declare_parameter('status_update_rate', 10.0)  # Hz
        self.declare_parameter('max_status_history', 100)
        self.declare_parameter('status_retention_time', 300.0)  # 5 minutes

        # Get parameters
        self.status_update_rate = self.get_parameter('status_update_rate').value
        self.max_status_history = self.get_parameter('max_status_history').value
        self.status_retention_time = self.get_parameter('status_retention_time').value

        # Create service for getting execution status
        self.status_service = self.create_service(
            ExecutePlan,
            'vla/actions/execution_status',
            self.get_execution_status_callback
        )

        # Create publisher for status updates
        self.status_pub = self.create_publisher(ExecutionStatus, 'vla/actions/execution_status', 10)

        # Store execution status history
        self.status_history: Dict[str, List[Dict[str, Any]]] = {}
        self.current_status: Dict[str, ExecutionStatus] = {}

        # Create timer for periodic status updates
        self.status_timer = self.create_timer(1.0/self.status_update_rate, self.publish_status_updates)

        self.get_logger().info('Action Status Service Node initialized')

    def get_execution_status_callback(self, request, response):
        """
        Service callback for getting execution status
        Expected request format based on API contract:
        {
          "execution_id": "exec_12345"
        }

        Expected response format:
        {
          "execution_id": "exec_12345",
          "status": "executing",
          "progress": 0.6,
          "completed_tasks": 3,
          "total_tasks": 5,
          "current_task": "navigate_to_kitchen",
          "estimated_completion": 120.0,
          "timestamp": "2025-12-25T10:00:05Z"
        }
        """
        try:
            execution_id = request.execution_id if hasattr(request, 'execution_id') else None

            if execution_id is None:
                # If no execution ID specified, return status for all executions
                response.status = 'multiple'
                response.message = f'Returning status for {len(self.current_status)} active executions'
                response.success = True
            else:
                # Get status for specific execution
                if execution_id in self.current_status:
                    status_msg = self.current_status[execution_id]
                    response.execution_id = execution_id
                    response.status = status_msg.overall_status
                    response.completed_tasks = status_msg.completed_tasks
                    response.total_tasks = status_msg.total_tasks
                    response.progress = status_msg.progress
                    response.message = f'Status for execution {execution_id}'
                    response.success = True
                else:
                    response.execution_id = execution_id
                    response.status = 'not_found'
                    response.message = f'Execution {execution_id} not found'
                    response.success = False

            response.timestamp = self.get_clock().now().to_msg()

            self.get_logger().info(f'Returned execution status for: {execution_id}')

        except Exception as e:
            self.get_logger().error(f'Error in execution status service: {str(e)}')
            response.execution_id = f'error_{int(time.time())}'
            response.status = 'error'
            response.message = f'Error getting status: {str(e)}'
            response.success = False
            response.timestamp = self.get_clock().now().to_msg()

        return response

    def update_execution_status(self, status_msg: ExecutionStatus):
        """
        Update the stored status for an execution
        """
        try:
            execution_id = status_msg.execution_id

            # Store current status
            self.current_status[execution_id] = status_msg

            # Add to history
            if execution_id not in self.status_history:
                self.status_history[execution_id] = []

            # Add timestamp to the status update
            status_update = {
                'timestamp': time.time(),
                'status': status_msg.overall_status,
                'progress': status_msg.progress,
                'completed_tasks': status_msg.completed_tasks,
                'total_tasks': status_msg.total_tasks,
                'current_task': status_msg.current_task_id if hasattr(status_msg, 'current_task_id') else ''
            }

            self.status_history[execution_id].append(status_update)

            # Limit history size
            if len(self.status_history[execution_id]) > self.max_status_history:
                self.status_history[execution_id] = self.status_history[execution_id][-self.max_status_history:]

            # Clean up old histories
            self.cleanup_old_histories()

            self.get_logger().info(f'Updated status for execution: {execution_id}')

        except Exception as e:
            self.get_logger().error(f'Error updating execution status: {str(e)}')

    def publish_status_updates(self):
        """
        Publish status updates for all active executions
        """
        try:
            for exec_id, status_msg in self.current_status.items():
                # Update timestamp
                status_msg.header.stamp = self.get_clock().now().to_msg()
                self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing status updates: {str(e)}')

    def get_execution_history(self, execution_id: str) -> List[Dict[str, Any]]:
        """
        Get the status history for a specific execution
        """
        try:
            return self.status_history.get(execution_id, [])
        except Exception as e:
            self.get_logger().error(f'Error getting execution history: {str(e)}')
            return []

    def get_current_status(self, execution_id: str) -> ExecutionStatus:
        """
        Get the current status for a specific execution
        """
        try:
            return self.current_status.get(execution_id, ExecutionStatus())
        except Exception as e:
            self.get_logger().error(f'Error getting current status: {str(e)}')
            return ExecutionStatus()

    def get_all_active_executions(self) -> List[str]:
        """
        Get list of all active execution IDs
        """
        try:
            return list(self.current_status.keys())
        except Exception as e:
            self.get_logger().error(f'Error getting active executions: {str(e)}')
            return []

    def cleanup_old_histories(self):
        """
        Remove old status histories based on retention time
        """
        try:
            current_time = time.time()
            expired_executions = []

            for exec_id, history in self.status_history.items():
                if history:
                    last_update = history[-1]['timestamp']
                    if current_time - last_update > self.status_retention_time:
                        expired_executions.append(exec_id)

            for exec_id in expired_executions:
                del self.status_history[exec_id]
                # Also remove from current status if it's no longer active
                if exec_id in self.current_status:
                    del self.current_status[exec_id]

        except Exception as e:
            self.get_logger().error(f'Error cleaning up old histories: {str(e)}')

    def calculate_estimated_completion(self, execution_id: str) -> float:
        """
        Calculate estimated time to completion based on progress history
        """
        try:
            history = self.get_execution_history(execution_id)
            if len(history) < 2:
                return -1.0  # Unknown

            # Use the most recent progress data
            recent_updates = history[-10:]  # Last 10 updates
            if len(recent_updates) < 2:
                return -1.0

            start_update = recent_updates[0]
            end_update = recent_updates[-1]

            time_diff = end_update['timestamp'] - start_update['timestamp']
            progress_diff = end_update['progress'] - start_update['progress']

            if progress_diff <= 0:
                return -1.0  # Not making progress

            # Calculate rate of progress
            progress_rate = progress_diff / time_diff
            remaining_progress = 1.0 - end_update['progress']

            if progress_rate > 0:
                estimated_time = remaining_progress / progress_rate
                return estimated_time
            else:
                return -1.0

        except Exception as e:
            self.get_logger().error(f'Error calculating estimated completion: {str(e)}')
            return -1.0

    def get_execution_summary(self, execution_id: str) -> Dict[str, Any]:
        """
        Get a summary of execution statistics
        """
        try:
            current_status = self.get_current_status(execution_id)
            history = self.get_execution_history(execution_id)

            summary = {
                'execution_id': execution_id,
                'current_status': current_status.overall_status,
                'progress': current_status.progress,
                'completed_tasks': current_status.completed_tasks,
                'total_tasks': current_status.total_tasks,
                'estimated_completion': self.calculate_estimated_completion(execution_id),
                'history_length': len(history),
                'start_time': history[0]['timestamp'] if history else None,
                'current_time': time.time()
            }

            return summary

        except Exception as e:
            self.get_logger().error(f'Error getting execution summary: {str(e)}')
            return {}


def main(args=None):
    rclpy.init(args=args)

    status_service = ActionStatusServiceNode()

    try:
        rclpy.spin(status_service)
    except KeyboardInterrupt:
        pass
    finally:
        status_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()