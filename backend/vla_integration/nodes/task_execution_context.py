#!/usr/bin/env python3

"""
Task Execution Context Manager for Vision-Language-Action (VLA) Module
Manages execution state and context during plan execution
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
from typing import Dict, List, Any, Optional
import json

# Import custom message types
from vla_integration.msg import ActionPlan, Task, ExecutionStatus
from vla_integration.srv import ExecutePlan


class TaskExecutionContextNode(Node):
    """
    ROS 2 Node for managing task execution context and state
    """

    def __init__(self):
        super().__init__('task_execution_context')

        # Declare parameters
        self.declare_parameter('context_retention_time', 300.0)  # 5 minutes
        self.declare_parameter('max_active_contexts', 10)
        self.declare_parameter('context_update_rate', 1.0)  # Hz

        # Get parameters
        self.context_retention_time = self.get_parameter('context_retention_time').value
        self.max_active_contexts = self.get_parameter('max_active_contexts').value
        self.context_update_rate = self.get_parameter('context_update_rate').value

        # Create publisher for context updates
        self.context_status_pub = self.create_publisher(ExecutionStatus, 'vla/context_status', 10)

        # Initialize context storage
        self.execution_contexts: Dict[str, Dict[str, Any]] = {}
        self.context_history: List[Dict[str, Any]] = []

        # Create timer for context updates
        self.context_timer = self.create_timer(1.0/self.context_update_rate, self.publish_context_status)

        self.get_logger().info('Task Execution Context Node initialized')

    def create_execution_context(self, plan: ActionPlan, execution_id: str) -> Dict[str, Any]:
        """Create a new execution context for a plan"""
        try:
            context = {
                'plan': plan,
                'execution_id': execution_id,
                'status': 'initialized',
                'start_time': time.time(),
                'current_task_idx': 0,
                'completed_tasks': [],
                'failed_tasks': [],
                'pending_tasks': [task.task_id for task in plan.tasks],
                'task_statuses': {task.task_id: 'pending' for task in plan.tasks},
                'task_results': {},
                'execution_history': [],
                'robot_state': {},
                'environment_state': {},
                'last_update': time.time()
            }

            self.execution_contexts[execution_id] = context
            self.get_logger().info(f'Created execution context for: {execution_id}')

            return context

        except Exception as e:
            self.get_logger().error(f'Error creating execution context: {str(e)}')
            return {}

    def update_task_status(self, execution_id: str, task_id: str, status: str, result: Any = None):
        """Update the status of a specific task in the execution context"""
        try:
            if execution_id not in self.execution_contexts:
                self.get_logger().error(f'Execution context not found: {execution_id}')
                return False

            context = self.execution_contexts[execution_id]

            # Update task status
            context['task_statuses'][task_id] = status

            # Update task lists based on status
            if status == 'completed':
                if task_id in context['pending_tasks']:
                    context['pending_tasks'].remove(task_id)
                if task_id not in context['completed_tasks']:
                    context['completed_tasks'].append(task_id)
            elif status == 'failed':
                if task_id in context['pending_tasks']:
                    context['pending_tasks'].remove(task_id)
                if task_id not in context['failed_tasks']:
                    context['failed_tasks'].append(task_id)

            # Store task result if provided
            if result is not None:
                context['task_results'][task_id] = result

            # Update last update time
            context['last_update'] = time.time()

            self.get_logger().info(f'Updated task {task_id} status to {status} for execution {execution_id}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error updating task status: {str(e)}')
            return False

    def update_execution_status(self, execution_id: str, status: str):
        """Update the overall execution status"""
        try:
            if execution_id not in self.execution_contexts:
                self.get_logger().error(f'Execution context not found: {execution_id}')
                return False

            context = self.execution_contexts[execution_id]
            context['status'] = status
            context['last_update'] = time.time()

            self.get_logger().info(f'Updated execution {execution_id} status to {status}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error updating execution status: {str(e)}')
            return False

    def get_execution_context(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the execution context for a specific execution ID"""
        try:
            return self.execution_contexts.get(execution_id, None)
        except Exception as e:
            self.get_logger().error(f'Error getting execution context: {str(e)}')
            return None

    def cleanup_expired_contexts(self):
        """Remove expired execution contexts"""
        try:
            current_time = time.time()
            expired_ids = []

            for exec_id, context in self.execution_contexts.items():
                if current_time - context['last_update'] > self.context_retention_time:
                    expired_ids.append(exec_id)

            for exec_id in expired_ids:
                # Move to history before removing
                if exec_id in self.execution_contexts:
                    self.context_history.append(self.execution_contexts[exec_id])
                    # Keep only recent history
                    if len(self.context_history) > self.max_active_contexts:
                        self.context_history = self.context_history[-self.max_active_contexts:]

                del self.execution_contexts[exec_id]
                self.get_logger().info(f'Removed expired execution context: {exec_id}')

        except Exception as e:
            self.get_logger().error(f'Error cleaning up expired contexts: {str(e)}')

    def publish_context_status(self):
        """Publish context status updates"""
        try:
            # Clean up expired contexts
            self.cleanup_expired_contexts()

            # Publish status for active contexts
            for exec_id, context in self.execution_contexts.items():
                status_msg = ExecutionStatus()
                status_msg.header.stamp = self.get_clock().now().to_msg()
                status_msg.header.frame_id = 'task_execution_context'
                status_msg.execution_id = exec_id
                status_msg.plan_id = context['plan'].plan_id
                status_msg.overall_status = context['status']
                status_msg.completed_tasks = len(context['completed_tasks'])
                status_msg.total_tasks = len(context['plan'].tasks)

                if status_msg.total_tasks > 0:
                    status_msg.progress = float(status_msg.completed_tasks) / status_msg.total_tasks
                else:
                    status_msg.progress = 0.0

                status_msg.error = ''

                self.context_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing context status: {str(e)}')

    def get_active_executions_count(self) -> int:
        """Get the count of active executions"""
        return len(self.execution_contexts)

    def get_execution_progress(self, execution_id: str) -> float:
        """Get progress percentage for a specific execution"""
        try:
            context = self.get_execution_context(execution_id)
            if context and context['plan'].tasks:
                return len(context['completed_tasks']) / len(context['plan'].tasks)
            return 0.0
        except Exception:
            return 0.0


def main(args=None):
    rclpy.init(args=args)

    context_manager = TaskExecutionContextNode()

    try:
        rclpy.spin(context_manager)
    except KeyboardInterrupt:
        pass
    finally:
        context_manager.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()