#!/usr/bin/env python3

"""
VLA Integration Node for Vision-Language-Action (VLA) Module
Integrates multi-step tasks and coordinates between VLA components
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import time
from typing import Dict, List, Any, Optional
import json
from enum import Enum

# Import custom message types
from vla_integration.msg import ActionPlan, Task, ExecutionStatus
from vla_integration.srv import ExecutePlan, GenerateActionPlan


class IntegrationState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"


class VLAIntegrationNode(Node):
    """
    ROS 2 Node for integrating multi-step tasks and coordinating between VLA components
    """

    def __init__(self):
        super().__init__('vla_integration')

        # Declare parameters
        self.declare_parameter('integration_timeout', 300.0)  # 5 minutes
        self.declare_parameter('max_concurrent_integrations', 3)
        self.declare_parameter('integration_rate', 10.0)  # Hz
        self.declare_parameter('recovery_enabled', True)

        # Get parameters
        self.integration_timeout = self.get_parameter('integration_timeout').value
        self.max_concurrent_integrations = self.get_parameter('max_concurrent_integrations').value
        self.integration_rate = self.get_parameter('integration_rate').value
        self.recovery_enabled = self.get_parameter('recovery_enabled').value

        # Create service for multi-step task integration
        self.integration_service = self.create_service(
            ExecutePlan,
            'vla/integrate_tasks',
            self.integrate_tasks_callback
        )

        # Create client for action execution
        self.execution_client = self.create_client(ExecutePlan, 'vla/actions/execute_plan')
        while not self.execution_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for action execution service...')

        # Create client for plan generation
        self.plan_client = self.create_client(GenerateActionPlan, 'vla/generate_plan')
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for plan generation service...')

        # Create publisher for integration status
        self.integration_status_pub = self.create_publisher(ExecutionStatus, 'vla/integration_status', 10)

        # Initialize integration state
        self.active_integrations: Dict[str, Dict[str, Any]] = {}
        self.integration_queue: List[Dict[str, Any]] = []
        self.integration_history: List[Dict[str, Any]] = []

        # Create timer for integration monitoring
        self.integration_timer = self.create_timer(1.0/self.integration_rate, self.integration_monitor_callback)

        self.get_logger().info('VLA Integration Node initialized')

    def integrate_tasks_callback(self, request, response):
        """
        Service callback for integrating multi-step tasks
        Expected request format based on API contract:
        {
          "command": "Perform complex multi-step task",
          "execution_context": {
            "robot_id": "humanoid_001",
            "environment_id": "kitchen_01"
          }
        }

        Expected response format:
        {
          "integration_id": "int_12345",
          "status": "started",
          "estimated_duration": 120.0,
          "sub_tasks": ["task_1", "task_2", "task_3"],
          "timestamp": "2025-12-25T10:00:05Z"
        }
        """
        try:
            integration_id = f'int_{int(time.time())}_{hash(str(request.plan.plan_id if hasattr(request, "plan") else request.command)) % 10000}'
            self.get_logger().info(f'Integrating tasks for command: {request.command if hasattr(request, "command") else "unknown"}, integration ID: {integration_id}')

            # Create integration context
            integration_context = {
                'integration_id': integration_id,
                'command': getattr(request, 'command', 'unknown'),
                'plan': getattr(request, 'plan', None),
                'execution_context': getattr(request, 'execution_context', {}),
                'state': IntegrationState.IDLE,
                'start_time': time.time(),
                'current_step': 0,
                'total_steps': 0,
                'completed_steps': [],
                'failed_steps': [],
                'execution_id': None,
                'sub_tasks': [],
                'dependencies': {},
                'last_update': time.time()
            }

            # If no plan provided, generate one
            if integration_context['plan'] is None:
                plan_request = GenerateActionPlan.Request()
                plan_request.command = integration_context['command']
                plan_request.execution_context = json.dumps(integration_context['execution_context'])

                # Call plan generation service
                future = self.plan_client.call_async(plan_request)
                # In a real implementation, we'd wait for this asynchronously
                # For simulation, we'll create a basic plan
                integration_context['plan'] = self.create_basic_plan(integration_context['command'])
            else:
                integration_context['plan'] = request.plan

            # Analyze dependencies and sub-tasks
            integration_context['sub_tasks'] = [task.task_id for task in integration_context['plan'].tasks]
            integration_context['total_steps'] = len(integration_context['plan'].tasks)

            # Store integration context
            self.active_integrations[integration_id] = integration_context

            # Start integration process
            integration_context['state'] = IntegrationState.PROCESSING
            self.start_integration_process(integration_context)

            # Prepare response
            response.execution_id = integration_id
            response.status = 'started'
            response.message = f'Integration started with {len(integration_context["sub_tasks"])} sub-tasks'
            response.success = True
            response.estimated_duration = self.estimate_integration_duration(integration_context['plan'])
            response.timestamp = self.get_clock().now().to_msg()

            self.get_logger().info(f'Integration started: {integration_id}')

        except Exception as e:
            self.get_logger().error(f'Error in integration service: {str(e)}')
            response.execution_id = f'int_error_{int(time.time())}'
            response.status = 'failed'
            response.message = f'Integration error: {str(e)}'
            response.success = False
            response.estimated_duration = 0.0
            response.timestamp = self.get_clock().now().to_msg()

        return response

    def create_basic_plan(self, command: str) -> ActionPlan:
        """
        Create a basic action plan for simulation purposes
        """
        try:
            plan = ActionPlan()
            plan.plan_id = f'plan_{int(time.time())}'

            # Create basic tasks based on command
            if 'navigate' in command.lower() or 'go to' in command.lower():
                task = Task()
                task.task_id = 'nav_task_1'
                task.type = 'navigate'
                task.description = f'Navigate based on command: {command}'
                plan.tasks.append(task)
            elif 'find' in command.lower() or 'locate' in command.lower():
                task = Task()
                task.task_id = 'detect_task_1'
                task.type = 'detect_object'
                task.description = f'Detect object based on command: {command}'
                plan.tasks.append(task)
            else:
                # Default task
                task = Task()
                task.task_id = 'default_task_1'
                task.type = 'wait'
                task.description = f'Process command: {command}'
                plan.tasks.append(task)

            return plan

        except Exception as e:
            self.get_logger().error(f'Error creating basic plan: {str(e)}')
            # Return a minimal plan
            plan = ActionPlan()
            plan.plan_id = 'error_plan'
            task = Task()
            task.task_id = 'error_task'
            task.type = 'wait'
            task.description = 'Error recovery task'
            plan.tasks.append(task)
            return plan

    def start_integration_process(self, integration_context: Dict[str, Any]):
        """
        Start the integration process for a given context
        """
        try:
            integration_context['state'] = IntegrationState.EXECUTING

            # Execute the plan through the action execution service
            exec_request = ExecutePlan.Request()
            exec_request.plan = integration_context['plan']
            exec_request.execution_context = json.dumps(integration_context['execution_context'])

            # Call execution service
            future = self.execution_client.call_async(exec_request)
            # In a real implementation, we'd handle this asynchronously
            # For simulation, we'll just mark as executing

            self.get_logger().info(f'Started integration process for: {integration_context["integration_id"]}')

        except Exception as e:
            self.get_logger().error(f'Error starting integration process: {str(e)}')
            integration_context['state'] = IntegrationState.FAILED

    def integration_monitor_callback(self):
        """
        Monitor active integrations and update their status
        """
        try:
            current_time = time.time()

            for integration_id, context in list(self.active_integrations.items()):
                # Check for timeout
                if current_time - context['start_time'] > self.integration_timeout:
                    context['state'] = IntegrationState.FAILED
                    self.get_logger().warn(f'Integration {integration_id} timed out')

                # Update status
                self.publish_integration_status(context)

                # Check if integration is complete
                if context['state'] in [IntegrationState.COMPLETED, IntegrationState.FAILED]:
                    # Move to history
                    self.integration_history.append(context)
                    # Keep only recent history
                    if len(self.integration_history) > 50:  # Keep last 50 integrations
                        self.integration_history = self.integration_history[-50:]
                    # Remove from active
                    del self.active_integrations[integration_id]

        except Exception as e:
            self.get_logger().error(f'Error in integration monitor: {str(e)}')

    def publish_integration_status(self, context: Dict[str, Any]):
        """
        Publish integration status updates
        """
        try:
            status_msg = ExecutionStatus()
            status_msg.header.stamp = self.get_clock().now().to_msg()
            status_msg.header.frame_id = 'vla_integration'
            status_msg.execution_id = context['integration_id']
            status_msg.plan_id = context['plan'].plan_id if context['plan'] else 'unknown'
            status_msg.overall_status = context['state'].value
            status_msg.completed_tasks = len(context['completed_steps'])
            status_msg.total_tasks = context['total_steps']

            if status_msg.total_tasks > 0:
                status_msg.progress = float(status_msg.completed_tasks) / status_msg.total_tasks
            else:
                status_msg.progress = 0.0

            status_msg.error = ''

            self.integration_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing integration status: {str(e)}')

    def estimate_integration_duration(self, plan: ActionPlan) -> float:
        """
        Estimate the total duration for an integration plan
        """
        try:
            if not plan or not plan.tasks:
                return 10.0  # Default estimate

            total_time = 0.0
            task_times = {
                'navigate': 5.0,
                'detect_object': 3.0,
                'grasp_object': 4.0,
                'place_object': 4.0,
                'approach_object': 2.0,
                'manipulate_object': 5.0,
                'wait': 1.0,
                'move_arm': 3.0,
                'move_gripper': 2.0,
                'look_at': 1.0
            }

            for task in plan.tasks:
                duration = task_times.get(task.type, 3.0)
                total_time += duration

            return total_time

        except Exception as e:
            self.get_logger().error(f'Error estimating integration duration: {str(e)}')
            return 30.0  # Default estimate

    def get_integration_status(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific integration
        """
        try:
            return self.active_integrations.get(integration_id, None)
        except Exception as e:
            self.get_logger().error(f'Error getting integration status: {str(e)}')
            return None

    def get_active_integrations(self) -> List[str]:
        """
        Get list of active integration IDs
        """
        try:
            return list(self.active_integrations.keys())
        except Exception as e:
            self.get_logger().error(f'Error getting active integrations: {str(e)}')
            return []

    def get_integration_history(self) -> List[Dict[str, Any]]:
        """
        Get the integration history
        """
        try:
            return self.integration_history.copy()
        except Exception as e:
            self.get_logger().error(f'Error getting integration history: {str(e)}')
            return []

    def cancel_integration(self, integration_id: str) -> bool:
        """
        Cancel a specific integration
        """
        try:
            if integration_id in self.active_integrations:
                self.active_integrations[integration_id]['state'] = IntegrationState.FAILED
                self.active_integrations[integration_id]['message'] = 'Cancelled by user'
                self.get_logger().info(f'Integration {integration_id} cancelled')
                return True
            return False
        except Exception as e:
            self.get_logger().error(f'Error cancelling integration: {str(e)}')
            return False

    def get_integration_statistics(self) -> Dict[str, Any]:
        """
        Get integration statistics
        """
        try:
            active_count = len(self.active_integrations)
            history_count = len(self.integration_history)

            stats = {
                'active_integrations': active_count,
                'completed_integrations': history_count,
                'total_integrations': active_count + history_count,
                'average_duration': self.calculate_average_duration(),
                'success_rate': self.calculate_success_rate()
            }

            return stats

        except Exception as e:
            self.get_logger().error(f'Error getting integration statistics: {str(e)}')
            return {}

    def calculate_average_duration(self) -> float:
        """
        Calculate average duration of completed integrations
        """
        try:
            completed_integrations = [
                integration for integration in self.integration_history
                if integration['state'] == IntegrationState.COMPLETED
            ]

            if not completed_integrations:
                return 0.0

            total_duration = sum(
                integration.get('end_time', 0) - integration.get('start_time', 0)
                for integration in completed_integrations
            )

            return total_duration / len(completed_integrations)

        except Exception as e:
            self.get_logger().error(f'Error calculating average duration: {str(e)}')
            return 0.0

    def calculate_success_rate(self) -> float:
        """
        Calculate success rate of integrations
        """
        try:
            if not self.integration_history:
                return 0.0

            completed_count = sum(
                1 for integration in self.integration_history
                if integration['state'] == IntegrationState.COMPLETED
            )

            return completed_count / len(self.integration_history)

        except Exception as e:
            self.get_logger().error(f'Error calculating success rate: {str(e)}')
            return 0.0

    def reset_integration_node(self):
        """
        Reset the integration node to initial state
        """
        try:
            self.active_integrations.clear()
            self.integration_queue.clear()
            self.integration_history.clear()
            self.get_logger().info('VLA Integration node reset')
        except Exception as e:
            self.get_logger().error(f'Error resetting integration node: {str(e)}')

    def queue_integration(self, integration_request: Dict[str, Any]):
        """
        Queue an integration request for processing
        """
        try:
            self.integration_queue.append(integration_request)
            self.get_logger().info(f'Queued integration request, queue size: {len(self.integration_queue)}')
        except Exception as e:
            self.get_logger().error(f'Error queuing integration: {str(e)}')


def main(args=None):
    rclpy.init(args=args)

    vla_integration = VLAIntegrationNode()

    try:
        rclpy.spin(vla_integration)
    except KeyboardInterrupt:
        pass
    finally:
        vla_integration.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()