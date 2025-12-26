#!/usr/bin/env python3

"""
Action Execution Service Node for Vision-Language-Action (VLA) Module
API contract implementation for /actions/execute_plan
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
from typing import Dict, List, Any
import json

# Import custom message types
from vla_integration.msg import ActionPlan, Task, ExecutionStatus
from vla_integration.srv import ExecutePlan, SafetyValidation


class ActionExecutionServiceNode(Node):
    """
    ROS 2 Node implementing the action execution API service
    """

    def __init__(self):
        super().__init__('action_execution_service')

        # Declare parameters
        self.declare_parameter('service_timeout', 60.0)
        self.declare_parameter('max_concurrent_plans', 1)
        self.declare_parameter('task_timeout', 30.0)
        self.declare_parameter('action_validation_enabled', True)

        # Get parameters
        self.service_timeout = self.get_parameter('service_timeout').value
        self.max_concurrent_plans = self.get_parameter('max_concurrent_plans').value
        self.task_timeout = self.get_parameter('task_timeout').value
        self.action_validation_enabled = self.get_parameter('action_validation_enabled').value

        # Create service for executing plans
        self.execute_service = self.create_service(
            ExecutePlan,
            'vla/actions/execute_plan',
            self.execute_plan_callback
        )

        # Create publisher for execution status
        self.execution_status_pub = self.create_publisher(ExecutionStatus, 'vla/actions/execution_status', 10)

        # Create client for safety validation
        self.safety_client = self.create_client(SafetyValidation, 'vla/safety/validate_action')
        while not self.safety_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for safety validation service...')

        # Track active executions
        self.active_executions: Dict[str, Dict[str, Any]] = {}

        self.get_logger().info('Action Execution Service Node initialized')

    def execute_plan_callback(self, request, response):
        """
        Service callback for executing action plans
        Expected request format based on API contract:
        {
          "plan_id": "plan_67890",
          "tasks": [
            {
              "task_id": "task_001",
              "type": "detect_object",
              "parameters": {
                "object_type": "cup",
                "color": "red"
              },
              "dependencies": []
            }
          ],
          "execution_context": {
            "robot_id": "humanoid_001",
            "environment_id": "kitchen_01"
          }
        }

        Expected response format:
        {
          "execution_id": "exec_54321",
          "status": "started",
          "estimated_duration": 15.0,
          "task_sequence": [
            {
              "task_id": "task_001",
              "status": "pending",
              "estimated_time": 5.0
            }
          ]
        }
        """
        try:
            # Generate execution ID
            execution_id = f'exec_{int(time.time())}_{hash(str(request.plan.plan_id)) % 10000}'
            self.get_logger().info(f'Executing action plan: {request.plan.plan_id}, execution ID: {execution_id}')

            # Validate action plan for safety if enabled
            if self.action_validation_enabled:
                is_safe, violations = self.validate_action_safety(request.plan)
                if not is_safe:
                    self.get_logger().error(f'Action plan {request.plan.plan_id} failed safety validation: {violations}')
                    response.execution_id = execution_id
                    response.status = 'failed'
                    response.message = f'Safety validation failed: {violations}'
                    response.success = False
                    return response

            # Store execution context
            execution_context = {
                'plan': request.plan,
                'execution_id': execution_id,
                'status': 'initialized',
                'start_time': time.time(),
                'robot_id': self.extract_context_param(request.execution_context, 'robot_id', 'default_robot'),
                'environment_id': self.extract_context_param(request.execution_context, 'environment_id', 'default_env'),
                'current_task_idx': 0,
                'completed_tasks': [],
                'failed_tasks': [],
                'task_statuses': {task.task_id: 'pending' for task in request.plan.tasks}
            }

            self.active_executions[execution_id] = execution_context

            # Start plan execution asynchronously
            self.execute_plan_in_thread(execution_context)

            # Prepare response
            response.execution_id = execution_id
            response.status = 'started'
            response.message = 'Plan execution started successfully'
            response.success = True

            # Estimate total duration
            total_duration = self.estimate_total_duration(request.plan)
            response.estimated_duration = total_duration

            # Create task sequence for response
            task_sequence = []
            for task in request.plan.tasks:
                task_seq = ExecutePlan.Response.TaskSequence()
                task_seq.task_id = task.task_id
                task_seq.status = 'pending'
                task_seq.estimated_time = self.estimate_task_duration(task)
                task_sequence.append(task_seq)

            response.task_sequence = task_sequence

            self.get_logger().info(f'Action plan execution started: {execution_id}')

        except Exception as e:
            self.get_logger().error(f'Error in action execution service: {str(e)}')
            response.execution_id = f'exec_error_{int(time.time())}'
            response.status = 'failed'
            response.message = f'Error executing plan: {str(e)}'
            response.success = False
            response.estimated_duration = 0.0
            response.task_sequence = []

        return response

    def execute_plan_in_thread(self, execution_context: Dict[str, Any]):
        """Execute the plan in a separate thread"""
        try:
            plan = execution_context['plan']
            execution_id = execution_context['execution_id']

            # Update execution status
            execution_context['status'] = 'executing'

            # Execute tasks in sequence
            for idx, task in enumerate(plan.tasks):
                if execution_context['status'] not in ['executing', 'paused']:
                    break

                # Update current task index
                execution_context['current_task_idx'] = idx

                # Update task status to executing
                execution_context['task_statuses'][task.task_id] = 'executing'

                # Execute the task
                task_success = self.execute_task(task, execution_context)

                if task_success:
                    execution_context['completed_tasks'].append(task.task_id)
                    execution_context['task_statuses'][task.task_id] = 'completed'
                    self.get_logger().info(f'Task {task.task_id} completed successfully')
                else:
                    execution_context['failed_tasks'].append(task.task_id)
                    execution_context['task_statuses'][task.task_id] = 'failed'
                    execution_context['status'] = 'failed'
                    self.get_logger().error(f'Task {task.task_id} failed, stopping execution')
                    break

            # Update final status
            if execution_context['status'] == 'executing':
                execution_context['status'] = 'completed'
                self.get_logger().info(f'Plan execution completed successfully: {execution_id}')
            else:
                self.get_logger().info(f'Plan execution ended with status: {execution_context["status"]}')

        except Exception as e:
            self.get_logger().error(f'Error in plan execution thread: {str(e)}')
            execution_context['status'] = 'error'

    def execute_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute a single task in the context of the plan"""
        try:
            self.get_logger().info(f'Executing task {task.task_id} ({task.type}): {task.description}')

            # Handle different task types
            success = False
            if task.type == 'navigate':
                success = self.execute_navigate_task(task, execution_context)
            elif task.type == 'detect_object':
                success = self.execute_detect_object_task(task, execution_context)
            elif task.type == 'grasp_object':
                success = self.execute_grasp_object_task(task, execution_context)
            elif task.type == 'place_object':
                success = self.execute_place_object_task(task, execution_context)
            elif task.type == 'approach_object':
                success = self.execute_approach_object_task(task, execution_context)
            elif task.type == 'manipulate_object':
                success = self.execute_manipulate_object_task(task, execution_context)
            elif task.type == 'wait':
                success = self.execute_wait_task(task, execution_context)
            else:
                # Default handling for unknown task types
                success = self.execute_default_task(task, execution_context)

            return success

        except Exception as e:
            self.get_logger().error(f'Error executing task {task.task_id}: {str(e)}')
            return False

    def execute_navigate_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute navigation task"""
        try:
            target_location = self.extract_task_param(task, 'target_location', 'unknown')
            self.get_logger().info(f'Navigating to {target_location}')

            # Simulate navigation
            time.sleep(3)  # Simulate navigation time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in navigation task: {str(e)}')
            return False

    def execute_detect_object_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute object detection task"""
        try:
            object_type = self.extract_task_param(task, 'object_type', 'any')
            self.get_logger().info(f'Detecting {object_type}')

            # Simulate detection
            time.sleep(2)  # Simulate detection time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in detection task: {str(e)}')
            return False

    def execute_grasp_object_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute object grasping task"""
        try:
            object_id = self.extract_task_param(task, 'object_id', 'unknown')
            self.get_logger().info(f'Grasping object {object_id}')

            # Simulate grasping
            time.sleep(2)  # Simulate grasp time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in grasp task: {str(e)}')
            return False

    def execute_place_object_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute object placement task"""
        try:
            destination = self.extract_task_param(task, 'destination', 'default')
            self.get_logger().info(f'Placing object at {destination}')

            # Simulate placement
            time.sleep(2)  # Simulate placement time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in placement task: {str(e)}')
            return False

    def execute_approach_object_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute approach object task"""
        try:
            object_id = self.extract_task_param(task, 'object_id', 'unknown')
            self.get_logger().info(f'Approaching object {object_id}')

            # Simulate approach
            time.sleep(1)  # Simulate approach time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in approach task: {str(e)}')
            return False

    def execute_manipulate_object_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute object manipulation task"""
        try:
            manipulation_type = self.extract_task_param(task, 'manipulation_type', 'generic')
            self.get_logger().info(f'Manipulating object ({manipulation_type})')

            # Simulate manipulation
            time.sleep(3)  # Simulate manipulation time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in manipulation task: {str(e)}')
            return False

    def execute_wait_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute wait task"""
        try:
            duration = float(self.extract_task_param(task, 'duration', '1.0'))
            self.get_logger().info(f'Waiting for {duration} seconds')

            # Simulate wait
            time.sleep(duration)

            return True

        except Exception as e:
            self.get_logger().error(f'Error in wait task: {str(e)}')
            return False

    def execute_default_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute default task handler for unknown task types"""
        try:
            self.get_logger().info(f'Executing default task {task.task_id}: {task.type}')

            # Simulate task execution
            time.sleep(2)  # Default task time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in default task: {str(e)}')
            return False

    def validate_action_safety(self, plan: ActionPlan) -> (bool, List[str]):
        """Validate an action plan for safety using the safety validation service"""
        try:
            # Create safety validation request
            safety_request = SafetyValidation.Request()
            safety_request.action_plan = plan

            # Call safety validation service
            future = self.safety_client.call_async(safety_request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

            result = future.result()
            if result:
                violations_list = [f"{v.type}: {v.description}" for v in result.violations]
                return result.is_safe, violations_list
            else:
                self.get_logger().warn('Safety validation service did not respond, assuming safe')
                return True, []

        except Exception as e:
            self.get_logger().error(f'Error in action safety validation: {str(e)}')
            return True, []  # Assume safe if validation fails

    def extract_task_param(self, task: Task, param_name: str, default: str = '') -> str:
        """Extract a parameter value from a task"""
        try:
            for param in task.parameters:
                if param.key == param_name:
                    return param.value
            return default
        except Exception:
            return default

    def extract_context_param(self, context_str: str, param_name: str, default: str = '') -> str:
        """Extract a parameter from execution context string"""
        try:
            context_dict = json.loads(context_str)
            return context_dict.get(param_name, default)
        except (json.JSONDecodeError, TypeError):
            return default

    def estimate_task_duration(self, task: Task) -> float:
        """Estimate duration for a single task"""
        try:
            # Define default durations for different task types
            task_durations = {
                'navigate': 5.0,
                'detect_object': 2.0,
                'grasp_object': 3.0,
                'place_object': 3.0,
                'approach_object': 2.0,
                'manipulate_object': 4.0,
                'wait': float(self.extract_task_param(task, 'duration', '1.0')),
                'move_arm': 2.0,
                'move_gripper': 1.5,
                'look_at': 1.0
            }

            return task_durations.get(task.type, 2.0)  # Default 2 seconds
        except Exception:
            return 2.0

    def estimate_total_duration(self, plan: ActionPlan) -> float:
        """Estimate total duration for an action plan"""
        try:
            total = 0.0
            for task in plan.tasks:
                total += self.estimate_task_duration(task)
            return total
        except Exception:
            return len(plan.tasks) * 3.0  # Default: 3 seconds per task


def main(args=None):
    rclpy.init(args=args)

    action_service = ActionExecutionServiceNode()

    try:
        rclpy.spin(action_service)
    except KeyboardInterrupt:
        pass
    finally:
        action_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()