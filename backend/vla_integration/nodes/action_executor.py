#!/usr/bin/env python3

"""
Action Executor Node for Vision-Language-Action (VLA) Module
Handles execution of action plans through ROS 2 control systems
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import time
from typing import Dict, List, Any, Optional
import json

# Import custom message types
from vla_integration.msg import ActionPlan, Task, ExecutionStatus
from vla_integration.srv import ExecutePlan, SafetyValidation
from vla_integration.srv import GenerateActionPlan  # For potential plan refinement


class ActionExecutorNode(Node):
    """
    ROS 2 Node for executing action plans through ROS 2 control systems
    """

    def __init__(self):
        super().__init__('action_executor')

        # Declare parameters
        self.declare_parameter('max_execution_time', 300.0)  # 5 minutes
        self.declare_parameter('task_timeout', 60.0)
        self.declare_parameter('retry_attempts', 3)
        self.declare_parameter('safety_validation_enabled', True)
        self.declare_parameter('recovery_enabled', True)
        self.declare_parameter('execution_rate', 10.0)  # Hz

        # Get parameters
        self.max_execution_time = self.get_parameter('max_execution_time').value
        self.task_timeout = self.get_parameter('task_timeout').value
        self.retry_attempts = self.get_parameter('retry_attempts').value
        self.safety_validation_enabled = self.get_parameter('safety_validation_enabled').value
        self.recovery_enabled = self.get_parameter('recovery_enabled').value
        self.execution_rate = self.get_parameter('execution_rate').value

        # Create service for executing plans
        self.execute_service = self.create_service(
            ExecutePlan,
            'vla/execute_plan',
            self.execute_plan_callback
        )

        # Create publisher for execution status
        self.execution_status_pub = self.create_publisher(ExecutionStatus, 'vla/execution_status', 10)

        # Create client for safety validation
        self.safety_client = self.create_client(SafetyValidation, 'vla/safety/validate_action')
        while not self.safety_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for safety validation service...')

        # Store active executions
        self.active_executions: Dict[str, Dict[str, Any]] = {}

        # Create timer for status updates
        self.status_timer = self.create_timer(1.0, self.publish_execution_status)

        self.get_logger().info('Action Executor Node initialized')

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
              }
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
            execution_id = f'exec_{int(time.time())}_{hash(str(request.plan.plan_id)) % 10000}'
            self.get_logger().info(f'Executing plan {request.plan.plan_id} with execution ID: {execution_id}')

            # Validate the plan for safety if enabled
            if self.safety_validation_enabled:
                is_safe, violations = self.validate_plan_safety(request.plan)
                if not is_safe:
                    self.get_logger().error(f'Plan {request.plan.plan_id} failed safety validation: {violations}')
                    response.execution_id = execution_id
                    response.status = 'failed'
                    response.message = f'Safety validation failed: {violations}'
                    return response

            # Store execution context
            execution_context = {
                'plan': request.plan,
                'execution_id': execution_id,
                'status': 'executing',
                'start_time': time.time(),
                'current_task_idx': 0,
                'completed_tasks': [],
                'failed_tasks': [],
                'task_statuses': {task.task_id: 'pending' for task in request.plan.tasks}
            }

            self.active_executions[execution_id] = execution_context

            # Execute the plan asynchronously
            self.execute_plan_async(execution_context)

            # Prepare response
            response.execution_id = execution_id
            response.status = 'started'
            response.message = 'Plan execution started'
            response.estimated_duration = self.estimate_execution_time(request.plan)

            # Create task sequence for response
            task_sequence = []
            for task in request.plan.tasks:
                task_seq_item = ExecutePlan.Response.TaskSequence()
                task_seq_item.task_id = task.task_id
                task_seq_item.status = 'pending'
                task_seq_item.estimated_time = self.estimate_task_time(task)
                task_sequence.append(task_seq_item)

            response.task_sequence = task_sequence

            self.get_logger().info(f'Plan execution started: {execution_id}')

        except Exception as e:
            self.get_logger().error(f'Error in plan execution: {str(e)}')
            response.execution_id = f'exec_{int(time.time())}_error'
            response.status = 'failed'
            response.message = f'Error executing plan: {str(e)}'
            response.estimated_duration = 0.0

        return response

    def execute_plan_async(self, execution_context: Dict[str, Any]):
        """Execute a plan asynchronously"""
        try:
            plan = execution_context['plan']
            execution_id = execution_context['execution_id']

            # Execute tasks in sequence
            for idx, task in enumerate(plan.tasks):
                if execution_context['status'] != 'executing':
                    break

                # Update current task index
                execution_context['current_task_idx'] = idx

                # Execute the task
                task_success = self.execute_task(task, execution_context)

                if task_success:
                    execution_context['completed_tasks'].append(task.task_id)
                    execution_context['task_statuses'][task.task_id] = 'completed'
                    self.get_logger().info(f'Task {task.task_id} completed successfully')
                else:
                    execution_context['failed_tasks'].append(task.task_id)
                    execution_context['task_statuses'][task.task_id] = 'failed'

                    if self.recovery_enabled:
                        recovered = self.attempt_recovery(task, execution_context)
                        if recovered:
                            execution_context['task_statuses'][task.task_id] = 'recovered'
                            execution_context['completed_tasks'].append(task.task_id)
                        else:
                            execution_context['status'] = 'failed'
                            self.get_logger().error(f'Task {task.task_id} failed and recovery unsuccessful')
                            break
                    else:
                        execution_context['status'] = 'failed'
                        self.get_logger().error(f'Task {task.task_id} failed, stopping execution')
                        break

            # Update final execution status
            if execution_context['status'] == 'executing':
                execution_context['status'] = 'completed'
                self.get_logger().info(f'Plan execution completed successfully: {execution_id}')
            else:
                self.get_logger().warn(f'Plan execution failed: {execution_id}')

        except Exception as e:
            self.get_logger().error(f'Error in asynchronous plan execution: {str(e)}')
            execution_context['status'] = 'error'

    def execute_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute a single task"""
        try:
            self.get_logger().info(f'Executing task: {task.task_id} ({task.type})')

            # Update task status to executing
            execution_context['task_statuses'][task.task_id] = 'executing'

            # Handle different task types
            if task.type == 'navigate':
                success = self.execute_navigation_task(task, execution_context)
            elif task.type == 'detect_object':
                success = self.execute_detection_task(task, execution_context)
            elif task.type == 'grasp_object':
                success = self.execute_grasp_task(task, execution_context)
            elif task.type == 'place_object':
                success = self.execute_placement_task(task, execution_context)
            elif task.type == 'approach_object':
                success = self.execute_approach_task(task, execution_context)
            elif task.type == 'manipulate_object':
                success = self.execute_manipulation_task(task, execution_context)
            elif task.type == 'wait':
                success = self.execute_wait_task(task, execution_context)
            else:
                success = self.execute_generic_task(task, execution_context)

            return success

        except Exception as e:
            self.get_logger().error(f'Error executing task {task.task_id}: {str(e)}')
            return False

    def execute_navigation_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute navigation task"""
        try:
            # Extract navigation parameters
            target_position = self.extract_position_from_task(task)

            self.get_logger().info(f'Navigating to position: {target_position}')

            # In a real implementation, this would call navigation actions/services
            # For simulation, we'll just return success after a delay
            time.sleep(2)  # Simulate navigation time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in navigation task: {str(e)}')
            return False

    def execute_detection_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute object detection task"""
        try:
            # Extract detection parameters
            object_type = self.extract_parameter(task, 'object_type', 'any')
            color = self.extract_parameter(task, 'color', 'any')

            self.get_logger().info(f'Detecting object: {color} {object_type}')

            # In a real implementation, this would call vision services
            # For simulation, we'll just return success
            time.sleep(1)  # Simulate detection time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in detection task: {str(e)}')
            return False

    def execute_grasp_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute object grasping task"""
        try:
            # Extract grasp parameters
            object_id = self.extract_parameter(task, 'object_id', 'unknown')

            self.get_logger().info(f'Grasping object: {object_id}')

            # In a real implementation, this would call manipulation actions
            # For simulation, we'll just return success
            time.sleep(2)  # Simulate grasp time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in grasp task: {str(e)}')
            return False

    def execute_placement_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute object placement task"""
        try:
            # Extract placement parameters
            destination = self.extract_parameter(task, 'destination', 'default')

            self.get_logger().info(f'Placing object at: {destination}')

            # In a real implementation, this would call manipulation actions
            # For simulation, we'll just return success
            time.sleep(2)  # Simulate placement time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in placement task: {str(e)}')
            return False

    def execute_approach_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute approach object task"""
        try:
            # Extract approach parameters
            object_id = self.extract_parameter(task, 'object_id', 'unknown')

            self.get_logger().info(f'Approaching object: {object_id}')

            # In a real implementation, this would call navigation actions
            # For simulation, we'll just return success
            time.sleep(1)  # Simulate approach time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in approach task: {str(e)}')
            return False

    def execute_manipulation_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute object manipulation task"""
        try:
            # Extract manipulation parameters
            manipulation_type = self.extract_parameter(task, 'manipulation_type', 'generic')

            self.get_logger().info(f'Manipulating object: {manipulation_type}')

            # In a real implementation, this would call manipulation actions
            # For simulation, we'll just return success
            time.sleep(2)  # Simulate manipulation time

            return True

        except Exception as e:
            self.get_logger().error(f'Error in manipulation task: {str(e)}')
            return False

    def execute_wait_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute wait task"""
        try:
            # Extract wait duration
            duration = float(self.extract_parameter(task, 'duration', '1.0'))

            self.get_logger().info(f'Waiting for {duration} seconds')

            time.sleep(duration)  # Wait for specified duration

            return True

        except Exception as e:
            self.get_logger().error(f'Error in wait task: {str(e)}')
            return False

    def execute_generic_task(self, task: Task, execution_context: Dict[str, Any]) -> bool:
        """Execute generic task"""
        try:
            self.get_logger().info(f'Executing generic task: {task.description}')

            # For simulation, we'll just return success after a short delay
            time.sleep(1)

            return True

        except Exception as e:
            self.get_logger().error(f'Error in generic task: {str(e)}')
            return False

    def attempt_recovery(self, failed_task: Task, execution_context: Dict[str, Any]) -> bool:
        """Attempt to recover from a failed task"""
        try:
            self.get_logger().info(f'Attempting recovery for task: {failed_task.task_id}')

            # Simple recovery strategy: retry the task
            retry_count = 0
            max_retries = self.retry_attempts

            while retry_count < max_retries:
                self.get_logger().info(f'Retry attempt {retry_count + 1} for task {failed_task.task_id}')

                success = self.execute_task(failed_task, execution_context)
                if success:
                    self.get_logger().info(f'Task {failed_task.task_id} recovered on attempt {retry_count + 1}')
                    return True

                retry_count += 1
                time.sleep(1)  # Brief delay before retry

            self.get_logger().error(f'Failed to recover task {failed_task.task_id} after {max_retries} attempts')
            return False

        except Exception as e:
            self.get_logger().error(f'Error in recovery attempt: {str(e)}')
            return False

    def validate_plan_safety(self, plan: ActionPlan) -> (bool, List[str]):
        """Validate an action plan for safety constraints"""
        try:
            # Create safety validation request
            request = SafetyValidation.Request()
            request.action_plan = plan

            # Call safety validation service
            future = self.safety_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

            result = future.result()
            if result:
                return result.is_safe, [f"{v.type}: {v.description}" for v in result.violations]
            else:
                self.get_logger().warn('Safety validation service did not respond, assuming safe')
                return True, []

        except Exception as e:
            self.get_logger().error(f'Error in safety validation: {str(e)}')
            return True, []  # Assume safe if validation fails

    def extract_parameter(self, task: Task, param_name: str, default_value: str = '') -> str:
        """Extract a parameter from a task's parameters"""
        try:
            for param in task.parameters:
                if param.key == param_name:
                    return param.value
            return default_value
        except Exception:
            return default_value

    def extract_position_from_task(self, task: Task) -> Dict[str, float]:
        """Extract position parameters from a task"""
        try:
            position = {'x': 0.0, 'y': 0.0, 'z': 0.0}

            for param in task.parameters:
                if param.key in ['x', 'y', 'z']:
                    try:
                        position[param.key] = float(param.value)
                    except ValueError:
                        continue
                elif param.key == 'target_position':
                    # Handle position as a JSON string
                    try:
                        pos_dict = json.loads(param.value)
                        for key in ['x', 'y', 'z']:
                            if key in pos_dict:
                                position[key] = float(pos_dict[key])
                    except (ValueError, TypeError):
                        continue

            return position
        except Exception:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}

    def estimate_execution_time(self, plan: ActionPlan) -> float:
        """Estimate total execution time for a plan"""
        try:
            total_time = 0.0
            for task in plan.tasks:
                total_time += self.estimate_task_time(task)
            return total_time
        except Exception:
            return len(plan.tasks) * 5.0  # Default: 5 seconds per task

    def estimate_task_time(self, task: Task) -> float:
        """Estimate execution time for a single task"""
        try:
            # Default times for different task types
            task_times = {
                'navigate': 5.0,
                'detect_object': 2.0,
                'grasp_object': 3.0,
                'place_object': 3.0,
                'approach_object': 2.0,
                'manipulate_object': 4.0,
                'wait': 1.0,
                'move_arm': 2.0,
                'move_gripper': 1.5,
                'look_at': 1.0,
                'follow_path': 5.0
            }

            return task_times.get(task.type, 3.0)  # Default 3 seconds for unknown task types
        except Exception:
            return 3.0

    def publish_execution_status(self):
        """Publish execution status for active executions"""
        try:
            for exec_id, context in self.active_executions.items():
                status_msg = ExecutionStatus()
                status_msg.header.stamp = self.get_clock().now().to_msg()
                status_msg.header.frame_id = 'action_executor'
                status_msg.execution_id = context['execution_id']
                status_msg.plan_id = context['plan'].plan_id
                status_msg.overall_status = context['status']

                # Set current task ID
                if context['current_task_idx'] < len(context['plan'].tasks):
                    current_task = context['plan'].tasks[context['current_task_idx']]
                    status_msg.current_task_id = current_task.task_id
                else:
                    status_msg.current_task_id = ''

                status_msg.completed_tasks = len(context['completed_tasks'])
                status_msg.total_tasks = len(context['plan'].tasks)

                if status_msg.total_tasks > 0:
                    status_msg.progress = float(status_msg.completed_tasks) / status_msg.total_tasks
                else:
                    status_msg.progress = 0.0

                status_msg.error = ''  # Set to empty string or specific error if any

                # Publish the status
                self.execution_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing execution status: {str(e)}')


def main(args=None):
    rclpy.init(args=args)

    action_executor = ActionExecutorNode()

    try:
        rclpy.spin(action_executor)
    except KeyboardInterrupt:
        pass
    finally:
        action_executor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()