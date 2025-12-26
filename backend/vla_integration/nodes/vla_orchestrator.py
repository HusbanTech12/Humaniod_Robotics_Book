#!/usr/bin/env python3

"""
VLA Orchestrator Node for Vision-Language-Action (VLA) Module
Main orchestrator for the complete VLA system
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
from vla_integration.msg import ActionPlan, Task, ExecutionStatus, VisionData
from vla_integration.srv import ExecutePlan, GenerateActionPlan, LocalizeObject


class OrchestratorState(Enum):
    IDLE = "idle"
    PROCESSING_VOICE = "processing_voice"
    GENERATING_PLAN = "generating_plan"
    PROCESSING_VISION = "processing_vision"
    EXECUTING_PLAN = "executing_plan"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"


class VLAOrchestratorNode(Node):
    """
    ROS 2 Node for orchestrating the complete VLA system
    """

    def __init__(self):
        super().__init__('vla_orchestrator')

        # Declare parameters
        self.declare_parameter('orchestration_timeout', 600.0)  # 10 minutes
        self.declare_parameter('max_concurrent_orchestrations', 1)
        self.declare_parameter('orchestration_rate', 10.0)  # Hz
        self.declare_parameter('recovery_enabled', True)
        self.declare_parameter('safety_validation_enabled', True)

        # Get parameters
        self.orchestration_timeout = self.get_parameter('orchestration_timeout').value
        self.max_concurrent_orchestrations = self.get_parameter('max_concurrent_orchestrations').value
        self.orchestration_rate = self.get_parameter('orchestration_rate').value
        self.recovery_enabled = self.get_parameter('recovery_enabled').value
        self.safety_validation_enabled = self.get_parameter('safety_validation_enabled').value

        # Create service for orchestration
        self.orchestration_service = self.create_service(
            ExecutePlan,
            'vla/orchestrate',
            self.orchestrate_callback
        )

        # Create clients for various services
        self.voice_service_client = self.create_client(GenerateActionPlan, 'vla/voice/process_command')
        self.plan_service_client = self.create_client(GenerateActionPlan, 'vla/generate_plan')
        self.vision_service_client = self.create_client(LocalizeObject, 'vla/vision/localize_object')
        self.execution_service_client = self.create_client(ExecutePlan, 'vla/actions/execute_plan')

        # Wait for services to be available
        for client, service_name in [
            (self.voice_service_client, 'voice/process_command'),
            (self.plan_service_client, 'generate_plan'),
            (self.vision_service_client, 'vision/localize_object'),
            (self.execution_service_client, 'actions/execute_plan')
        ]:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for {service_name} service...')

        # Create publisher for orchestration status
        self.orchestration_status_pub = self.create_publisher(ExecutionStatus, 'vla/orchestration_status', 10)

        # Initialize orchestrator state
        self.active_orchestrations: Dict[str, Dict[str, Any]] = {}
        self.orchestration_queue: List[Dict[str, Any]] = []
        self.orchestration_history: List[Dict[str, Any]] = []

        # Create timer for orchestration monitoring
        self.orchestration_timer = self.create_timer(1.0/self.orchestration_rate, self.orchestration_monitor_callback)

        self.get_logger().info('VLA Orchestrator Node initialized')

    def orchestrate_callback(self, request, response):
        """
        Service callback for orchestrating the complete VLA pipeline
        Expected request format based on API contract:
        {
          "command": "Natural language command",
          "execution_context": {
            "robot_id": "humanoid_001",
            "environment_id": "kitchen_01"
          }
        }

        Expected response format:
        {
          "orchestration_id": "orch_12345",
          "status": "started",
          "estimated_duration": 120.0,
          "steps": ["voice", "planning", "vision", "execution"],
          "timestamp": "2025-12-25T10:00:05Z"
        }
        """
        try:
            orchestration_id = f'orch_{int(time.time())}_{hash(str(request.command if hasattr(request, "command") else "unknown")) % 10000}'
            self.get_logger().info(f'Orchestrating VLA pipeline for command: {request.command if hasattr(request, "command") else "unknown"}, orchestration ID: {orchestration_id}')

            # Create orchestration context
            orchestration_context = {
                'orchestration_id': orchestration_id,
                'command': getattr(request, 'command', 'unknown'),
                'execution_context': getattr(request, 'execution_context', {}),
                'state': OrchestratorState.IDLE,
                'start_time': time.time(),
                'current_step': 0,
                'total_steps': 4,  # voice, planning, vision, execution
                'completed_steps': [],
                'failed_steps': [],
                'plan': None,
                'vision_data': None,
                'execution_result': None,
                'last_update': time.time(),
                'step_results': {}
            }

            # Store orchestration context
            self.active_orchestrations[orchestration_id] = orchestration_context

            # Start orchestration process
            orchestration_context['state'] = OrchestratorState.PROCESSING_VOICE
            self.start_orchestration_process(orchestration_context)

            # Prepare response
            response.execution_id = orchestration_id
            response.status = 'started'
            response.message = 'VLA orchestration started'
            response.success = True
            response.estimated_duration = self.estimate_orchestration_duration(orchestration_context['command'])
            response.timestamp = self.get_clock().now().to_msg()

            self.get_logger().info(f'Orchestration started: {orchestration_id}')

        except Exception as e:
            self.get_logger().error(f'Error in orchestration service: {str(e)}')
            response.execution_id = f'orch_error_{int(time.time())}'
            response.status = 'failed'
            response.message = f'Orchestration error: {str(e)}'
            response.success = False
            response.estimated_duration = 0.0
            response.timestamp = self.get_clock().now().to_msg()

        return response

    def start_orchestration_process(self, orchestration_context: Dict[str, Any]):
        """
        Start the orchestration process for a given context
        """
        try:
            # Process voice command to generate plan
            self.process_voice_command(orchestration_context)

        except Exception as e:
            self.get_logger().error(f'Error starting orchestration process: {str(e)}')
            orchestration_context['state'] = OrchestratorState.FAILED
            orchestration_context['failed_steps'].append('initialization')

    def process_voice_command(self, orchestration_context: Dict[str, Any]):
        """
        Process voice command and generate action plan
        """
        try:
            orchestration_context['state'] = OrchestratorState.GENERATING_PLAN

            # Create plan generation request
            plan_request = GenerateActionPlan.Request()
            plan_request.command = orchestration_context['command']
            plan_request.execution_context = json.dumps(orchestration_context['execution_context'])

            # Call plan generation service
            future = self.plan_service_client.call_async(plan_request)
            # In a real implementation, we'd handle this asynchronously
            # For simulation, we'll create a basic plan
            basic_plan = self.create_basic_plan(orchestration_context['command'])
            orchestration_context['plan'] = basic_plan
            orchestration_context['completed_steps'].append('planning')
            orchestration_context['step_results']['planning'] = {'success': True, 'plan': basic_plan}

            # Process vision data if needed
            self.process_vision_data(orchestration_context)

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {str(e)}')
            orchestration_context['state'] = OrchestratorState.FAILED
            orchestration_context['failed_steps'].append('planning')

    def process_vision_data(self, orchestration_context: Dict[str, Any]):
        """
        Process vision data for the orchestration
        """
        try:
            orchestration_context['state'] = OrchestratorState.PROCESSING_VISION

            # In a real implementation, this would call vision services
            # For simulation, we'll just create mock vision data
            vision_data = VisionData()
            vision_data.data_id = f'vision_{int(time.time())}'
            orchestration_context['vision_data'] = vision_data
            orchestration_context['completed_steps'].append('vision')
            orchestration_context['step_results']['vision'] = {'success': True, 'data': vision_data}

            # Execute the plan
            self.execute_plan(orchestration_context)

        except Exception as e:
            self.get_logger().error(f'Error processing vision data: {str(e)}')
            orchestration_context['state'] = OrchestratorState.FAILED
            orchestration_context['failed_steps'].append('vision')

    def execute_plan(self, orchestration_context: Dict[str, Any]):
        """
        Execute the generated action plan
        """
        try:
            orchestration_context['state'] = OrchestratorState.EXECUTING_PLAN

            if orchestration_context['plan'] is None:
                self.get_logger().error('No plan to execute')
                orchestration_context['state'] = OrchestratorState.FAILED
                orchestration_context['failed_steps'].append('execution')
                return

            # Create execution request
            exec_request = ExecutePlan.Request()
            exec_request.plan = orchestration_context['plan']
            exec_request.execution_context = json.dumps(orchestration_context['execution_context'])

            # Call execution service
            future = self.execution_service_client.call_async(exec_request)
            # In a real implementation, we'd handle this asynchronously
            # For simulation, we'll just mark as completed
            orchestration_context['completed_steps'].append('execution')
            orchestration_context['step_results']['execution'] = {'success': True, 'result': 'completed'}
            orchestration_context['state'] = OrchestratorState.COMPLETED
            orchestration_context['execution_result'] = 'success'

            self.get_logger().info(f'Plan execution completed for orchestration: {orchestration_context["orchestration_id"]}')

        except Exception as e:
            self.get_logger().error(f'Error executing plan: {str(e)}')
            orchestration_context['state'] = OrchestratorState.FAILED
            orchestration_context['failed_steps'].append('execution')

    def create_basic_plan(self, command: str) -> ActionPlan:
        """
        Create a basic action plan for simulation purposes
        """
        try:
            plan = ActionPlan()
            plan.plan_id = f'plan_{int(time.time())}'

            # Create tasks based on command
            command_lower = command.lower()
            if 'navigate' in command_lower or 'go to' in command_lower:
                task = Task()
                task.task_id = f'nav_task_{int(time.time())}'
                task.type = 'navigate'
                task.description = f'Navigate based on command: {command}'
                plan.tasks.append(task)
            elif 'find' in command_lower or 'locate' in command_lower:
                task = Task()
                task.task_id = f'detect_task_{int(time.time())}'
                task.type = 'detect_object'
                task.description = f'Detect object based on command: {command}'
                plan.tasks.append(task)
            elif 'grasp' in command_lower or 'pick up' in command_lower:
                task = Task()
                task.task_id = f'grasp_task_{int(time.time())}'
                task.type = 'grasp_object'
                task.description = f'Grasp object based on command: {command}'
                plan.tasks.append(task)
            else:
                # Default task
                task = Task()
                task.task_id = f'default_task_{int(time.time())}'
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

    def orchestration_monitor_callback(self):
        """
        Monitor active orchestrations and update their status
        """
        try:
            current_time = time.time()

            for orch_id, context in list(self.active_orchestrations.items()):
                # Check for timeout
                if current_time - context['start_time'] > self.orchestration_timeout:
                    context['state'] = OrchestratorState.FAILED
                    self.get_logger().warn(f'Orchestration {orch_id} timed out')

                # Update status
                self.publish_orchestration_status(context)

                # Check if orchestration is complete
                if context['state'] in [OrchestratorState.COMPLETED, OrchestratorState.FAILED]:
                    # Move to history
                    self.orchestration_history.append(context)
                    # Keep only recent history
                    if len(self.orchestration_history) > 50:  # Keep last 50 orchestrations
                        self.orchestration_history = self.orchestration_history[-50:]
                    # Remove from active
                    del self.active_orchestrations[orch_id]

        except Exception as e:
            self.get_logger().error(f'Error in orchestration monitor: {str(e)}')

    def publish_orchestration_status(self, context: Dict[str, Any]):
        """
        Publish orchestration status updates
        """
        try:
            status_msg = ExecutionStatus()
            status_msg.header.stamp = self.get_clock().now().to_msg()
            status_msg.header.frame_id = 'vla_orchestrator'
            status_msg.execution_id = context['orchestration_id']
            status_msg.plan_id = context['plan'].plan_id if context['plan'] else 'unknown'
            status_msg.overall_status = context['state'].value
            status_msg.completed_tasks = len(context['completed_steps'])
            status_msg.total_tasks = context['total_steps']

            if status_msg.total_tasks > 0:
                status_msg.progress = float(status_msg.completed_tasks) / status_msg.total_tasks
            else:
                status_msg.progress = 0.0

            status_msg.error = ', '.join(context['failed_steps']) if context['failed_steps'] else ''

            self.orchestration_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing orchestration status: {str(e)}')

    def estimate_orchestration_duration(self, command: str) -> float:
        """
        Estimate the total duration for an orchestration
        """
        try:
            # Base time + additional time based on command complexity
            base_time = 10.0
            complexity_factor = 1.0

            command_lower = command.lower()
            if 'complex' in command_lower or 'multiple' in command_lower:
                complexity_factor = 2.0
            elif 'simple' in command_lower or 'single' in command_lower:
                complexity_factor = 0.5

            return base_time * complexity_factor

        except Exception as e:
            self.get_logger().error(f'Error estimating orchestration duration: {str(e)}')
            return 30.0  # Default estimate

    def get_orchestration_status(self, orchestration_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific orchestration
        """
        try:
            return self.active_orchestrations.get(orchestration_id, None)
        except Exception as e:
            self.get_logger().error(f'Error getting orchestration status: {str(e)}')
            return None

    def get_active_orchestrations(self) -> List[str]:
        """
        Get list of active orchestration IDs
        """
        try:
            return list(self.active_orchestrations.keys())
        except Exception as e:
            self.get_logger().error(f'Error getting active orchestrations: {str(e)}')
            return []

    def get_orchestration_history(self) -> List[Dict[str, Any]]:
        """
        Get the orchestration history
        """
        try:
            return self.orchestration_history.copy()
        except Exception as e:
            self.get_logger().error(f'Error getting orchestration history: {str(e)}')
            return []

    def cancel_orchestration(self, orchestration_id: str) -> bool:
        """
        Cancel a specific orchestration
        """
        try:
            if orchestration_id in self.active_orchestrations:
                self.active_orchestrations[orchestration_id]['state'] = OrchestratorState.FAILED
                self.active_orchestrations[orchestration_id]['message'] = 'Cancelled by user'
                self.get_logger().info(f'Orchestration {orchestration_id} cancelled')
                return True
            return False
        except Exception as e:
            self.get_logger().error(f'Error cancelling orchestration: {str(e)}')
            return False

    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """
        Get orchestration statistics
        """
        try:
            active_count = len(self.active_orchestrations)
            history_count = len(self.orchestration_history)

            stats = {
                'active_orchestrations': active_count,
                'completed_orchestrations': history_count,
                'total_orchestrations': active_count + history_count,
                'average_duration': self.calculate_average_duration(),
                'success_rate': self.calculate_success_rate(),
                'step_success_rates': self.calculate_step_success_rates()
            }

            return stats

        except Exception as e:
            self.get_logger().error(f'Error getting orchestration statistics: {str(e)}')
            return {}

    def calculate_average_duration(self) -> float:
        """
        Calculate average duration of completed orchestrations
        """
        try:
            completed_orchestrations = [
                orchestration for orchestration in self.orchestration_history
                if orchestration['state'] == OrchestratorState.COMPLETED
            ]

            if not completed_orchestrations:
                return 0.0

            total_duration = sum(
                orchestration.get('end_time', 0) - orchestration.get('start_time', 0)
                for orchestration in completed_orchestrations
            )

            return total_duration / len(completed_orchestrations)

        except Exception as e:
            self.get_logger().error(f'Error calculating average duration: {str(e)}')
            return 0.0

    def calculate_success_rate(self) -> float:
        """
        Calculate success rate of orchestrations
        """
        try:
            if not self.orchestration_history:
                return 0.0

            completed_count = sum(
                1 for orchestration in self.orchestration_history
                if orchestration['state'] == OrchestratorState.COMPLETED
            )

            return completed_count / len(self.orchestration_history)

        except Exception as e:
            self.get_logger().error(f'Error calculating success rate: {str(e)}')
            return 0.0

    def calculate_step_success_rates(self) -> Dict[str, float]:
        """
        Calculate success rates for each orchestration step
        """
        try:
            if not self.orchestration_history:
                return {}

            step_counts = {
                'planning': 0,
                'vision': 0,
                'execution': 0
            }
            step_successes = {
                'planning': 0,
                'vision': 0,
                'execution': 0
            }

            for orchestration in self.orchestration_history:
                steps = orchestration.get('step_results', {})
                for step, result in steps.items():
                    if step in step_counts:
                        step_counts[step] += 1
                        if result.get('success', False):
                            step_successes[step] += 1

            success_rates = {}
            for step in step_counts:
                if step_counts[step] > 0:
                    success_rates[step] = step_successes[step] / step_counts[step]
                else:
                    success_rates[step] = 0.0

            return success_rates

        except Exception as e:
            self.get_logger().error(f'Error calculating step success rates: {str(e)}')
            return {}

    def reset_orchestrator_node(self):
        """
        Reset the orchestrator node to initial state
        """
        try:
            self.active_orchestrations.clear()
            self.orchestration_queue.clear()
            self.orchestration_history.clear()
            self.get_logger().info('VLA Orchestrator node reset')
        except Exception as e:
            self.get_logger().error(f'Error resetting orchestrator node: {str(e)}')

    def queue_orchestration(self, orchestration_request: Dict[str, Any]):
        """
        Queue an orchestration request for processing
        """
        try:
            self.orchestration_queue.append(orchestration_request)
            self.get_logger().info(f'Queued orchestration request, queue size: {len(self.orchestration_queue)}')
        except Exception as e:
            self.get_logger().error(f'Error queuing orchestration: {str(e)}')

    def is_orchestration_active(self) -> bool:
        """
        Check if any orchestration is currently active
        """
        return len(self.active_orchestrations) > 0


def main(args=None):
    rclpy.init(args=args)

    vla_orchestrator = VLAOrchestratorNode()

    try:
        rclpy.spin(vla_orchestrator)
    except KeyboardInterrupt:
        pass
    finally:
        vla_orchestrator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()