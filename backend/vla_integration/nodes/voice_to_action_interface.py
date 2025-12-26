#!/usr/bin/env python3

"""
Voice-to-Action Interface Node for Vision-Language-Action (VLA) Module
Connects voice processing to action execution
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
from vla_integration.srv import ExecutePlan, GenerateActionPlan


class VoiceToActionInterfaceNode(Node):
    """
    ROS 2 Node for connecting voice processing to action execution
    """

    def __init__(self):
        super().__init__('voice_to_action_interface')

        # Declare parameters
        self.declare_parameter('interface_timeout', 300.0)  # 5 minutes
        self.declare_parameter('max_concurrent_interfaces', 3)
        self.declare_parameter('interface_rate', 10.0)  # Hz
        self.declare_parameter('auto_execute_enabled', True)

        # Get parameters
        self.interface_timeout = self.get_parameter('interface_timeout').value
        self.max_concurrent_interfaces = self.get_parameter('max_concurrent_interfaces').value
        self.interface_rate = self.get_parameter('interface_rate').value
        self.auto_execute_enabled = self.get_parameter('auto_execute_enabled').value

        # Create service for voice-to-action interface
        self.voice_to_action_service = self.create_service(
            GenerateActionPlan,
            'vla/voice_to_action',
            self.voice_to_action_callback
        )

        # Create client for plan generation
        self.plan_generation_client = self.create_client(GenerateActionPlan, 'vla/generate_plan')
        while not self.plan_generation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for plan generation service...')

        # Create client for action execution
        self.action_execution_client = self.create_client(ExecutePlan, 'vla/actions/execute_plan')
        while not self.action_execution_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for action execution service...')

        # Create publisher for interface status
        self.interface_status_pub = self.create_publisher(ExecutionStatus, 'vla/voice_to_action_status', 10)

        # Initialize interface state
        self.active_interfaces: Dict[str, Dict[str, Any]] = {}
        self.interface_history: List[Dict[str, Any]] = []

        # Create timer for interface monitoring
        self.interface_timer = self.create_timer(1.0/self.interface_rate, self.interface_monitor_callback)

        self.get_logger().info('Voice-to-Action Interface Node initialized')

    def voice_to_action_callback(self, request, response):
        """
        Service callback for voice-to-action interface
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
          "interface_id": "iface_12345",
          "action_plan": ActionPlan,
          "execution_id": "exec_67890",
          "status": "completed",
          "timestamp": "2025-12-25T10:00:05Z"
        }
        """
        try:
            interface_id = f'iface_{int(time.time())}_{hash(str(request.command)) % 10000}'
            self.get_logger().info(f'Processing voice-to-action for command: {request.command}, interface ID: {interface_id}')

            # Create interface context
            interface_context = {
                'interface_id': interface_id,
                'command': request.command,
                'execution_context': request.execution_context,
                'start_time': time.time(),
                'status': 'processing',
                'plan': None,
                'execution_id': None,
                'execution_result': None,
                'last_update': time.time()
            }

            # Store interface context
            self.active_interfaces[interface_id] = interface_context

            # Generate action plan from voice command
            plan = self.generate_plan_from_command(interface_context['command'], interface_context['execution_context'])
            interface_context['plan'] = plan

            # Execute the plan if auto-execute is enabled
            if self.auto_execute_enabled:
                execution_id = self.execute_plan(plan, interface_context['execution_context'])
                interface_context['execution_id'] = execution_id
                interface_context['status'] = 'executing'
            else:
                interface_context['status'] = 'plan_generated'

            # Prepare response
            response.interface_id = interface_id
            response.action_plan = plan
            response.execution_id = interface_context['execution_id'] or 'none'
            response.status = interface_context['status']
            response.timestamp = self.get_clock().now().to_msg()
            response.success = True
            response.message = f'Voice command processed, plan generated with {len(plan.tasks)} tasks'

            self.get_logger().info(f'Voice-to-action completed: {interface_id}')

        except Exception as e:
            self.get_logger().error(f'Error in voice-to-action interface: {str(e)}')
            response.interface_id = f'iface_error_{int(time.time())}'
            response.action_plan = ActionPlan()
            response.execution_id = 'none'
            response.status = 'failed'
            response.timestamp = self.get_clock().now().to_msg()
            response.success = False
            response.message = f'Voice-to-action error: {str(e)}'

        return response

    def generate_plan_from_command(self, command: str, execution_context: str) -> ActionPlan:
        """
        Generate an action plan from a voice command
        """
        try:
            # Create plan generation request
            plan_request = GenerateActionPlan.Request()
            plan_request.command = command
            plan_request.execution_context = execution_context

            # Call plan generation service
            future = self.plan_generation_client.call_async(plan_request)
            # In a real implementation, we'd wait for this asynchronously
            # For simulation, we'll create a plan based on the command

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
            elif 'place' in command_lower or 'put' in command_lower:
                task = Task()
                task.task_id = f'place_task_{int(time.time())}'
                task.type = 'place_object'
                task.description = f'Place object based on command: {command}'
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
            self.get_logger().error(f'Error generating plan from command: {str(e)}')
            # Return a minimal plan
            plan = ActionPlan()
            plan.plan_id = 'error_plan'
            task = Task()
            task.task_id = 'error_task'
            task.type = 'wait'
            task.description = 'Error recovery task'
            plan.tasks.append(task)
            return plan

    def execute_plan(self, plan: ActionPlan, execution_context: str) -> str:
        """
        Execute an action plan
        """
        try:
            # Create execution request
            exec_request = ExecutePlan.Request()
            exec_request.plan = plan
            exec_request.execution_context = execution_context

            # Call execution service
            future = self.action_execution_client.call_async(exec_request)
            # In a real implementation, we'd wait for this asynchronously
            # For simulation, we'll just return an execution ID

            execution_id = f'exec_{int(time.time())}'
            self.get_logger().info(f'Plan execution initiated: {execution_id}')
            return execution_id

        except Exception as e:
            self.get_logger().error(f'Error executing plan: {str(e)}')
            return f'exec_error_{int(time.time())}'

    def interface_monitor_callback(self):
        """
        Monitor active interfaces and update their status
        """
        try:
            current_time = time.time()

            for iface_id, context in list(self.active_interfaces.items()):
                # Check for timeout
                if current_time - context['start_time'] > self.interface_timeout:
                    context['status'] = 'timeout'
                    self.get_logger().warn(f'Interface {iface_id} timed out')

                # Update status
                self.publish_interface_status(context)

                # Check if interface is complete
                if context['status'] in ['completed', 'failed', 'timeout']:
                    # Move to history
                    self.interface_history.append(context)
                    # Keep only recent history
                    if len(self.interface_history) > 50:  # Keep last 50 interfaces
                        self.interface_history = self.interface_history[-50:]
                    # Remove from active
                    del self.active_interfaces[iface_id]

        except Exception as e:
            self.get_logger().error(f'Error in interface monitor: {str(e)}')

    def publish_interface_status(self, context: Dict[str, Any]):
        """
        Publish interface status updates
        """
        try:
            status_msg = ExecutionStatus()
            status_msg.header.stamp = self.get_clock().now().to_msg()
            status_msg.header.frame_id = 'voice_to_action_interface'
            status_msg.execution_id = context['interface_id']
            status_msg.plan_id = context['plan'].plan_id if context['plan'] else 'unknown'
            status_msg.overall_status = context['status']
            status_msg.completed_tasks = 1 if context['plan'] else 0
            status_msg.total_tasks = len(context['plan'].tasks) if context['plan'] else 1

            if status_msg.total_tasks > 0:
                status_msg.progress = float(status_msg.completed_tasks) / status_msg.total_tasks
            else:
                status_msg.progress = 0.0

            status_msg.error = ''

            self.interface_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing interface status: {str(e)}')

    def get_interface_status(self, interface_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific interface
        """
        try:
            return self.active_interfaces.get(interface_id, None)
        except Exception as e:
            self.get_logger().error(f'Error getting interface status: {str(e)}')
            return None

    def get_active_interfaces(self) -> List[str]:
        """
        Get list of active interface IDs
        """
        try:
            return list(self.active_interfaces.keys())
        except Exception as e:
            self.get_logger().error(f'Error getting active interfaces: {str(e)}')
            return []

    def get_interface_history(self) -> List[Dict[str, Any]]:
        """
        Get the interface history
        """
        try:
            return self.interface_history.copy()
        except Exception as e:
            self.get_logger().error(f'Error getting interface history: {str(e)}')
            return []

    def cancel_interface(self, interface_id: str) -> bool:
        """
        Cancel a specific interface operation
        """
        try:
            if interface_id in self.active_interfaces:
                self.active_interfaces[interface_id]['status'] = 'cancelled'
                self.get_logger().info(f'Interface {interface_id} cancelled')
                return True
            return False
        except Exception as e:
            self.get_logger().error(f'Error cancelling interface: {str(e)}')
            return False

    def get_interface_statistics(self) -> Dict[str, Any]:
        """
        Get interface statistics
        """
        try:
            active_count = len(self.active_interfaces)
            history_count = len(self.interface_history)

            stats = {
                'active_interfaces': active_count,
                'completed_interfaces': history_count,
                'total_interfaces': active_count + history_count,
                'auto_execute_enabled': self.auto_execute_enabled
            }

            return stats

        except Exception as e:
            self.get_logger().error(f'Error getting interface statistics: {str(e)}')
            return {}

    def process_voice_command(self, command: str, execution_context: str) -> Dict[str, Any]:
        """
        Process a voice command and return the result
        """
        try:
            # Generate plan from command
            plan = self.generate_plan_from_command(command, execution_context)

            # Execute if auto-execute is enabled
            execution_id = None
            if self.auto_execute_enabled:
                execution_id = self.execute_plan(plan, execution_context)

            result = {
                'command': command,
                'plan': plan,
                'execution_id': execution_id,
                'success': True,
                'message': f'Command processed successfully with {len(plan.tasks)} tasks'
            }

            return result

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {str(e)}')
            return {
                'command': command,
                'plan': ActionPlan(),
                'execution_id': None,
                'success': False,
                'message': f'Error processing command: {str(e)}'
            }

    def reset_interface_node(self):
        """
        Reset the interface node to initial state
        """
        try:
            self.active_interfaces.clear()
            self.interface_history.clear()
            self.get_logger().info('Voice-to-Action Interface node reset')
        except Exception as e:
            self.get_logger().error(f'Error resetting interface node: {str(e)}')

    def is_interface_active(self) -> bool:
        """
        Check if any interface is currently active
        """
        return len(self.active_interfaces) > 0

    def get_command_analysis(self, command: str) -> Dict[str, Any]:
        """
        Analyze a command to determine what types of tasks it would generate
        """
        try:
            analysis = {
                'command': command,
                'task_types': [],
                'estimated_complexity': 1,
                'keywords': []
            }

            command_lower = command.lower()
            keywords = command_lower.split()

            # Identify task types based on keywords
            if any(word in command_lower for word in ['navigate', 'go to', 'move to', 'walk to']):
                analysis['task_types'].append('navigate')
                analysis['estimated_complexity'] += 1
            if any(word in command_lower for word in ['find', 'locate', 'look for', 'search for']):
                analysis['task_types'].append('detect_object')
                analysis['estimated_complexity'] += 1
            if any(word in command_lower for word in ['grasp', 'pick up', 'grab', 'take']):
                analysis['task_types'].append('grasp_object')
                analysis['estimated_complexity'] += 1
            if any(word in command_lower for word in ['place', 'put', 'set down', 'drop']):
                analysis['task_types'].append('place_object')
                analysis['estimated_complexity'] += 1
            if any(word in command_lower for word in ['move', 'manipulate', 'push', 'pull']):
                analysis['task_types'].append('manipulate_object')
                analysis['estimated_complexity'] += 1

            analysis['keywords'] = keywords
            return analysis

        except Exception as e:
            self.get_logger().error(f'Error analyzing command: {str(e)}')
            return {
                'command': command,
                'task_types': ['wait'],
                'estimated_complexity': 1,
                'keywords': command.split()
            }


def main(args=None):
    rclpy.init(args=args)

    voice_to_action_interface = VoiceToActionInterfaceNode()

    try:
        rclpy.spin(voice_to_action_interface)
    except KeyboardInterrupt:
        pass
    finally:
        voice_to_action_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()