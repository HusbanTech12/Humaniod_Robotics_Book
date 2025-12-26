#!/usr/bin/env python3

"""
Voice Command Service Node for Vision-Language-Action (VLA) Module
API contract implementation for /voice/process_command
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import json
import time
from typing import Dict, Any

# Import custom message types
from vla_integration.msg import VoiceCommand, ActionPlan
from vla_integration.srv import GenerateActionPlan
from std_msgs.msg import String


class VoiceCommandServiceNode(Node):
    """
    ROS 2 Node implementing the voice command processing API service
    """

    def __init__(self):
        super().__init__('voice_command_service')

        # Declare parameters
        self.declare_parameter('service_timeout', 10.0)
        self.declare_parameter('max_concurrent_requests', 5)
        self.declare_parameter('transcription_accuracy_target', 0.90)

        # Get parameters
        self.service_timeout = self.get_parameter('service_timeout').value
        self.max_concurrent_requests = self.get_parameter('max_concurrent_requests').value
        self.transcription_accuracy_target = self.get_parameter('transcription_accuracy_target').value

        # Create service for processing voice commands
        self.voice_process_service = self.create_service(
            GenerateActionPlan,  # Using this service type for voice processing
            'vla/voice/process_command',
            self.process_voice_command_callback
        )

        # Create publisher for voice commands
        self.voice_cmd_pub = self.create_publisher(VoiceCommand, 'vla/voice/command_raw', 10)

        # Create publisher for action plans
        self.action_plan_pub = self.create_publisher(ActionPlan, 'vla/action_plan_from_voice', 10)

        self.get_logger().info('Voice Command Service Node initialized')

    def process_voice_command_callback(self, request, response):
        """
        Service callback for processing voice commands
        Expected request format based on API contract:
        {
          "audio_data": "base64_encoded_audio",
          "command_text": "Pick up the red cup",
          "timestamp": "2025-12-25T10:00:00Z"
        }

        Expected response format:
        {
          "command_id": "cmd_12345",
          "status": "processing",
          "estimated_completion": "2.5",
          "action_plan": { ... }
        }
        """
        try:
            self.get_logger().info(f'Processing voice command: {request.command}')

            # Create VoiceCommand message
            cmd_msg = VoiceCommand()
            cmd_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_msg.header.frame_id = 'voice_service'
            cmd_msg.command_id = f'cmd_{int(time.time())}'

            # If command_text is provided in request, use it
            if request.command.strip():
                cmd_msg.transcribed_text = request.command
                cmd_msg.confidence = 1.0  # Text was provided directly
            else:
                # If no text provided, we'd normally process audio_data here
                cmd_msg.transcribed_text = ""
                cmd_msg.confidence = 0.0

            cmd_msg.language = 'en'  # Default language
            cmd_msg.timestamp = self.get_clock().now().to_msg()

            # Publish the voice command
            self.voice_cmd_pub.publish(cmd_msg)

            # Generate an action plan based on the command
            action_plan = self.generate_action_plan_from_command(cmd_msg.transcribed_text)

            if action_plan:
                # Create response
                response.plan = action_plan
                response.success = True
                response.message = 'Voice command processed successfully'

                # Publish the action plan
                self.action_plan_pub.publish(action_plan)

                self.get_logger().info(f'Voice command processed successfully: {cmd_msg.command_id}')
            else:
                response.success = False
                response.message = 'Failed to generate action plan from command'
                self.get_logger().error('Failed to generate action plan from voice command')

        except Exception as e:
            self.get_logger().error(f'Error in voice command processing: {str(e)}')
            response.success = False
            response.message = f'Error processing voice command: {str(e)}'

        return response

    def generate_action_plan_from_command(self, command: str) -> ActionPlan:
        """
        Generate an action plan from a natural language command
        This is a simplified implementation - in reality, this would connect to the LLM planner
        """
        try:
            if not command.strip():
                return None

            # Create a basic action plan based on common command patterns
            plan = ActionPlan()
            plan.header.stamp = self.get_clock().now().to_msg()
            plan.header.frame_id = 'voice_service'
            plan.plan_id = f'plan_{int(time.time())}_from_voice'
            plan.command_id = f'cmd_{int(time.time())}'
            plan.created_at = self.get_clock().now().to_msg()
            plan.updated_at = self.get_clock().now().to_msg()
            plan.status = 'planned'

            # Simple command parsing to generate appropriate tasks
            command_lower = command.lower()

            # Create tasks based on command content
            tasks = []

            # Detect object-related tasks
            if 'cup' in command_lower or 'glass' in command_lower or 'bottle' in command_lower:
                detect_task = self.create_task(
                    task_id=f'task_detect_{int(time.time())}',
                    task_type='detect_object',
                    description='Detect the requested object',
                    parameters={'object_type': 'drinkware'}
                )
                tasks.append(detect_task)

            # Detect navigation tasks
            if 'go to' in command_lower or 'move to' in command_lower or 'navigate to' in command_lower:
                navigate_task = self.create_task(
                    task_id=f'task_navigate_{int(time.time())}',
                    task_type='navigate',
                    description='Navigate to the specified location',
                    parameters={'target_location': 'parsed_from_command'}
                )
                tasks.append(navigate_task)

            # Detect manipulation tasks
            if 'pick up' in command_lower or 'grasp' in command_lower or 'take' in command_lower:
                grasp_task = self.create_task(
                    task_id=f'task_grasp_{int(time.time())}',
                    task_type='grasp_object',
                    description='Grasp the specified object',
                    parameters={'object_id': 'detected_object_id'}
                )
                tasks.append(grasp_task)

            # Detect placement tasks
            if 'put' in command_lower or 'place' in command_lower or 'drop' in command_lower:
                place_task = self.create_task(
                    task_id=f'task_place_{int(time.time())}',
                    task_type='place_object',
                    description='Place the object at the destination',
                    parameters={'destination': 'parsed_from_command'}
                )
                tasks.append(place_task)

            # If no specific tasks were identified, create a generic task
            if not tasks:
                generic_task = self.create_task(
                    task_id=f'task_generic_{int(time.time())}',
                    task_type='execute_command',
                    description=f'Execute command: {command}',
                    parameters={'raw_command': command}
                )
                tasks.append(generic_task)

            plan.tasks = tasks
            return plan

        except Exception as e:
            self.get_logger().error(f'Error generating action plan: {str(e)}')
            return None

    def create_task(self, task_id: str, task_type: str, description: str, parameters: Dict[str, str]) -> Any:
        """
        Create a Task message with the given parameters
        """
        from vla_integration.msg import Task

        task = Task()
        task.task_id = task_id
        task.type = task_type
        task.description = description
        task.priority = 1  # Default priority
        task.status = 'pending'

        # Convert parameters dictionary to key-value pairs
        for key, value in parameters.items():
            kv = Task.KeyValue()
            kv.key = str(key)
            kv.value = str(value)
            task.parameters.append(kv)

        return task


def main(args=None):
    rclpy.init(args=args)

    voice_service = VoiceCommandServiceNode()

    try:
        rclpy.spin(voice_service)
    except KeyboardInterrupt:
        pass
    finally:
        voice_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()