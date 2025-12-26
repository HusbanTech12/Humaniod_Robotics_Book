#!/usr/bin/env python3

"""
LLM Planner Node for Vision-Language-Action (VLA) Module
Handles natural language processing and task decomposition using LLMs
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import openai
import json
import time
from typing import Dict, List, Any

# Import custom message types
from vla_integration.msg import VoiceCommand, ActionPlan, Task
from vla_integration.srv import GenerateActionPlan


class LLMPlannerNode(Node):
    """
    ROS 2 Node for processing natural language commands and generating action plans using LLMs
    """

    def __init__(self):
        super().__init__('llm_planner')

        # Declare parameters
        self.declare_parameter('llm_model', 'gpt-4')
        self.declare_parameter('api_key', '')
        self.declare_parameter('max_plan_steps', 20)
        self.declare_parameter('temperature', 0.3)
        self.declare_parameter('max_tokens', 1000)

        # Get parameters
        self.llm_model = self.get_parameter('llm_model').value
        self.api_key = self.get_parameter('api_key').value
        self.max_plan_steps = self.get_parameter('max_plan_steps').value
        self.temperature = self.get_parameter('temperature').value
        self.max_tokens = self.get_parameter('max_tokens').value

        # Set OpenAI API key if provided
        if self.api_key:
            openai.api_key = self.api_key

        # Create subscriber for voice commands
        self.voice_cmd_sub = self.create_subscription(
            VoiceCommand,
            'vla/voice/command',
            self.voice_command_callback,
            10
        )

        # Create publisher for action plans
        self.action_plan_pub = self.create_publisher(ActionPlan, 'vla/action_plan', 10)

        # Create service for generating action plans
        self.plan_service = self.create_service(
            GenerateActionPlan,
            'vla/generate_action_plan',
            self.generate_plan_callback
        )

        self.get_logger().info('LLM Planner Node initialized')

    def voice_command_callback(self, msg: VoiceCommand):
        """Callback for processing voice commands and generating action plans"""
        try:
            self.get_logger().info(f'Processing voice command: {msg.transcribed_text}')

            # Generate action plan from voice command
            plan = self.generate_action_plan(msg.transcribed_text)

            if plan:
                # Publish the action plan
                plan_msg = self.create_action_plan_msg(plan, msg.command_id)
                self.action_plan_pub.publish(plan_msg)
                self.get_logger().info(f'Action plan published with ID: {plan_msg.plan_id}')
            else:
                self.get_logger().error('Failed to generate action plan')

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {str(e)}')

    def generate_plan_callback(self, request, response):
        """Service callback for generating action plans"""
        try:
            self.get_logger().info(f'Generating action plan for command: {request.command}')

            # Generate action plan
            plan = self.generate_action_plan(request.command, request.context)

            if plan:
                # Create response
                response.plan = self.create_action_plan_msg(plan, f'plan_{int(time.time())}')
                response.success = True
                response.message = 'Action plan generated successfully'
                response.confidence = 0.95  # Placeholder confidence value
            else:
                response.success = False
                response.message = 'Failed to generate action plan'
                response.confidence = 0.0

        except Exception as e:
            self.get_logger().error(f'Error in plan generation service: {str(e)}')
            response.success = False
            response.message = f'Error generating plan: {str(e)}'
            response.confidence = 0.0

        return response

    def generate_action_plan(self, command: str, context: str = "") -> Dict[str, Any]:
        """
        Generate action plan from natural language command using LLM
        """
        try:
            # Define the system prompt for task decomposition
            system_prompt = """
            You are an AI assistant for a Vision-Language-Action (VLA) system that controls a humanoid robot.
            Your task is to decompose high-level natural language commands into structured action plans.

            The output should be a JSON object with the following structure:
            {
                "plan_id": "unique_plan_id",
                "command": "original command",
                "tasks": [
                    {
                        "task_id": "unique_task_id",
                        "type": "task_type (navigate, detect_object, grasp, etc.)",
                        "description": "brief description of the task",
                        "parameters": {"key": "value"},
                        "priority": 1 (highest) to 10 (lowest),
                        "dependencies": ["task_id_1", "task_id_2"]
                    }
                ]
            }

            Available task types: navigate, detect_object, grasp_object, place_object, approach_object,
            manipulate_object, wait, move_arm, move_gripper, look_at, follow_path

            Be specific with object names, locations, and parameters. Ensure the plan is executable by a robot.
            """

            # Prepare the user prompt
            user_prompt = f"Command: {command}\nContext: {context}\n\nGenerate a structured action plan in JSON format:"

            # Call the LLM
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )

            # Extract the plan from the response
            plan_json = response.choices[0].message['content']
            plan = json.loads(plan_json)

            self.get_logger().info(f'Generated action plan: {plan}')
            return plan

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Error parsing LLM response as JSON: {str(e)}')
            return None
        except Exception as e:
            self.get_logger().error(f'Error generating action plan: {str(e)}')
            return None

    def create_action_plan_msg(self, plan_data: Dict[str, Any], original_command_id: str) -> ActionPlan:
        """
        Create an ActionPlan message from plan data
        """
        plan_msg = ActionPlan()
        plan_msg.header.stamp = self.get_clock().now().to_msg()
        plan_msg.header.frame_id = 'llm_planner'
        plan_msg.plan_id = plan_data.get('plan_id', f'plan_{int(time.time())}')
        plan_msg.command_id = original_command_id
        plan_msg.created_at = self.get_clock().now().to_msg()
        plan_msg.updated_at = self.get_clock().now().to_msg()
        plan_msg.status = 'planned'

        # Create task messages
        tasks = []
        for task_data in plan_data.get('tasks', []):
            task_msg = Task()
            task_msg.task_id = task_data.get('task_id', f'task_{len(tasks)}')
            task_msg.type = task_data.get('type', 'unknown')
            task_msg.description = task_data.get('description', '')
            task_msg.priority = min(10, max(1, task_data.get('priority', 5)))  # Clamp between 1-10
            task_msg.status = 'pending'
            task_msg.dependencies = task_data.get('dependencies', [])

            # Convert parameters to key-value pairs
            params = task_data.get('parameters', {})
            for key, value in params.items():
                kv = Task.KeyValue()
                kv.key = str(key)
                kv.value = str(value)
                task_msg.parameters.append(kv)

            tasks.append(task_msg)

        plan_msg.tasks = tasks
        return plan_msg


def main(args=None):
    rclpy.init(args=args)

    llm_planner = LLMPlannerNode()

    try:
        rclpy.spin(llm_planner)
    except KeyboardInterrupt:
        pass
    finally:
        llm_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()