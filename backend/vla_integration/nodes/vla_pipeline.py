#!/usr/bin/env python3

"""
VLA Pipeline Node for Vision-Language-Action (VLA) Module
Main orchestrator for the VLA pipeline connecting voice, vision, and action components
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
from typing import Dict, List, Any

# Import custom message types
from vla_integration.msg import VoiceCommand, ActionPlan, ExecutionStatus
from vla_integration.srv import GenerateActionPlan, ExecutePlan


class VLAPipelineNode(Node):
    """
    ROS 2 Node that orchestrates the complete VLA pipeline
    """

    def __init__(self):
        super().__init__('vla_pipeline')

        # Declare parameters
        self.declare_parameter('pipeline_timeout', 300.0)  # 5 minutes
        self.declare_parameter('max_concurrent_plans', 1)
        self.declare_parameter('retry_attempts', 3)
        self.declare_parameter('recovery_enabled', True)

        # Get parameters
        self.pipeline_timeout = self.get_parameter('pipeline_timeout').value
        self.max_concurrent_plans = self.get_parameter('max_concurrent_plans').value
        self.retry_attempts = self.get_parameter('retry_attempts').value
        self.recovery_enabled = self.get_parameter('recovery_enabled').value

        # Create subscribers
        self.voice_cmd_sub = self.create_subscription(
            VoiceCommand,
            'vla/voice/command',
            self.voice_command_callback,
            10
        )

        # Create publishers
        self.execution_status_pub = self.create_publisher(ExecutionStatus, 'vla/execution_status', 10)

        # Create clients for services
        self.plan_client = self.create_client(GenerateActionPlan, 'vla/generate_action_plan')
        self.execute_client = self.create_client(ExecutePlan, 'vla/execute_plan')

        # Wait for services to be available
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for generate_action_plan service...')

        while not self.execute_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for execute_plan service...')

        # Track ongoing executions
        self.active_executions = {}

        self.get_logger().info('VLA Pipeline Node initialized')

    def voice_command_callback(self, msg: VoiceCommand):
        """Callback for processing voice commands through the complete VLA pipeline"""
        try:
            self.get_logger().info(f'Received voice command: {msg.transcribed_text}')

            # Generate action plan
            plan_future = self.generate_action_plan_async(msg.transcribed_text)
            if plan_future:
                # Execute the plan
                execution_result = self.execute_plan_async(plan_future.result().plan)

                if execution_result.success:
                    self.get_logger().info(f'Pipeline completed successfully: {execution_result.execution_id}')
                else:
                    self.get_logger().error(f'Pipeline execution failed: {execution_result.message}')
            else:
                self.get_logger().error('Failed to generate action plan from voice command')

        except Exception as e:
            self.get_logger().error(f'Error in VLA pipeline: {str(e)}')

    def generate_action_plan_async(self, command: str) -> Any:
        """Asynchronously generate action plan from command"""
        try:
            request = GenerateActionPlan.Request()
            request.command = command
            request.context = '{}'  # Empty context for now

            future = self.plan_client.call_async(request)

            # Wait for response (in a real system, this would be handled asynchronously)
            rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

            return future.result()
        except Exception as e:
            self.get_logger().error(f'Error calling generate_action_plan service: {str(e)}')
            return None

    def execute_plan_async(self, plan: ActionPlan) -> Any:
        """Asynchronously execute action plan"""
        try:
            request = ExecutePlan.Request()
            request.plan = plan
            request.execution_context = '{"robot_id": "humanoid_001", "environment_id": "kitchen_01"}'

            future = self.execute_client.call_async(request)

            # Wait for response (in a real system, this would be handled asynchronously)
            rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)

            return future.result()
        except Exception as e:
            self.get_logger().error(f'Error calling execute_plan service: {str(e)}')
            return None

    def execute_pipeline(self, command: str) -> bool:
        """
        Execute the complete VLA pipeline: voice -> language -> action
        """
        try:
            self.get_logger().info(f'Starting VLA pipeline for command: {command}')

            # Step 1: Generate action plan
            plan_future = self.generate_action_plan_async(command)
            if not plan_future or not plan_future.success:
                self.get_logger().error('Failed to generate action plan')
                return False

            plan = plan_future.plan
            self.get_logger().info(f'Generated action plan with {len(plan.tasks)} tasks')

            # Step 2: Execute the plan
            execution_result = self.execute_plan_async(plan)
            if not execution_result or not execution_result.success:
                self.get_logger().error(f'Plan execution failed: {execution_result.message if execution_result else "Unknown error"}')
                return False

            self.get_logger().info(f'Pipeline executed successfully: {execution_result.execution_id}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in pipeline execution: {str(e)}')
            return False


def main(args=None):
    rclpy.init(args=args)

    vla_pipeline = VLAPipelineNode()

    try:
        rclpy.spin(vla_pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        vla_pipeline.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()