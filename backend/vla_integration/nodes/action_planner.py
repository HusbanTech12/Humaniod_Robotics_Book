#!/usr/bin/env python3

"""
Action Planner Node for Vision-Language-Action (VLA) Module
Handles structured plan generation and validation
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import json
from typing import Dict, List, Any

# Import custom message types
from vla_integration.msg import ActionPlan, Task
from vla_integration.srv import GenerateActionPlan


class ActionPlannerNode(Node):
    """
    ROS 2 Node for validating and refining action plans
    """

    def __init__(self):
        super().__init__('action_planner')

        # Declare parameters
        self.declare_parameter('max_plan_steps', 50)
        self.declare_parameter('plan_validation_enabled', True)
        self.declare_parameter('safety_constraints_enabled', True)

        # Get parameters
        self.max_plan_steps = self.get_parameter('max_plan_steps').value
        self.plan_validation_enabled = self.get_parameter('plan_validation_enabled').value
        self.safety_constraints_enabled = self.get_parameter('safety_constraints_enabled').value

        # Create subscriber for action plans from LLM planner
        self.action_plan_sub = self.create_subscription(
            ActionPlan,
            'vla/action_plan',
            self.action_plan_callback,
            10
        )

        # Create publisher for validated action plans
        self.validated_plan_pub = self.create_publisher(ActionPlan, 'vla/validated_action_plan', 10)

        self.get_logger().info('Action Planner Node initialized')

    def action_plan_callback(self, msg: ActionPlan):
        """Callback for processing action plans and validating them"""
        try:
            self.get_logger().info(f'Validating action plan: {msg.plan_id}')

            # Validate the action plan
            is_valid, validation_errors = self.validate_action_plan(msg)

            if is_valid:
                # If plan is valid, publish it
                self.validated_plan_pub.publish(msg)
                self.get_logger().info(f'Action plan validated and published: {msg.plan_id}')
            else:
                self.get_logger().error(f'Action plan validation failed: {validation_errors}')
                # In a real system, we might want to send the plan back for correction

        except Exception as e:
            self.get_logger().error(f'Error validating action plan: {str(e)}')

    def validate_action_plan(self, plan: ActionPlan) -> (bool, List[str]):
        """
        Validate an action plan against various constraints
        Returns: (is_valid, list_of_errors)
        """
        errors = []

        # Check number of steps
        if len(plan.tasks) > self.max_plan_steps:
            errors.append(f"Plan has {len(plan.tasks)} tasks, which exceeds maximum of {self.max_plan_steps}")

        # Validate each task
        for task in plan.tasks:
            # Validate task type
            valid_task_types = [
                'navigate', 'detect_object', 'grasp_object', 'place_object', 'approach_object',
                'manipulate_object', 'wait', 'move_arm', 'move_gripper', 'look_at', 'follow_path'
            ]
            if task.type not in valid_task_types:
                errors.append(f"Invalid task type '{task.type}' in task {task.task_id}")

            # Validate priority
            if task.priority < 1 or task.priority > 10:
                errors.append(f"Invalid priority {task.priority} for task {task.task_id} (must be 1-10)")

            # Validate dependencies exist
            for dep_id in task.dependencies:
                if not any(t.task_id == dep_id for t in plan.tasks):
                    errors.append(f"Dependency {dep_id} for task {task.task_id} does not exist in plan")

        # Check for circular dependencies
        if self.has_circular_dependencies(plan):
            errors.append("Plan contains circular dependencies")

        return len(errors) == 0, errors

    def has_circular_dependencies(self, plan: ActionPlan) -> bool:
        """
        Check if the plan has circular dependencies
        """
        # Build dependency graph
        graph = {}
        for task in plan.tasks:
            graph[task.task_id] = set(task.dependencies)

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for task in plan.tasks:
            if task.task_id not in visited:
                if dfs(task.task_id):
                    return True

        return False

    def refine_action_plan(self, plan: ActionPlan) -> ActionPlan:
        """
        Refine an action plan by optimizing task order and checking constraints
        """
        # Create a copy of the plan
        refined_plan = ActionPlan()
        refined_plan.header = plan.header
        refined_plan.plan_id = plan.plan_id
        refined_plan.command_id = plan.command_id
        refined_plan.created_at = plan.created_at
        refined_plan.updated_at = plan.updated_at
        refined_plan.status = plan.status

        # Sort tasks by priority (lower number = higher priority)
        sorted_tasks = sorted(plan.tasks, key=lambda t: t.priority)

        # Add refined tasks to the plan
        refined_plan.tasks = sorted_tasks

        return refined_plan


def main(args=None):
    rclpy.init(args=args)

    action_planner = ActionPlannerNode()

    try:
        rclpy.spin(action_planner)
    except KeyboardInterrupt:
        pass
    finally:
        action_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()