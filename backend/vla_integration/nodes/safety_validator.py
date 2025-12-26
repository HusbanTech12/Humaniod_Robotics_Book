#!/usr/bin/env python3

"""
Safety Validator Node for Vision-Language-Action (VLA) Module
Implements safety validation and constraint checking
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
from typing import Dict, List, Any, Tuple
import json
from enum import Enum

# Import custom message types
from vla_integration.msg import ActionPlan, Task, ExecutionStatus
from vla_integration.srv import ExecutePlan, SafetyValidation


class SafetyConstraintType(Enum):
    KINEMATIC = "kinematic"
    ENVIRONMENTAL = "environmental"
    ROBOT_CAPABILITY = "robot_capability"
    HUMAN_SAFETY = "human_safety"
    COLLISION = "collision"
    FORCE_LIMIT = "force_limit"


class SafetyViolation:
    def __init__(self, constraint_type: SafetyConstraintType, description: str, severity: str = "medium"):
        self.type = constraint_type.value
        self.description = description
        self.severity = severity


class SafetyValidatorNode(Node):
    """
    ROS 2 Node for safety validation and constraint checking
    """

    def __init__(self):
        super().__init__('safety_validator')

        # Declare parameters
        self.declare_parameter('safety_check_timeout', 5.0)
        self.declare_parameter('collision_threshold', 0.1)  # meters
        self.declare_parameter('force_limit_factor', 0.8)  # percentage of max force
        self.declare_parameter('safety_validation_enabled', True)

        # Get parameters
        self.safety_check_timeout = self.get_parameter('safety_check_timeout').value
        self.collision_threshold = self.get_parameter('collision_threshold').value
        self.force_limit_factor = self.get_parameter('force_limit_factor').value
        self.safety_validation_enabled = self.get_parameter('safety_validation_enabled').value

        # Create service for safety validation
        self.safety_service = self.create_service(
            SafetyValidation,
            'vla/safety/validate_action',
            self.validate_action_callback
        )

        # Create publisher for safety status
        self.safety_status_pub = self.create_publisher(ExecutionStatus, 'vla/safety_status', 10)

        # Initialize safety state
        self.safety_constraints = self.initialize_safety_constraints()
        self.violation_history: Dict[str, List[SafetyViolation]] = {}

        self.get_logger().info('Safety Validator Node initialized')

    def initialize_safety_constraints(self) -> Dict[str, Any]:
        """
        Initialize default safety constraints
        """
        constraints = {
            'kinematic': {
                'joint_limits': True,
                'workspace_bounds': True,
                'velocity_limits': True,
                'acceleration_limits': True
            },
            'environmental': {
                'obstacle_avoidance': True,
                'safe_zones': True,
                'hazard_detection': True
            },
            'robot_capability': {
                'payload_limits': True,
                'reachability': True,
                'stability': True
            },
            'human_safety': {
                'proximity_limits': True,
                'speed_limits_near_humans': True
            }
        }
        return constraints

    def validate_action_callback(self, request, response):
        """
        Service callback for validating actions for safety
        Expected request format based on API contract:
        {
          "action_plan": ActionPlan,
          "execution_context": {
            "robot_id": "robot_001",
            "environment_id": "env_001"
          }
        }

        Expected response format:
        {
          "is_safe": true,
          "violations": [
            {
              "type": "collision",
              "description": "Potential collision with obstacle",
              "severity": "high"
            }
          ],
          "safety_score": 0.95,
          "timestamp": "2025-12-25T10:00:05Z"
        }
        """
        try:
            self.get_logger().info(f'Validating action plan: {request.action_plan.plan_id}')

            if not self.safety_validation_enabled:
                response.is_safe = True
                response.violations = []
                response.safety_score = 1.0
                response.message = 'Safety validation disabled'
                response.timestamp = self.get_clock().now().to_msg()
                return response

            # Validate the action plan
            is_safe, violations, safety_score = self.validate_action_plan(request.action_plan)

            response.is_safe = is_safe
            response.violations = []
            response.safety_score = safety_score
            response.timestamp = self.get_clock().now().to_msg()

            # Convert violations to message format
            for violation in violations:
                # We need to create a violation message based on our custom type
                # For now, we'll add to violation history and return basic info
                response.message = f'Found {len(violations)} safety violations'

            # Store violations in history
            exec_id = f'validation_{int(time.time())}'
            self.violation_history[exec_id] = violations

            if is_safe:
                response.message = 'Action plan is safe for execution'
                self.get_logger().info(f'Action plan {request.action_plan.plan_id} is safe')
            else:
                response.message = f'Action plan has {len(violations)} safety violations'
                self.get_logger().warn(f'Action plan {request.action_plan.plan_id} has safety violations')

        except Exception as e:
            self.get_logger().error(f'Error in safety validation: {str(e)}')
            response.is_safe = False
            response.violations = []
            response.safety_score = 0.0
            response.message = f'Safety validation error: {str(e)}'
            response.timestamp = self.get_clock().now().to_msg()

        return response

    def validate_action_plan(self, plan: ActionPlan) -> Tuple[bool, List[SafetyViolation], float]:
        """
        Validate an entire action plan for safety
        """
        try:
            all_violations = []
            for task in plan.tasks:
                task_violations = self.validate_task(task)
                all_violations.extend(task_violations)

            is_safe = len(all_violations) == 0
            safety_score = self.calculate_safety_score(all_violations, len(plan.tasks))

            return is_safe, all_violations, safety_score

        except Exception as e:
            self.get_logger().error(f'Error validating action plan: {str(e)}')
            return False, [SafetyViolation(SafetyConstraintType.ROBOT_CAPABILITY, f'Validation error: {str(e)}')], 0.0

    def validate_task(self, task: Task) -> List[SafetyViolation]:
        """
        Validate a single task for safety
        """
        try:
            violations = []

            # Validate based on task type
            if task.type == 'navigate':
                violations.extend(self.validate_navigation_task(task))
            elif task.type == 'grasp_object':
                violations.extend(self.validate_manipulation_task(task))
            elif task.type == 'manipulate_object':
                violations.extend(self.validate_manipulation_task(task))
            elif task.type == 'detect_object':
                violations.extend(self.validate_perception_task(task))
            else:
                violations.extend(self.validate_generic_task(task))

            return violations

        except Exception as e:
            self.get_logger().error(f'Error validating task {task.task_id}: {str(e)}')
            return [SafetyViolation(SafetyConstraintType.ROBOT_CAPABILITY, f'Task validation error: {str(e)}')]

    def validate_navigation_task(self, task: Task) -> List[SafetyViolation]:
        """
        Validate navigation task for safety
        """
        try:
            violations = []

            # Check for collision risks
            target_location = self.extract_task_parameter(task, 'target_location', 'unknown')
            if target_location != 'unknown':
                # In a real implementation, this would check against map data
                # For simulation, we'll assume it's safe unless explicitly marked
                pass

            # Check workspace bounds
            # In a real implementation, this would validate coordinates against robot workspace
            # For simulation, we'll assume valid bounds

            return violations

        except Exception as e:
            self.get_logger().error(f'Error validating navigation task: {str(e)}')
            return [SafetyViolation(SafetyConstraintType.ENVIRONMENTAL, f'Navigation validation error: {str(e)}')]

    def validate_manipulation_task(self, task: Task) -> List[SafetyViolation]:
        """
        Validate manipulation task for safety
        """
        try:
            violations = []

            # Check payload limits
            object_weight = float(self.extract_task_parameter(task, 'object_weight', '0.0'))
            max_payload = 5.0  # kg - example value
            if object_weight > max_payload:
                violations.append(SafetyViolation(
                    SafetyConstraintType.ROBOT_CAPABILITY,
                    f'Object weight ({object_weight}kg) exceeds payload limit ({max_payload}kg)',
                    'high'
                ))

            # Check reachability
            target_position = self.extract_task_parameter(task, 'target_position', '0,0,0')
            # In a real implementation, this would check inverse kinematics
            # For simulation, we'll assume reachable

            return violations

        except Exception as e:
            self.get_logger().error(f'Error validating manipulation task: {str(e)}')
            return [SafetyViolation(SafetyConstraintType.ROBOT_CAPABILITY, f'Manipulation validation error: {str(e)}')]

    def validate_perception_task(self, task: Task) -> List[SafetyViolation]:
        """
        Validate perception task for safety
        """
        try:
            violations = []

            # Perception tasks are generally safe, but check for any constraints
            # In a real implementation, this might check for safe camera positioning

            return violations

        except Exception as e:
            self.get_logger().error(f'Error validating perception task: {str(e)}')
            return [SafetyViolation(SafetyConstraintType.ROBOT_CAPABILITY, f'Perception validation error: {str(e)}')]

    def validate_generic_task(self, task: Task) -> List[SafetyViolation]:
        """
        Validate generic task for safety
        """
        try:
            violations = []

            # Check common parameters
            task_type = task.type
            if task_type not in ['navigate', 'grasp_object', 'manipulate_object', 'detect_object', 'place_object', 'approach_object', 'wait']:
                violations.append(SafetyViolation(
                    SafetyConstraintType.ROBOT_CAPABILITY,
                    f'Unknown task type: {task_type}',
                    'medium'
                ))

            return violations

        except Exception as e:
            self.get_logger().error(f'Error validating generic task: {str(e)}')
            return [SafetyViolation(SafetyConstraintType.ROBOT_CAPABILITY, f'Generic validation error: {str(e)}')]

    def extract_task_parameter(self, task: Task, param_name: str, default_value: str = '') -> str:
        """
        Extract a parameter from a task
        """
        try:
            for param in task.parameters:
                if param.key == param_name:
                    return param.value
            return default_value
        except Exception:
            return default_value

    def calculate_safety_score(self, violations: List[SafetyViolation], total_tasks: int) -> float:
        """
        Calculate a safety score based on violations
        """
        try:
            if not violations:
                return 1.0  # Perfect safety

            # Calculate score based on violation severity
            severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.7, 'critical': 1.0}
            total_penalty = 0.0

            for violation in violations:
                severity = getattr(violation, 'severity', 'medium')
                penalty = severity_weights.get(severity, 0.3)
                total_penalty += penalty

            # Normalize by number of tasks
            if total_tasks > 0:
                avg_penalty = total_penalty / total_tasks
            else:
                avg_penalty = total_penalty

            # Ensure score is between 0 and 1
            safety_score = max(0.0, min(1.0, 1.0 - avg_penalty))
            return safety_score

        except Exception as e:
            self.get_logger().error(f'Error calculating safety score: {str(e)}')
            return 0.0

    def check_environmental_safety(self, plan: ActionPlan) -> List[SafetyViolation]:
        """
        Check environmental safety constraints
        """
        try:
            violations = []

            # In a real implementation, this would check against environmental data
            # For simulation, we'll return empty list

            return violations

        except Exception as e:
            self.get_logger().error(f'Error checking environmental safety: {str(e)}')
            return [SafetyViolation(SafetyConstraintType.ENVIRONMENTAL, f'Environmental check error: {str(e)}')]

    def check_kinematic_constraints(self, plan: ActionPlan) -> List[SafetyViolation]:
        """
        Check kinematic constraints
        """
        try:
            violations = []

            # In a real implementation, this would check joint limits, workspace, etc.
            # For simulation, we'll return empty list

            return violations

        except Exception as e:
            self.get_logger().error(f'Error checking kinematic constraints: {str(e)}')
            return [SafetyViolation(SafetyConstraintType.KINEMATIC, f'Kinematic check error: {str(e)}')]

    def get_violation_history(self, execution_id: str = None) -> List[SafetyViolation]:
        """
        Get safety violation history
        """
        try:
            if execution_id:
                return self.violation_history.get(execution_id, [])
            else:
                # Return all violations
                all_violations = []
                for violations in self.violation_history.values():
                    all_violations.extend(violations)
                return all_violations
        except Exception as e:
            self.get_logger().error(f'Error getting violation history: {str(e)}')
            return []

    def get_safety_statistics(self) -> Dict[str, Any]:
        """
        Get safety validation statistics
        """
        try:
            total_violations = sum(len(violations) for violations in self.violation_history.values())
            total_executions = len(self.violation_history)

            stats = {
                'total_validations': total_executions,
                'total_violations': total_violations,
                'violation_rate': total_violations / max(1, total_executions),
                'constraint_types': [c.value for c in SafetyConstraintType]
            }

            return stats

        except Exception as e:
            self.get_logger().error(f'Error getting safety statistics: {str(e)}')
            return {}

    def publish_safety_status(self, is_safe: bool, violations: List[SafetyViolation]):
        """
        Publish safety status updates
        """
        try:
            status_msg = ExecutionStatus()
            status_msg.header.stamp = self.get_clock().now().to_msg()
            status_msg.header.frame_id = 'safety_validator'
            status_msg.overall_status = 'safe' if is_safe else 'unsafe'
            status_msg.completed_tasks = len(violations)
            status_msg.total_tasks = 1  # For safety validation
            status_msg.progress = 1.0 if is_safe else 0.0
            status_msg.error = f'{len(violations)} violations' if violations else 'No violations'

            self.safety_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing safety status: {str(e)}')


def main(args=None):
    rclpy.init(args=args)

    safety_validator = SafetyValidatorNode()

    try:
        rclpy.spin(safety_validator)
    except KeyboardInterrupt:
        pass
    finally:
        safety_validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()