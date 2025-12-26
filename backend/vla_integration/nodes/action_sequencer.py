#!/usr/bin/env python3

"""
Action Sequencer Node for Vision-Language-Action (VLA) Module
Manages task sequencing and synchronization during plan execution
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import time
from typing import Dict, List, Any, Optional
import json
from collections import defaultdict

# Import custom message types
from vla_integration.msg import ActionPlan, Task, ExecutionStatus
from vla_integration.srv import ExecutePlan


class ActionSequencerNode(Node):
    """
    ROS 2 Node for managing action sequencing and synchronization
    """

    def __init__(self):
        super().__init__('action_sequencer')

        # Declare parameters
        self.declare_parameter('max_parallel_tasks', 3)
        self.declare_parameter('dependency_check_interval', 0.5)  # seconds
        self.declare_parameter('task_timeout', 60.0)  # seconds
        self.declare_parameter('sequencer_rate', 10.0)  # Hz

        # Get parameters
        self.max_parallel_tasks = self.get_parameter('max_parallel_tasks').value
        self.dependency_check_interval = self.get_parameter('dependency_check_interval').value
        self.task_timeout = self.get_parameter('task_timeout').value
        self.sequencer_rate = self.get_parameter('sequencer_rate').value

        # Create publisher for sequencing updates
        self.sequencing_status_pub = self.create_publisher(ExecutionStatus, 'vla/sequencing_status', 10)

        # Create service for sequencing operations
        self.sequence_service = self.create_service(
            ExecutePlan,
            'vla/sequence_plan',
            self.sequence_plan_callback
        )

        # Initialize sequencer state
        self.active_sequences: Dict[str, Dict[str, Any]] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        self.task_ready_queue: Dict[str, List[str]] = defaultdict(list)

        # Create timer for sequencer operations
        self.sequencer_timer = self.create_timer(1.0/self.sequencer_rate, self.sequencer_callback)

        self.get_logger().info('Action Sequencer Node initialized')

    def sequence_plan_callback(self, request, response):
        """
        Service callback for sequencing action plans
        """
        try:
            execution_id = f'seq_{int(time.time())}_{hash(str(request.plan.plan_id)) % 10000}'
            self.get_logger().info(f'Sequencing plan: {request.plan.plan_id}, execution ID: {execution_id}')

            # Analyze task dependencies
            dependencies = self.analyze_dependencies(request.plan)

            # Create sequence context
            sequence_context = {
                'plan': request.plan,
                'execution_id': execution_id,
                'status': 'sequencing',
                'start_time': time.time(),
                'dependencies': dependencies,
                'ready_tasks': self.get_ready_tasks(request.plan, dependencies),
                'executing_tasks': [],
                'completed_tasks': [],
                'failed_tasks': [],
                'task_start_times': {},
                'task_results': {}
            }

            self.active_sequences[execution_id] = sequence_context

            # Prepare response
            response.execution_id = execution_id
            response.status = 'sequenced'
            response.message = f'Plan sequenced successfully with {len(sequence_context["ready_tasks"])} ready tasks'
            response.success = True

            self.get_logger().info(f'Plan sequenced successfully: {execution_id}')

        except Exception as e:
            self.get_logger().error(f'Error in plan sequencing: {str(e)}')
            response.execution_id = f'seq_error_{int(time.time())}'
            response.status = 'failed'
            response.message = f'Error sequencing plan: {str(e)}'
            response.success = False

        return response

    def analyze_dependencies(self, plan: ActionPlan) -> Dict[str, List[str]]:
        """
        Analyze task dependencies in the action plan
        """
        try:
            dependencies = {}
            for task in plan.tasks:
                # Extract dependencies from task definition
                task_deps = []
                for param in task.parameters:
                    if param.key == 'dependencies':
                        task_deps = [dep.strip() for dep in param.value.split(',')]
                        break
                dependencies[task.task_id] = task_deps

            return dependencies
        except Exception as e:
            self.get_logger().error(f'Error analyzing dependencies: {str(e)}')
            return {}

    def get_ready_tasks(self, plan: ActionPlan, dependencies: Dict[str, List[str]]) -> List[str]:
        """
        Get tasks that are ready to execute (no unmet dependencies)
        """
        try:
            ready_tasks = []
            completed_tasks = set()  # Initially empty, but we'll consider all tasks without dependencies

            for task in plan.tasks:
                task_deps = dependencies.get(task.task_id, [])
                # A task is ready if all its dependencies are satisfied
                # For initial sequencing, we consider tasks without dependencies as ready
                if not task_deps:
                    ready_tasks.append(task.task_id)

            return ready_tasks
        except Exception as e:
            self.get_logger().error(f'Error getting ready tasks: {str(e)}')
            return []

    def sequencer_callback(self):
        """
        Main sequencer callback that manages task execution order
        """
        try:
            for exec_id, seq_context in list(self.active_sequences.items()):
                if seq_context['status'] not in ['executing', 'sequencing']:
                    continue

                # Update sequence status
                seq_context['status'] = 'executing'

                # Check for completed and failed tasks
                self.check_task_completions(seq_context)

                # Determine which tasks are ready to execute
                ready_tasks = self.get_available_tasks(seq_context)

                # Start executing ready tasks up to the parallel limit
                self.start_ready_tasks(seq_context, ready_tasks)

                # Check if all tasks are completed
                if len(seq_context['completed_tasks']) == len(seq_context['plan'].tasks):
                    seq_context['status'] = 'completed'
                    self.get_logger().info(f'Sequence completed: {exec_id}')
                elif len(seq_context['failed_tasks']) > 0:
                    seq_context['status'] = 'failed'
                    self.get_logger().info(f'Sequence failed: {exec_id}')

        except Exception as e:
            self.get_logger().error(f'Error in sequencer callback: {str(e)}')

    def check_task_completions(self, seq_context: Dict[str, Any]):
        """
        Check for completed and failed tasks
        """
        try:
            # In a real implementation, this would check with execution services
            # For simulation, we'll just mark tasks as completed after a timeout
            current_time = time.time()

            for task_id in list(seq_context['executing_tasks']):
                if task_id in seq_context['task_start_times']:
                    start_time = seq_context['task_start_times'][task_id]
                    if current_time - start_time > self.task_timeout:
                        # Mark as failed due to timeout
                        seq_context['executing_tasks'].remove(task_id)
                        seq_context['failed_tasks'].append(task_id)
                        self.get_logger().warn(f'Task {task_id} timed out')

        except Exception as e:
            self.get_logger().error(f'Error checking task completions: {str(e)}')

    def get_available_tasks(self, seq_context: Dict[str, Any]) -> List[str]:
        """
        Get tasks that are available for execution (dependencies met, not already executing/completed)
        """
        try:
            available_tasks = []
            dependencies = seq_context['dependencies']
            completed_tasks = set(seq_context['completed_tasks'])
            executing_tasks = set(seq_context['executing_tasks'])

            for task in seq_context['plan'].tasks:
                task_id = task.task_id

                # Skip if already executing or completed
                if task_id in executing_tasks or task_id in completed_tasks:
                    continue

                # Check if all dependencies are met
                task_deps = dependencies.get(task_id, [])
                all_deps_met = all(dep in completed_tasks for dep in task_deps)

                if all_deps_met:
                    available_tasks.append(task_id)

            return available_tasks
        except Exception as e:
            self.get_logger().error(f'Error getting available tasks: {str(e)}')
            return []

    def start_ready_tasks(self, seq_context: Dict[str, Any], ready_tasks: List[str]):
        """
        Start executing ready tasks up to the parallel limit
        """
        try:
            # Calculate how many more tasks we can start
            max_new_tasks = self.max_parallel_tasks - len(seq_context['executing_tasks'])
            tasks_to_start = ready_tasks[:max_new_tasks]

            for task_id in tasks_to_start:
                if task_id not in seq_context['executing_tasks']:
                    seq_context['executing_tasks'].append(task_id)
                    seq_context['task_start_times'][task_id] = time.time()
                    self.get_logger().info(f'Starting task {task_id}')

        except Exception as e:
            self.get_logger().error(f'Error starting ready tasks: {str(e)}')

    def get_sequence_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific sequence
        """
        try:
            return self.active_sequences.get(execution_id, None)
        except Exception as e:
            self.get_logger().error(f'Error getting sequence status: {str(e)}')
            return None

    def cancel_sequence(self, execution_id: str) -> bool:
        """
        Cancel a specific sequence
        """
        try:
            if execution_id in self.active_sequences:
                self.active_sequences[execution_id]['status'] = 'cancelled'
                self.get_logger().info(f'Sequence cancelled: {execution_id}')
                return True
            return False
        except Exception as e:
            self.get_logger().error(f'Error cancelling sequence: {str(e)}')
            return False

    def get_active_sequences_count(self) -> int:
        """
        Get the count of active sequences
        """
        return len(self.active_sequences)

    def publish_sequencing_status(self):
        """
        Publish sequencing status updates
        """
        try:
            for exec_id, seq_context in self.active_sequences.items():
                status_msg = ExecutionStatus()
                status_msg.header.stamp = self.get_clock().now().to_msg()
                status_msg.header.frame_id = 'action_sequencer'
                status_msg.execution_id = exec_id
                status_msg.plan_id = seq_context['plan'].plan_id
                status_msg.overall_status = seq_context['status']
                status_msg.completed_tasks = len(seq_context['completed_tasks'])
                status_msg.total_tasks = len(seq_context['plan'].tasks)

                if status_msg.total_tasks > 0:
                    status_msg.progress = float(status_msg.completed_tasks) / status_msg.total_tasks
                else:
                    status_msg.progress = 0.0

                status_msg.error = ''

                self.sequencing_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing sequencing status: {str(e)}')


def main(args=None):
    rclpy.init(args=args)

    sequencer = ActionSequencerNode()

    try:
        rclpy.spin(sequencer)
    except KeyboardInterrupt:
        pass
    finally:
        sequencer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()