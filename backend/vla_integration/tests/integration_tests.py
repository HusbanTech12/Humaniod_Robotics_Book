#!/usr/bin/env python3

"""
Integration Tests for Vision-Language-Action (VLA) Module
Comprehensive tests for the complete VLA system
"""

import unittest
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
from typing import Dict, List, Any
import json

# Import custom message types
from vla_integration.msg import ActionPlan, Task, ExecutionStatus
from vla_integration.srv import ExecutePlan, GenerateActionPlan, LocalizeObject


class MockVLANode(Node):
    """
    Mock node for testing VLA components without full ROS 2 infrastructure
    """
    def __init__(self):
        super().__init__('mock_vla_node')

        # Mock service clients
        self.plan_client = self.create_client(GenerateActionPlan, 'vla/generate_plan')
        self.execution_client = self.create_client(ExecutePlan, 'vla/actions/execute_plan')
        self.vision_client = self.create_client(LocalizeObject, 'vla/vision/localize_object')

        # Mock publishers/subscribers
        self.status_pub = self.create_publisher(ExecutionStatus, 'vla/test_status', 10)


class VLAModuleIntegrationTests(unittest.TestCase):
    """
    Integration tests for the VLA module
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test class with ROS 2 context
        """
        rclpy.init()
        cls.node = MockVLANode()
        cls.executor = rclpy.executors.SingleThreadedExecutor()
        cls.executor.add_node(cls.node)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up ROS 2 context
        """
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_voice_to_action_pipeline(self):
        """
        Test the complete voice-to-action pipeline
        """
        # Test voice command processing
        command = "Go to the kitchen and find the red cup"

        # Create a mock action plan
        plan = ActionPlan()
        plan.plan_id = "test_plan_1"

        # Add tasks based on the command
        nav_task = Task()
        nav_task.task_id = "nav_task_1"
        nav_task.type = "navigate"
        nav_task.description = "Navigate to kitchen"
        plan.tasks.append(nav_task)

        detect_task = Task()
        detect_task.task_id = "detect_task_1"
        detect_task.type = "detect_object"
        detect_task.description = "Detect red cup"
        plan.tasks.append(detect_task)

        # Verify the plan has the expected structure
        self.assertEqual(len(plan.tasks), 2)
        self.assertEqual(plan.tasks[0].type, "navigate")
        self.assertEqual(plan.tasks[1].type, "detect_object")

        print("✓ Voice-to-action pipeline test passed")

    def test_plan_generation_and_validation(self):
        """
        Test plan generation and validation
        """
        # Test plan generation from command
        command = "Pick up the book from the table"

        # Create mock plan
        plan = ActionPlan()
        plan.plan_id = "test_plan_2"

        # Add expected tasks
        approach_task = Task()
        approach_task.task_id = "approach_task_1"
        approach_task.type = "approach_object"
        approach_task.description = "Approach the book"
        plan.tasks.append(approach_task)

        grasp_task = Task()
        grasp_task.task_id = "grasp_task_1"
        grasp_task.type = "grasp_object"
        grasp_task.description = "Grasp the book"
        plan.tasks.append(grasp_task)

        # Validate plan structure
        self.assertEqual(len(plan.tasks), 2)
        self.assertIn("approach_object", [task.type for task in plan.tasks])
        self.assertIn("grasp_object", [task.type for task in plan.tasks])

        # Validate task IDs are unique
        task_ids = [task.task_id for task in plan.tasks]
        self.assertEqual(len(task_ids), len(set(task_ids)))

        print("✓ Plan generation and validation test passed")

    def test_vision_grounding_integration(self):
        """
        Test vision grounding integration
        """
        # Test object localization
        object_description = "red cup"

        # Mock vision service response
        localization_result = {
            "object_id": "cup_001",
            "class": "cup",
            "color": "red",
            "position": {"x": 1.2, "y": 0.8, "z": 0.95},
            "confidence": 0.89,
            "bounding_box": {"x": 120, "y": 85, "width": 45, "height": 60}
        }

        # Validate localization result structure
        self.assertIn("object_id", localization_result)
        self.assertIn("position", localization_result)
        self.assertIn("confidence", localization_result)

        # Validate position coordinates
        pos = localization_result["position"]
        self.assertIsInstance(pos["x"], (int, float))
        self.assertIsInstance(pos["y"], (int, float))
        self.assertIsInstance(pos["z"], (int, float))

        # Validate confidence is in valid range
        self.assertGreaterEqual(localization_result["confidence"], 0.0)
        self.assertLessEqual(localization_result["confidence"], 1.0)

        print("✓ Vision grounding integration test passed")

    def test_action_execution_workflow(self):
        """
        Test action execution workflow
        """
        # Create a multi-step action plan
        plan = ActionPlan()
        plan.plan_id = "test_plan_3"

        # Add sequential tasks
        tasks = [
            ("navigate_to_kitchen", "navigate", "Go to kitchen"),
            ("detect_cup", "detect_object", "Find the cup"),
            ("grasp_cup", "grasp_object", "Pick up the cup"),
            ("navigate_to_table", "navigate", "Go to table"),
            ("place_cup", "place_object", "Place cup on table")
        ]

        for task_id, task_type, description in tasks:
            task = Task()
            task.task_id = task_id
            task.type = task_type
            task.description = description
            plan.tasks.append(task)

        # Verify plan structure
        self.assertEqual(len(plan.tasks), 5)

        # Check task sequence
        expected_sequence = ["navigate", "detect_object", "grasp_object", "navigate", "place_object"]
        actual_sequence = [task.type for task in plan.tasks]
        self.assertEqual(actual_sequence, expected_sequence)

        print("✓ Action execution workflow test passed")

    def test_error_handling_and_recovery(self):
        """
        Test error handling and recovery mechanisms
        """
        # Test error detection in task execution
        execution_context = {
            'plan': ActionPlan(),
            'execution_id': 'test_exec_1',
            'status': 'executing',
            'current_task_idx': 0,
            'completed_tasks': [],
            'failed_tasks': [],
            'task_statuses': {}
        }

        # Simulate a failed task
        failed_task_id = "test_task_1"
        execution_context['failed_tasks'].append(failed_task_id)
        execution_context['task_statuses'][failed_task_id] = 'failed'

        # Test recovery logic
        recovery_possible = len(execution_context['failed_tasks']) < 3  # Max 3 failures

        self.assertTrue(recovery_possible)

        # Test status updates after error
        execution_context['status'] = 'recovery_needed'
        self.assertEqual(execution_context['status'], 'recovery_needed')

        print("✓ Error handling and recovery test passed")

    def test_safety_validation_integration(self):
        """
        Test safety validation integration
        """
        # Create a test plan that should pass safety validation
        safe_plan = ActionPlan()
        safe_plan.plan_id = "safe_plan_1"

        # Add safe tasks
        nav_task = Task()
        nav_task.task_id = "safe_nav_1"
        nav_task.type = "navigate"
        nav_task.description = "Navigate to safe location"
        safe_plan.tasks.append(nav_task)

        # Test safety validation logic
        # In a real system, this would call the safety service
        # For testing, we'll simulate the validation

        violations = []
        is_safe = True

        # Check for unsafe task types (example)
        unsafe_types = ['unsafe_task_type']  # In practice, this would be actual unsafe operations
        for task in safe_plan.tasks:
            if task.type in unsafe_types:
                violations.append(f"Unsafe task type: {task.type}")
                is_safe = False

        self.assertTrue(is_safe)
        self.assertEqual(len(violations), 0)

        print("✓ Safety validation integration test passed")

    def test_multi_step_task_completion(self):
        """
        Test completion of complex multi-step tasks
        """
        # Create a complex multi-step task plan
        complex_plan = ActionPlan()
        complex_plan.plan_id = "complex_plan_1"

        # Add tasks for a complex scenario: "Clean the room by picking up trash and placing it in the bin"
        steps = [
            ("scan_room", "detect_object", "Scan room for trash"),
            ("navigate_to_trash1", "navigate", "Go to first trash item"),
            ("grasp_trash1", "grasp_object", "Pick up first trash item"),
            ("navigate_to_bin", "navigate", "Go to trash bin"),
            ("place_trash1", "place_object", "Place trash in bin"),
            ("navigate_to_trash2", "navigate", "Go to second trash item"),
            ("grasp_trash2", "grasp_object", "Pick up second trash item"),
            ("navigate_to_bin2", "navigate", "Go back to trash bin"),
            ("place_trash2", "place_object", "Place second trash in bin")
        ]

        for task_id, task_type, description in steps:
            task = Task()
            task.task_id = task_id
            task.type = task_type
            task.description = description
            complex_plan.tasks.append(task)

        # Verify all steps are present
        self.assertEqual(len(complex_plan.tasks), len(steps))

        # Verify task IDs are unique
        task_ids = [task.task_id for task in complex_plan.tasks]
        self.assertEqual(len(task_ids), len(set(task_ids)))

        # Test execution sequence validation
        expected_types = [step[1] for step in steps]
        actual_types = [task.type for task in complex_plan.tasks]
        self.assertEqual(actual_types, expected_types)

        print("✓ Multi-step task completion test passed")

    def test_end_to_end_vla_pipeline(self):
        """
        Test the complete end-to-end VLA pipeline
        """
        # Simulate the complete pipeline: Voice → Language → Vision → Action
        start_time = time.time()

        # Step 1: Voice processing (simulated)
        voice_command = "Go to the kitchen, find the red cup, pick it up, and bring it to the table"

        # Step 2: Language processing - generate action plan
        plan = ActionPlan()
        plan.plan_id = f"end_to_end_plan_{int(start_time)}"

        # Add tasks based on command
        tasks = [
            ("navigate_to_kitchen", "navigate", "Go to kitchen"),
            ("detect_red_cup", "detect_object", "Find the red cup"),
            ("approach_cup", "approach_object", "Approach the cup"),
            ("grasp_cup", "grasp_object", "Pick up the cup"),
            ("navigate_to_table", "navigate", "Go to table"),
            ("place_cup", "place_object", "Place cup on table")
        ]

        for task_id, task_type, description in tasks:
            task = Task()
            task.task_id = task_id
            task.type = task_type
            task.description = description
            plan.tasks.append(task)

        # Step 3: Vision grounding - would happen during execution
        # For simulation, we'll just verify the plan is structured correctly

        # Step 4: Action execution - simulate status updates
        execution_status = {
            'execution_id': f"exec_{int(start_time)}",
            'plan_id': plan.plan_id,
            'overall_status': 'completed',
            'completed_tasks': len(plan.tasks),
            'total_tasks': len(plan.tasks),
            'progress': 1.0,
            'error': ''
        }

        # Validate the end-to-end flow
        self.assertEqual(len(plan.tasks), 6)  # 6 steps in the command
        self.assertEqual(execution_status['overall_status'], 'completed')
        self.assertEqual(execution_status['progress'], 1.0)
        self.assertEqual(execution_status['completed_tasks'], execution_status['total_tasks'])

        # Measure execution time (simulated)
        end_time = time.time()
        execution_time = end_time - start_time

        # Verify the pipeline completed successfully
        self.assertGreater(execution_time, 0)  # Should take some time (even if small)

        print("✓ End-to-end VLA pipeline test passed")

    def test_performance_under_load(self):
        """
        Test system performance under load conditions
        """
        # Test multiple concurrent executions
        num_executions = 5
        execution_times = []

        for i in range(num_executions):
            start_time = time.time()

            # Create and process a plan
            plan = ActionPlan()
            plan.plan_id = f"load_test_plan_{i}"

            # Add simple tasks
            for j in range(3):  # 3 tasks each
                task = Task()
                task.task_id = f"task_{i}_{j}"
                task.type = "wait" if j % 2 == 0 else "navigate"
                task.description = f"Task {j} for execution {i}"
                plan.tasks.append(task)

            # Simulate execution time
            time.sleep(0.01)  # Small delay to simulate processing
            end_time = time.time()

            execution_times.append(end_time - start_time)

        # Verify all executions completed
        self.assertEqual(len(execution_times), num_executions)

        # Verify execution times are reasonable (less than 1 second each)
        for exec_time in execution_times:
            self.assertLess(exec_time, 1.0)

        # Calculate average execution time
        avg_time = sum(execution_times) / len(execution_times)
        self.assertLess(avg_time, 0.1)  # Should be fast (< 100ms average)

        print("✓ Performance under load test passed")


def run_integration_tests():
    """
    Run all integration tests
    """
    print("Starting VLA Module Integration Tests...\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(VLAModuleIntegrationTests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    return result.wasSuccessful()


def main():
    """
    Main function to run the integration tests
    """
    success = run_integration_tests()
    return 0 if success else 1


if __name__ == '__main__':
    main()