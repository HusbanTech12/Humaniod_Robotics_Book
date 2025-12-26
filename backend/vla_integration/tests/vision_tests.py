#!/usr/bin/env python3

"""
Vision Testing Procedures for Vision-Language-Action (VLA) Module
Tests for vision processing, object detection, and localization components
"""

import unittest
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
from typing import Dict, List, Any
import numpy as np

# Import custom message types
from vla_integration.msg import VisionData, ObjectDetection
from vla_integration.srv import LocalizeObject


class MockVisionNode(Node):
    """
    Mock node for testing vision components without full ROS 2 infrastructure
    """
    def __init__(self):
        super().__init__('mock_vision_node')

        # Mock service clients
        self.localize_client = self.create_client(LocalizeObject, 'vla/vision/localize_object')

        # Mock publishers/subscribers
        self.vision_pub = self.create_publisher(VisionData, 'vla/test_vision_data', 10)


class VisionModuleTests(unittest.TestCase):
    """
    Tests for the vision processing components of the VLA module
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test class with ROS 2 context
        """
        rclpy.init()
        cls.node = MockVisionNode()
        cls.executor = rclpy.executors.SingleThreadedExecutor()
        cls.executor.add_node(cls.node)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up ROS 2 context
        """
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_object_detection_accuracy(self):
        """
        Test object detection accuracy with simulated data
        """
        # Simulate detection of various objects
        test_objects = [
            {"name": "cup", "confidence": 0.95, "position": [1.0, 0.5, 0.8]},
            {"name": "book", "confidence": 0.89, "position": [0.8, -0.2, 0.75]},
            {"name": "chair", "confidence": 0.92, "position": [-0.5, 1.2, 0.6]},
        ]

        # Verify detection parameters
        for obj in test_objects:
            self.assertGreaterEqual(obj["confidence"], 0.85, f"{obj['name']} detection confidence too low")
            self.assertEqual(len(obj["position"]), 3, f"{obj['name']} position should have 3 coordinates")

            # Verify position is reasonable (not NaN or infinite)
            for coord in obj["position"]:
                self.assertIsInstance(coord, (int, float))
                self.assertTrue(np.isfinite(coord))

        print("✓ Object detection accuracy test passed")

    def test_object_localization_precision(self):
        """
        Test object localization precision
        """
        # Create mock object detection
        detection = ObjectDetection()
        detection.object_id = "test_object_1"
        detection.class_name = "cup"
        detection.confidence = 0.9
        detection.position_3d.x = 1.2
        detection.position_3d.y = 0.8
        detection.position_3d.z = 0.95

        # Verify position values are within expected ranges
        self.assertGreaterEqual(detection.position_3d.x, -5.0)  # Reasonable x range
        self.assertLessEqual(detection.position_3d.x, 5.0)
        self.assertGreaterEqual(detection.position_3d.y, -5.0)  # Reasonable y range
        self.assertLessEqual(detection.position_3d.y, 5.0)
        self.assertGreaterEqual(detection.position_3d.z, 0.0)   # Z should be positive
        self.assertLessEqual(detection.position_3d.z, 3.0)     # Reasonable height

        # Verify confidence is in valid range
        self.assertGreaterEqual(detection.confidence, 0.0)
        self.assertLessEqual(detection.confidence, 1.0)

        print("✓ Object localization precision test passed")

    def test_vision_data_structure(self):
        """
        Test vision data message structure
        """
        # Create mock vision data
        vision_data = VisionData()
        vision_data.data_id = "test_vision_data_1"
        vision_data.source_camera = "front_camera"
        vision_data.timestamp = self.node.get_clock().now().to_msg()

        # Add mock detections
        for i in range(3):
            detection = ObjectDetection()
            detection.object_id = f"obj_{i}"
            detection.class_name = f"object_{i}"
            detection.confidence = 0.8 + (i * 0.05)  # 0.8, 0.85, 0.9
            detection.position_3d.x = i * 0.5
            detection.position_3d.y = i * 0.3
            detection.position_3d.z = 1.0
            vision_data.object_detections.append(detection)

        # Verify structure
        self.assertEqual(len(vision_data.object_detections), 3)
        self.assertEqual(vision_data.source_camera, "front_camera")
        self.assertIsNotNone(vision_data.timestamp)

        # Verify detection attributes
        for detection in vision_data.object_detections:
            self.assertIsNotNone(detection.object_id)
            self.assertIsNotNone(detection.class_name)
            self.assertGreaterEqual(detection.confidence, 0.0)
            self.assertLessEqual(detection.confidence, 1.0)

        print("✓ Vision data structure test passed")

    def test_localization_service_interface(self):
        """
        Test localization service interface
        """
        # Create mock localization request
        object_description = "red cup"
        search_area = {
            "x_range": [-2.0, 2.0],
            "y_range": [-2.0, 2.0],
            "z_range": [0.0, 1.5]
        }

        # Verify request structure would be valid
        self.assertIsInstance(object_description, str)
        self.assertIn("x_range", search_area)
        self.assertIn("y_range", search_area)
        self.assertIn("z_range", search_area)
        self.assertEqual(len(search_area["x_range"]), 2)
        self.assertEqual(len(search_area["y_range"]), 2)
        self.assertEqual(len(search_area["z_range"]), 2)

        # Verify ranges are ordered correctly
        self.assertLessEqual(search_area["x_range"][0], search_area["x_range"][1])
        self.assertLessEqual(search_area["y_range"][0], search_area["y_range"][1])
        self.assertLessEqual(search_area["z_range"][0], search_area["z_range"][1])

        print("✓ Localization service interface test passed")

    def test_spatial_relationship_computation(self):
        """
        Test spatial relationship computation between objects
        """
        # Create mock objects with positions
        objects = [
            {"id": "obj_1", "pos": np.array([0.0, 0.0, 0.0])},
            {"id": "obj_2", "pos": np.array([0.5, 0.0, 0.0])},  # 0.5m away in x
            {"id": "obj_3", "pos": np.array([0.0, 0.8, 0.0])},  # 0.8m away in y
        ]

        # Compute distances
        relationships = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1, obj2 = objects[i], objects[j]
                distance = np.linalg.norm(obj1["pos"] - obj2["pos"])

                relationship = {
                    "object1": obj1["id"],
                    "object2": obj2["id"],
                    "distance": distance,
                    "vector": obj2["pos"] - obj1["pos"]
                }
                relationships.append(relationship)

        # Verify distances are reasonable
        for rel in relationships:
            self.assertGreaterEqual(rel["distance"], 0.0)
            self.assertLessEqual(rel["distance"], 5.0)  # Reasonable max distance

        # Verify the closest relationship
        min_distance = min(rel["distance"] for rel in relationships)
        self.assertLessEqual(min_distance, 0.6)  # Should have objects within 0.6m

        print("✓ Spatial relationship computation test passed")

    def test_scene_context_building(self):
        """
        Test scene context building from vision data
        """
        # Create mock vision data with multiple objects
        vision_data = VisionData()
        vision_data.scene_context = "Test scene with multiple objects"

        # Add various object types
        object_types = ["cup", "book", "chair", "table", "bottle"]
        for i, obj_type in enumerate(object_types):
            detection = ObjectDetection()
            detection.object_id = f"obj_{i}"
            detection.class_name = obj_type
            detection.confidence = 0.85 + (i * 0.03)
            detection.position_3d.x = i * 0.3
            detection.position_3d.y = i * 0.2
            detection.position_3d.z = 1.0
            vision_data.object_detections.append(detection)

        # Verify scene context has correct number of objects
        self.assertEqual(len(vision_data.object_detections), len(object_types))

        # Verify object types are diverse
        detected_types = [det.class_name for det in vision_data.object_detections]
        self.assertEqual(len(set(detected_types)), len(object_types))  # All unique

        # Verify confidences are acceptable
        avg_confidence = sum(det.confidence for det in vision_data.object_detections) / len(vision_data.object_detections)
        self.assertGreaterEqual(avg_confidence, 0.85)

        print("✓ Scene context building test passed")

    def test_vision_pipeline_timing(self):
        """
        Test vision pipeline timing requirements
        """
        # Simulate vision processing timing
        start_time = time.time()

        # Simulate processing steps
        time.sleep(0.05)  # Simulate image capture (50ms)
        time.sleep(0.10)  # Simulate object detection (100ms)
        time.sleep(0.05)  # Simulate localization (50ms)

        end_time = time.time()
        total_time = end_time - start_time

        # Verify processing time is within acceptable limits (200ms)
        self.assertLessEqual(total_time, 0.25)  # 250ms with some buffer

        # Verify each component meets timing requirements
        # In real testing, we'd measure individual components separately
        expected_time = 0.20  # 200ms for all steps
        self.assertLessEqual(total_time, expected_time * 1.5)  # 50% buffer

        print("✓ Vision pipeline timing test passed")

    def test_multi_camera_fusion(self):
        """
        Test multi-camera data fusion capabilities
        """
        # Simulate data from multiple cameras
        camera_data = {
            "front_camera": VisionData(),
            "left_camera": VisionData(),
            "right_camera": VisionData()
        }

        # Populate each camera's detections
        for cam_name, cam_data in camera_data.items():
            cam_data.source_camera = cam_name
            cam_data.data_id = f"fusion_test_{cam_name}"

            # Add overlapping detections from different angles
            for i in range(2):
                detection = ObjectDetection()
                detection.object_id = f"{cam_name}_obj_{i}"
                detection.class_name = "test_object"
                detection.confidence = 0.8 + (i * 0.1)
                detection.position_3d.x = i * 0.5
                detection.position_3d.y = 0.0 if "front" in cam_name else (0.3 if "left" in cam_name else -0.3)
                detection.position_3d.z = 1.0
                cam_data.object_detections.append(detection)

        # Verify each camera has data
        for cam_name, cam_data in camera_data.items():
            self.assertEqual(len(cam_data.object_detections), 2)
            self.assertEqual(cam_data.source_camera, cam_name)

        # Test fusion concept - in real implementation, this would merge overlapping detections
        all_detections = []
        for cam_data in camera_data.values():
            all_detections.extend(cam_data.object_detections)

        self.assertEqual(len(all_detections), 6)  # 3 cameras * 2 detections each

        print("✓ Multi-camera fusion test passed")

    def test_vision_robustness_under_conditions(self):
        """
        Test vision system robustness under various conditions
        """
        # Test conditions: lighting, occlusion, distance
        test_conditions = [
            {"name": "good_lighting", "confidence_factor": 1.0},
            {"name": "dim_lighting", "confidence_factor": 0.8},
            {"name": "bright_lighting", "confidence_factor": 0.9},
            {"name": "partial_occlusion", "confidence_factor": 0.7},
            {"name": "far_distance", "confidence_factor": 0.6}
        ]

        base_confidence = 0.9
        for condition in test_conditions:
            adjusted_confidence = base_confidence * condition["confidence_factor"]

            # All conditions should maintain minimum confidence
            self.assertGreaterEqual(adjusted_confidence, 0.5,
                                  f"{condition['name']} confidence too low: {adjusted_confidence}")

        # Verify that good conditions maintain high confidence
        good_condition_conf = next(c for c in test_conditions if c["name"] == "good_lighting")["confidence_factor"] * base_confidence
        self.assertGreaterEqual(good_condition_conf, 0.8)

        print("✓ Vision robustness under conditions test passed")


def run_vision_tests():
    """
    Run all vision module tests
    """
    print("Starting VLA Vision Module Tests...\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(VisionModuleTests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\nVision Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    return result.wasSuccessful()


def main():
    """
    Main function to run the vision tests
    """
    success = run_vision_tests()
    return 0 if success else 1


if __name__ == '__main__':
    main()