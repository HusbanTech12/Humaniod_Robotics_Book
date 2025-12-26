#!/usr/bin/env python3

"""
Vision Grounding Node for Vision-Language-Action (VLA) Module
Handles linking perception outputs to planning and execution
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
from typing import Dict, List, Any
import json

# Import custom message types
from vla_integration.msg import VisionData, ObjectDetection, ActionPlan, Task
from vla_integration.srv import LocalizeObject


class VisionGroundingNode(Node):
    """
    ROS 2 Node for grounding visual perceptions in the context of action planning
    """

    def __init__(self):
        super().__init__('vision_grounding')

        # Declare parameters
        self.declare_parameter('grounding_accuracy_threshold', 0.85)
        self.declare_parameter('max_grounding_attempts', 3)
        self.declare_parameter('spatial_tolerance', 0.1)  # meters
        self.declare_parameter('object_matching_threshold', 0.7)

        # Get parameters
        self.grounding_accuracy_threshold = self.get_parameter('grounding_accuracy_threshold').value
        self.max_grounding_attempts = self.get_parameter('max_grounding_attempts').value
        self.spatial_tolerance = self.get_parameter('spatial_tolerance').value
        self.object_matching_threshold = self.get_parameter('object_matching_threshold').value

        # Create subscriber for vision data
        self.vision_data_sub = self.create_subscription(
            VisionData,
            'vla/vision_data',
            self.vision_data_callback,
            10
        )

        # Create publisher for grounded vision data
        self.grounded_vision_pub = self.create_publisher(VisionData, 'vla/grounded_vision_data', 10)

        # Create publisher for grounded tasks
        self.grounded_tasks_pub = self.create_publisher(Task, 'vla/grounded_tasks', 10)

        # Create service for vision grounding
        self.grounding_service = self.create_service(
            LocalizeObject,
            'vla/vision_ground_object',
            self.ground_object_callback
        )

        # Maintain object database for tracking
        self.object_database = {}
        self.scene_context = {}

        self.get_logger().info('Vision Grounding Node initialized')

    def vision_data_callback(self, msg: VisionData):
        """Callback for processing vision data and grounding it in the scene"""
        try:
            self.get_logger().info(f'Processing vision data with {len(msg.object_detections)} detections')

            # Ground the vision data in the scene context
            grounded_data = self.ground_vision_data(msg)

            # Publish the grounded vision data
            self.grounded_vision_pub.publish(grounded_data)

            # Update object database
            self.update_object_database(msg)

            self.get_logger().info('Vision data grounded successfully')

        except Exception as e:
            self.get_logger().error(f'Error in vision grounding: {str(e)}')

    def ground_vision_data(self, vision_data: VisionData) -> VisionData:
        """
        Ground the vision data in the scene context
        """
        try:
            # Create a copy of the vision data to modify
            grounded_data = VisionData()
            grounded_data.header = vision_data.header
            grounded_data.data_id = f'grounded_{vision_data.data_id}'
            grounded_data.source_camera = vision_data.source_camera
            grounded_data.timestamp = vision_data.timestamp

            # Ground each object detection
            grounded_detections = []
            for detection in vision_data.object_detections:
                grounded_detection = self.ground_object_detection(detection)
                if grounded_detection:
                    grounded_detections.append(grounded_detection)

            grounded_data.object_detections = grounded_detections
            grounded_data.spatial_coordinates = vision_data.spatial_coordinates
            grounded_data.scene_context = self.build_scene_context(grounded_detections)

            return grounded_data

        except Exception as e:
            self.get_logger().error(f'Error grounding vision data: {str(e)}')
            return vision_data  # Return original if grounding fails

    def ground_object_detection(self, detection: ObjectDetection) -> ObjectDetection:
        """
        Ground an individual object detection in the scene context
        """
        try:
            # Add scene context information to the detection
            grounded_detection = ObjectDetection()
            grounded_detection.object_id = detection.object_id
            grounded_detection.class_name = detection.class_name
            grounded_detection.bounding_box = detection.bounding_box
            grounded_detection.position_3d = detection.position_3d
            grounded_detection.confidence = detection.confidence
            grounded_detection.tracking_id = detection.tracking_id
            grounded_detection.attributes = detection.attributes

            # Add spatial context
            spatial_context = self.get_spatial_context(detection.position_3d)
            grounded_detection.attributes.extend(spatial_context)

            return grounded_detection

        except Exception as e:
            self.get_logger().error(f'Error grounding object detection: {str(e)}')
            return detection  # Return original if grounding fails

    def get_spatial_context(self, position) -> List[str]:
        """
        Determine spatial context for an object position
        """
        try:
            # Determine spatial relationships with other objects
            context_attrs = []

            # Example: Check if object is near other known objects
            for obj_id, obj_data in self.object_database.items():
                if obj_id != position.object_id:  # Don't compare with itself
                    dist = self.calculate_distance(position, obj_data['position'])
                    if dist < self.spatial_tolerance:
                        context_attrs.append(f'near:{obj_data["class_name"]}')

            # Determine relative position in scene
            if position.x > 0:
                context_attrs.append('position:right')
            else:
                context_attrs.append('position:left')

            if position.y > 0:
                context_attrs.append('position:front')
            else:
                context_attrs.append('position:back')

            return context_attrs

        except Exception as e:
            self.get_logger().error(f'Error getting spatial context: {str(e)}')
            return []

    def calculate_distance(self, pos1, pos2) -> float:
        """
        Calculate Euclidean distance between two positions
        """
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        dz = pos1.z - pos2.z
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def build_scene_context(self, detections: List[ObjectDetection]) -> str:
        """
        Build a textual description of the scene context
        """
        try:
            # Create a summary of the scene
            scene_elements = []
            for det in detections:
                scene_elements.append(f"{det.class_name} at ({det.position_3d.x:.2f}, {det.position_3d.y:.2f}, {det.position_3d.z:.2f})")

            return f"Scene contains: {', '.join(scene_elements)}"

        except Exception as e:
            self.get_logger().error(f'Error building scene context: {str(e)}')
            return "Scene context unavailable"

    def update_object_database(self, vision_data: VisionData):
        """
        Update the object database with new detections
        """
        try:
            for detection in vision_data.object_detections:
                self.object_database[detection.object_id] = {
                    'class_name': detection.class_name,
                    'position': detection.position_3d,
                    'confidence': detection.confidence,
                    'timestamp': vision_data.timestamp,
                    'attributes': detection.attributes
                }

            # Update scene context
            self.scene_context = {
                'last_updated': vision_data.timestamp,
                'objects_count': len(vision_data.object_detections),
                'detection_confidence_avg': np.mean([det.confidence for det in vision_data.object_detections]) if vision_data.object_detections else 0.0
            }

        except Exception as e:
            self.get_logger().error(f'Error updating object database: {str(e)}')

    def ground_object_callback(self, request, response):
        """
        Service callback for grounding objects in the scene
        """
        try:
            self.get_logger().info(f'Grounding object: {request.object_description}')

            # Find matching objects in our database
            matched_objects = self.find_matching_objects(request.object_description)

            # Create response with grounded detections
            grounded_detections = []
            for obj_id, obj_data in matched_objects:
                detection = ObjectDetection()
                detection.object_id = obj_id
                detection.class_name = obj_data['class_name']
                detection.position_3d = obj_data['position']
                detection.confidence = obj_data['confidence']
                detection.attributes = obj_data['attributes']

                # Set a default bounding box
                detection.bounding_box = [0, 0, 64, 64]  # Default size
                detection.tracking_id = hash(obj_id) % 10000  # Simple tracking ID

                grounded_detections.append(detection)

            response.detections = grounded_detections
            response.success = len(grounded_detections) > 0
            response.message = f'Grounded {len(grounded_detections)} object(s)' if grounded_detections else 'No matching objects found'
            response.timestamp = self.get_clock().now().to_msg()

            self.get_logger().info(f'Object grounding completed: {response.message}')

        except Exception as e:
            self.get_logger().error(f'Error in object grounding: {str(e)}')
            response.success = False
            response.message = f'Error grounding object: {str(e)}'
            response.timestamp = self.get_clock().now().to_msg()

        return response

    def find_matching_objects(self, description: str) -> List[tuple]:
        """
        Find objects in the database that match the description
        """
        try:
            matches = []
            description_lower = description.lower()

            for obj_id, obj_data in self.object_database.items():
                # Simple matching based on class name
                if obj_data['class_name'].lower() in description_lower or description_lower in obj_data['class_name'].lower():
                    matches.append((obj_id, obj_data))
                # Also check attributes
                elif any(attr.lower() in description_lower for attr in obj_data['attributes']):
                    matches.append((obj_id, obj_data))

            return matches

        except Exception as e:
            self.get_logger().error(f'Error finding matching objects: {str(e)}')
            return []


def main(args=None):
    rclpy.init(args=args)

    vision_grounding = VisionGroundingNode()

    try:
        rclpy.spin(vision_grounding)
    except KeyboardInterrupt:
        pass
    finally:
        vision_grounding.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()