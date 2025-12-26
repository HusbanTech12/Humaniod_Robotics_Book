#!/usr/bin/env python3

"""
Vision Service Node for Vision-Language-Action (VLA) Module
API contract implementation for /vision/localize_object
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
from typing import Dict, List, Any
import json
import time

# Import ROS 2 message types
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo

# Import custom message types
from vla_integration.msg import VisionData, ObjectDetection
from vla_integration.srv import LocalizeObject


class VisionServiceNode(Node):
    """
    ROS 2 Node implementing the vision localization API service
    """

    def __init__(self):
        super().__init__('vision_service')

        # Declare parameters
        self.declare_parameter('service_timeout', 15.0)
        self.declare_parameter('max_concurrent_requests', 3)
        self.declare_parameter('object_detection_accuracy', 0.85)
        self.declare_parameter('spatial_accuracy', 0.05)  # meters

        # Get parameters
        self.service_timeout = self.get_parameter('service_timeout').value
        self.max_concurrent_requests = self.get_parameter('max_concurrent_requests').value
        self.object_detection_accuracy = self.get_parameter('object_detection_accuracy').value
        self.spatial_accuracy = self.get_parameter('spatial_accuracy').value

        # Create service for object localization
        self.localize_service = self.create_service(
            LocalizeObject,
            'vla/vision/localize_object',
            self.localize_object_callback
        )

        # Create subscriber for vision data
        self.vision_data_sub = self.create_subscription(
            VisionData,
            'vla/localized_objects',
            self.vision_data_callback,
            10
        )

        # Create publisher for localized objects
        self.localized_objects_pub = self.create_publisher(VisionData, 'vla/service_localized_objects', 10)

        # Store recent vision data for service queries
        self.recent_vision_data = None
        self.last_vision_update = 0.0

        # Store object database
        self.object_database = {}

        self.get_logger().info('Vision Service Node initialized')

    def vision_data_callback(self, msg: VisionData):
        """Callback for storing recent vision data"""
        try:
            self.recent_vision_data = msg
            self.last_vision_update = time.time()

            # Update object database
            for detection in msg.object_detections:
                self.object_database[detection.object_id] = {
                    'detection': detection,
                    'timestamp': msg.timestamp,
                    'confidence': detection.confidence
                }

            self.get_logger().info(f'Updated vision data with {len(msg.object_detections)} objects')

        except Exception as e:
            self.get_logger().error(f'Error in vision data callback: {str(e)}')

    def localize_object_callback(self, request, response):
        """
        Service callback for localizing objects in the environment
        Expected request format based on API contract:
        {
          "object_description": "red cup",
          "camera_view": "front_camera",
          "search_area": {
            "x_range": [-2.0, 2.0],
            "y_range": [-2.0, 2.0],
            "z_range": [0.0, 1.5]
          }
        }

        Expected response format:
        {
          "detections": [
            {
              "object_id": "obj_001",
              "class": "cup",
              "color": "red",
              "position": {"x": 1.2, "y": 0.8, "z": 0.95},
              "confidence": 0.89,
              "bounding_box": {"x": 120, "y": 85, "width": 45, "height": 60}
            }
          ],
          "timestamp": "2025-12-25T10:00:05Z"
        }
        """
        try:
            self.get_logger().info(f'Localizing object: {request.object_description}')

            # Check if we have recent vision data
            if self.recent_vision_data is None or (time.time() - self.last_vision_update) > 5.0:
                self.get_logger().warn('Using outdated or no vision data for localization')
                # We might want to trigger a new vision processing cycle here

            # Find objects matching the description
            matching_detections = self.find_matching_objects(
                request.object_description,
                request.search_area if hasattr(request, 'search_area') else None
            )

            # Convert detections to response format
            response_detections = []
            for detection in matching_detections:
                response_det = ObjectDetection()
                response_det.object_id = detection.object_id
                response_det.class_name = detection.class_name
                response_det.bounding_box = detection.bounding_box
                response_det.position_3d = detection.position_3d
                response_det.confidence = detection.confidence
                response_det.tracking_id = detection.tracking_id
                response_det.attributes = detection.attributes

                response_detections.append(response_det)

            response.detections = response_detections
            response.success = len(response_detections) > 0
            response.message = f'Located {len(response_detections)} object(s)' if response_detections else 'No objects found'
            response.timestamp = self.get_clock().now().to_msg()

            self.get_logger().info(f'Object localization completed: {response.message}')

            # Publish the localized objects
            if response_detections:
                vision_data = self.create_vision_data_from_detections(response_detections, response.timestamp)
                self.localized_objects_pub.publish(vision_data)

        except Exception as e:
            self.get_logger().error(f'Error in object localization service: {str(e)}')
            response.success = False
            response.message = f'Error localizing object: {str(e)}'
            response.timestamp = self.get_clock().now().to_msg()

        return response

    def find_matching_objects(self, description: str, search_area: Any = None) -> List[ObjectDetection]:
        """
        Find objects in the database that match the description
        """
        try:
            matches = []
            description_lower = description.lower()

            # Search through stored objects
            for obj_id, obj_data in self.object_database.items():
                detection = obj_data['detection']

                # Check if the object matches the description
                if self.matches_description(detection, description_lower):
                    # Check if object is in search area if specified
                    if search_area is None or self.in_search_area(detection.position_3d, search_area):
                        matches.append(detection)

            # Sort by confidence (highest first)
            matches.sort(key=lambda x: x.confidence, reverse=True)

            return matches

        except Exception as e:
            self.get_logger().error(f'Error finding matching objects: {str(e)}')
            return []

    def matches_description(self, detection: ObjectDetection, description: str) -> bool:
        """
        Check if a detection matches the description
        """
        try:
            # Check class name
            if detection.class_name.lower() in description or description in detection.class_name.lower():
                return True

            # Check attributes (like color)
            for attr in detection.attributes:
                if attr.startswith('color:') and attr.split(':')[1].lower() in description:
                    return True
                elif attr in description:
                    return True

            # Check if description contains common object types
            common_objects = ['cup', 'bottle', 'chair', 'table', 'person', 'book', 'box', 'plant']
            for obj in common_objects:
                if obj in description and obj in detection.class_name.lower():
                    return True

            return False

        except Exception as e:
            self.get_logger().error(f'Error matching description: {str(e)}')
            return False

    def in_search_area(self, position: Point, search_area: Any) -> bool:
        """
        Check if a position is within the specified search area
        """
        try:
            # In a real implementation, search_area would have x_range, y_range, z_range
            # For now, we'll use default ranges if not specified
            x_min, x_max = getattr(search_area, 'x_range', [-np.inf, np.inf])
            y_min, y_max = getattr(search_area, 'y_range', [-np.inf, np.inf])
            z_min, z_max = getattr(search_area, 'z_range', [-np.inf, np.inf])

            return (x_min <= position.x <= x_max and
                    y_min <= position.y <= y_max and
                    z_min <= position.z <= z_max)

        except Exception as e:
            self.get_logger().error(f'Error checking search area: {str(e)}')
            return True  # Return True if there's an error to avoid excluding objects

    def create_vision_data_from_detections(self, detections: List[ObjectDetection], timestamp) -> VisionData:
        """
        Create a VisionData message from a list of detections
        """
        try:
            vision_data = VisionData()
            vision_data.header.stamp = timestamp
            vision_data.header.frame_id = 'vision_service'
            vision_data.data_id = f'service_response_{int(time.time())}'
            vision_data.source_camera = 'service_query'
            vision_data.timestamp = timestamp
            vision_data.object_detections = detections
            vision_data.scene_context = 'queried_objects'

            return vision_data

        except Exception as e:
            self.get_logger().error(f'Error creating vision data from detections: {str(e)}')
            return VisionData()


def main(args=None):
    rclpy.init(args=args)

    vision_service = VisionServiceNode()

    try:
        rclpy.spin(vision_service)
    except KeyboardInterrupt:
        pass
    finally:
        vision_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()