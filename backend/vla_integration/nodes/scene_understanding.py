#!/usr/bin/env python3

"""
Scene Understanding Node for Vision-Language-Action (VLA) Module
Handles scene context integration and spatial reasoning
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
from typing import Dict, List, Any, Tuple
import json
import time
from collections import defaultdict

# Import custom message types
from vla_integration.msg import VisionData, ObjectDetection, ActionPlan, Task


class SceneUnderstandingNode(Node):
    """
    ROS 2 Node for understanding the scene context and spatial relationships
    between objects to support planning and execution
    """

    def __init__(self):
        super().__init__('scene_understanding')

        # Declare parameters
        self.declare_parameter('spatial_relationship_threshold', 0.5)  # meters
        self.declare_parameter('scene_context_window', 30.0)  # seconds
        self.declare_parameter('object_persistence_time', 60.0)  # seconds
        self.declare_parameter('spatial_reasoning_enabled', True)

        # Get parameters
        self.spatial_relationship_threshold = self.get_parameter('spatial_relationship_threshold').value
        self.scene_context_window = self.get_parameter('scene_context_window').value
        self.object_persistence_time = self.get_parameter('object_persistence_time').value
        self.spatial_reasoning_enabled = self.get_parameter('spatial_reasoning_enabled').value

        # Create subscriber for vision data
        self.vision_data_sub = self.create_subscription(
            VisionData,
            'vla/localized_objects',
            self.vision_data_callback,
            10
        )

        # Create publisher for scene context
        self.scene_context_pub = self.create_publisher(VisionData, 'vla/scene_context', 10)

        # Create publisher for spatial relationships
        self.spatial_relations_pub = self.create_publisher(VisionData, 'vla/spatial_relationships', 10)

        # Initialize scene context storage
        self.object_database = {}  # Stores object information with timestamps
        self.spatial_graph = defaultdict(list)  # Graph of spatial relationships
        self.scene_history = []  # History of scene states

        self.get_logger().info('Scene Understanding Node initialized')

    def vision_data_callback(self, msg: VisionData):
        """Callback for processing vision data and updating scene understanding"""
        try:
            self.get_logger().info(f'Processing scene with {len(msg.object_detections)} objects')

            # Update object database with new detections
            self.update_object_database(msg)

            # Compute spatial relationships
            if self.spatial_reasoning_enabled:
                self.compute_spatial_relationships()

            # Update scene context
            scene_context = self.build_scene_context(msg)

            # Publish updated scene context
            self.scene_context_pub.publish(scene_context)

            # Publish spatial relationships
            spatial_context = self.build_spatial_context(msg)
            self.spatial_relations_pub.publish(spatial_context)

            # Store in scene history
            self.scene_history.append({
                'timestamp': msg.timestamp,
                'object_count': len(msg.object_detections),
                'spatial_relationships': dict(self.spatial_graph),
                'context': scene_context
            })

            # Clean up old history entries
            self.cleanup_old_entries()

            self.get_logger().info(f'Scene understanding updated with {len(msg.object_detections)} objects')

        except Exception as e:
            self.get_logger().error(f'Error in scene understanding: {str(e)}')

    def update_object_database(self, vision_data: VisionData):
        """Update the object database with new vision data"""
        try:
            current_time = time.time()

            for detection in vision_data.object_detections:
                # Store object with timestamp
                self.object_database[detection.object_id] = {
                    'detection': detection,
                    'timestamp': current_time,
                    'position': detection.position_3d,
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'attributes': detection.attributes
                }

            # Clean up expired objects
            self.cleanup_expired_objects()

        except Exception as e:
            self.get_logger().error(f'Error updating object database: {str(e)}')

    def cleanup_expired_objects(self):
        """Remove objects that have been in the database too long"""
        try:
            current_time = time.time()
            expired_ids = []

            for obj_id, obj_data in self.object_database.items():
                if current_time - obj_data['timestamp'] > self.object_persistence_time:
                    expired_ids.append(obj_id)

            for obj_id in expired_ids:
                del self.object_database[obj_id]
                self.get_logger().info(f'Removed expired object: {obj_id}')

        except Exception as e:
            self.get_logger().error(f'Error cleaning up expired objects: {str(e)}')

    def compute_spatial_relationships(self):
        """Compute spatial relationships between objects"""
        try:
            # Clear previous relationships
            self.spatial_graph.clear()

            # Get all object positions
            obj_positions = {}
            for obj_id, obj_data in self.object_database.items():
                pos = obj_data['position']
                obj_positions[obj_id] = np.array([pos.x, pos.y, pos.z])

            # Compute relationships between all pairs of objects
            obj_ids = list(obj_positions.keys())
            for i in range(len(obj_ids)):
                for j in range(i + 1, len(obj_ids)):
                    id1, id2 = obj_ids[i], obj_ids[j]
                    pos1, pos2 = obj_positions[id1], obj_positions[id2]

                    # Calculate distance
                    distance = np.linalg.norm(pos1 - pos2)

                    # If objects are close enough, create relationship
                    if distance <= self.spatial_relationship_threshold:
                        relationship = {
                            'object_id': id2,
                            'distance': distance,
                            'relative_position': (pos2 - pos1).tolist(),
                            'relationship_type': self.classify_relationship(pos1, pos2, distance)
                        }
                        self.spatial_graph[id1].append(relationship)

                        # Add reverse relationship
                        reverse_relationship = {
                            'object_id': id1,
                            'distance': distance,
                            'relative_position': (pos1 - pos2).tolist(),
                            'relationship_type': self.classify_relationship(pos2, pos1, distance)
                        }
                        self.spatial_graph[id2].append(reverse_relationship)

        except Exception as e:
            self.get_logger().error(f'Error computing spatial relationships: {str(e)}')

    def classify_relationship(self, pos1: np.ndarray, pos2: np.ndarray, distance: float) -> str:
        """Classify the spatial relationship between two objects"""
        try:
            # Calculate vector from pos1 to pos2
            vec = pos2 - pos1

            # Determine primary direction
            dominant_axis = np.argmax(np.abs(vec))
            direction = ['x', 'y', 'z'][dominant_axis]

            # Determine if it's positive or negative direction
            sign = '+' if vec[dominant_axis] > 0 else '-'

            # Classify based on distance
            if distance < 0.1:
                return 'on_top_of' if abs(vec[2]) > 0.05 and vec[2] > 0 else 'touching'
            elif distance < 0.3:
                return f'near_{sign}{direction}'
            elif distance < 0.8:
                return f'close_to_{sign}{direction}'
            else:
                return f'far_from_{sign}{direction}'

        except Exception as e:
            self.get_logger().error(f'Error classifying relationship: {str(e)}')
            return 'unknown'

    def build_scene_context(self, vision_data: VisionData) -> VisionData:
        """Build comprehensive scene context from vision data"""
        try:
            # Create new vision data message for scene context
            scene_context = VisionData()
            scene_context.header = vision_data.header
            scene_context.header.frame_id = 'scene_understanding'
            scene_context.data_id = f'scene_context_{int(time.time())}'
            scene_context.source_camera = vision_data.source_camera
            scene_context.timestamp = vision_data.timestamp

            # Copy object detections
            scene_context.object_detections = vision_data.object_detections

            # Build scene description
            scene_description = self.describe_scene(vision_data)
            scene_context.scene_context = scene_description

            # Add spatial context as attributes to detections
            for detection in scene_context.object_detections:
                spatial_attrs = self.get_spatial_attributes(detection.object_id)
                detection.attributes.extend(spatial_attrs)

            return scene_context

        except Exception as e:
            self.get_logger().error(f'Error building scene context: {str(e)}')
            return vision_data

    def describe_scene(self, vision_data: VisionData) -> str:
        """Create a textual description of the scene"""
        try:
            # Count objects by type
            obj_counts = defaultdict(int)
            for detection in vision_data.object_detections:
                obj_counts[detection.class_name] += 1

            # Create object summary
            obj_summaries = []
            for obj_type, count in obj_counts.items():
                if count == 1:
                    obj_summaries.append(f'1 {obj_type}')
                else:
                    obj_summaries.append(f'{count} {obj_type}s')

            # Get spatial relationships
            relationship_summary = self.summarize_relationships()

            # Combine into scene description
            scene_desc = f"Scene contains: {', '.join(obj_summaries)}. "
            if relationship_summary:
                scene_desc += f"Spatial relationships: {relationship_summary}"

            return scene_desc

        except Exception as e:
            self.get_logger().error(f'Error describing scene: {str(e)}')
            return "Scene context unavailable"

    def get_spatial_attributes(self, obj_id: str) -> List[str]:
        """Get spatial attributes for an object based on relationships"""
        try:
            attrs = []

            # Add proximity information
            relationships = self.spatial_graph.get(obj_id, [])
            nearby_objects = [rel['object_id'] for rel in relationships if rel['distance'] < 0.5]
            if nearby_objects:
                attrs.append(f'near_objects:{",".join(nearby_objects[:3])}')  # Limit to 3 nearby objects

            # Add positional information
            if obj_id in self.object_database:
                pos = self.object_database[obj_id]['position']
                attrs.append(f'pos_x:{pos.x:.2f}')
                attrs.append(f'pos_y:{pos.y:.2f}')
                attrs.append(f'pos_z:{pos.z:.2f}')

            return attrs

        except Exception as e:
            self.get_logger().error(f'Error getting spatial attributes: {str(e)}')
            return []

    def build_spatial_context(self, vision_data: VisionData) -> VisionData:
        """Build spatial context message with relationship information"""
        try:
            spatial_context = VisionData()
            spatial_context.header = vision_data.header
            spatial_context.header.frame_id = 'spatial_reasoning'
            spatial_context.data_id = f'spatial_context_{int(time.time())}'
            spatial_context.source_camera = vision_data.source_camera
            spatial_context.timestamp = vision_data.timestamp

            # Create object detections with spatial relationship information
            spatial_detections = []
            for detection in vision_data.object_detections:
                spatial_detection = ObjectDetection()
                spatial_detection.object_id = detection.object_id
                spatial_detection.class_name = detection.class_name
                spatial_detection.bounding_box = detection.bounding_box
                spatial_detection.position_3d = detection.position_3d
                spatial_detection.confidence = detection.confidence
                spatial_detection.tracking_id = detection.tracking_id
                spatial_detection.attributes = detection.attributes.copy()

                # Add spatial relationships
                relationships = self.spatial_graph.get(detection.object_id, [])
                for rel in relationships:
                    rel_attr = f"related_to:{rel['object_id']}_{rel['relationship_type']}_dist_{rel['distance']:.2f}"
                    spatial_detection.attributes.append(rel_attr)

                spatial_detections.append(spatial_detection)

            spatial_context.object_detections = spatial_detections
            spatial_context.scene_context = self.summarize_relationships()

            return spatial_context

        except Exception as e:
            self.get_logger().error(f'Error building spatial context: {str(e)}')
            return vision_data

    def summarize_relationships(self) -> str:
        """Summarize the current spatial relationships"""
        try:
            summaries = []
            for obj_id, relationships in self.spatial_graph.items():
                if relationships:
                    rel_strs = []
                    for rel in relationships[:3]:  # Limit to first 3 relationships
                        rel_strs.append(f"{rel['object_id']}({rel['relationship_type']})")
                    summaries.append(f"{obj_id}: {', '.join(rel_strs)}")

            return '; '.join(summaries[:5])  # Limit to first 5 objects

        except Exception as e:
            self.get_logger().error(f'Error summarizing relationships: {str(e)}')
            return ""

    def cleanup_old_entries(self):
        """Clean up old entries in scene history"""
        try:
            current_time = time.time()
            cutoff_time = current_time - self.scene_context_window

            # Remove entries older than the window
            self.scene_history = [
                entry for entry in self.scene_history
                if isinstance(entry.get('timestamp'), (int, float)) and entry['timestamp'] > cutoff_time
            ]

        except Exception as e:
            self.get_logger().error(f'Error cleaning up scene history: {str(e)}')


def main(args=None):
    rclpy.init(args=args)

    scene_understanding = SceneUnderstandingNode()

    try:
        rclpy.spin(scene_understanding)
    except KeyboardInterrupt:
        pass
    finally:
        scene_understanding.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()