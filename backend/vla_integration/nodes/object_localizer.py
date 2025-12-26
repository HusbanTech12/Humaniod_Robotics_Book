#!/usr/bin/env python3

"""
Object Localizer Node for Vision-Language-Action (VLA) Module
Handles precise localization of objects in 3D space using depth and camera data
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
from typing import Dict, List, Any
import time

# Import ROS 2 message types
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Pose, TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
import tf2_ros
import tf2_geometry_msgs

# Import custom message types
from vla_integration.msg import VisionData, ObjectDetection


class ObjectLocalizerNode(Node):
    """
    ROS 2 Node for precise 3D localization of objects using camera and depth information
    """

    def __init__(self):
        super().__init__('object_localizer')

        # Declare parameters
        self.declare_parameter('depth_scale_factor', 0.001)  # Default for depth in mm
        self.declare_parameter('max_depth', 5.0)  # Maximum depth in meters
        self.declare_parameter('min_depth', 0.1)  # Minimum depth in meters
        self.declare_parameter('localization_accuracy', 0.05)  # 5cm accuracy
        self.declare_parameter('reprojection_threshold', 2.0)  # pixels

        # Get parameters
        self.depth_scale_factor = self.get_parameter('depth_scale_factor').value
        self.max_depth = self.get_parameter('max_depth').value
        self.min_depth = self.get_parameter('min_depth').value
        self.localization_accuracy = self.get_parameter('localization_accuracy').value
        self.reprojection_threshold = self.get_parameter('reprojection_threshold').value

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create subscribers
        self.vision_data_sub = self.create_subscription(
            VisionData,
            'vla/grounded_vision_data',
            self.vision_data_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            QoSProfile(depth=10)
        )

        # Create publisher for localized objects
        self.localized_objects_pub = self.create_publisher(VisionData, 'vla/localized_objects', 10)

        # Store camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.camera_info_received = False

        self.get_logger().info('Object Localizer Node initialized')

    def camera_info_callback(self, msg: CameraInfo):
        """Callback for receiving camera calibration parameters"""
        try:
            # Extract camera matrix and distortion coefficients
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)
            self.camera_info_received = True

            self.get_logger().info('Camera parameters updated')

        except Exception as e:
            self.get_logger().error(f'Error processing camera info: {str(e)}')

    def vision_data_callback(self, msg: VisionData):
        """Callback for processing vision data and performing 3D localization"""
        try:
            if not self.camera_info_received:
                self.get_logger().warn('Waiting for camera info...')
                return

            self.get_logger().info(f'Localizing objects from {len(msg.object_detections)} detections')

            # Localize each object in 3D space
            localized_detections = []
            for detection in msg.object_detections:
                localized_detection = self.localize_object_3d(detection, msg)
                if localized_detection:
                    localized_detections.append(localized_detection)

            # Create and publish localized vision data
            localized_vision_data = VisionData()
            localized_vision_data.header = msg.header
            localized_vision_data.header.frame_id = 'map'  # Global frame
            localized_vision_data.data_id = f'localized_{msg.data_id}'
            localized_vision_data.source_camera = msg.source_camera
            localized_vision_data.timestamp = msg.timestamp
            localized_vision_data.object_detections = localized_detections
            localized_vision_data.scene_context = msg.scene_context

            self.localized_objects_pub.publish(localized_vision_data)

            # Broadcast TF transforms for detected objects
            self.broadcast_object_transforms(localized_detections, msg.header.stamp)

            self.get_logger().info(f'Localized {len(localized_detections)} objects in 3D space')

        except Exception as e:
            self.get_logger().error(f'Error in object localization: {str(e)}')

    def localize_object_3d(self, detection: ObjectDetection, vision_data: VisionData) -> ObjectDetection:
        """
        Perform 3D localization of an object using bounding box and depth information
        """
        try:
            # Extract bounding box coordinates
            bbox_x, bbox_y, bbox_width, bbox_height = detection.bounding_box

            # Calculate center of bounding box
            center_x = bbox_x + bbox_width // 2
            center_y = bbox_y + bbox_height // 2

            # Get depth at the center of the bounding box
            # In a real system, we'd use the actual depth image
            # For simulation, we'll use a representative depth value
            depth_value = self.estimate_depth_at_pixel(center_x, center_y)

            if depth_value is None or depth_value < self.min_depth or depth_value > self.max_depth:
                self.get_logger().warn(f'Depth value out of range for object {detection.object_id}: {depth_value}')
                return None

            # Convert pixel coordinates to 3D world coordinates
            world_point = self.pixel_to_world(
                center_x, center_y, depth_value,
                self.camera_matrix, self.distortion_coeffs
            )

            if world_point is not None:
                # Update the detection with accurate 3D position
                localized_detection = ObjectDetection()
                localized_detection.object_id = detection.object_id
                localized_detection.class_name = detection.class_name
                localized_detection.bounding_box = detection.bounding_box
                localized_detection.position_3d.x = world_point[0]
                localized_detection.position_3d.y = world_point[1]
                localized_detection.position_3d.z = world_point[2]
                localized_detection.confidence = detection.confidence
                localized_detection.tracking_id = detection.tracking_id
                localized_detection.attributes = detection.attributes

                # Add localization-specific attributes
                localization_error = self.estimate_localization_error(depth_value, bbox_width, bbox_height)
                localized_detection.attributes.append(f'localization_error:{localization_error:.3f}')

                return localized_detection

        except Exception as e:
            self.get_logger().error(f'Error localizing object {detection.object_id}: {str(e)}')

        return None

    def estimate_depth_at_pixel(self, x: int, y: int) -> float:
        """
        Estimate depth at a given pixel
        In a real system, this would sample the depth image
        For simulation, we'll return a plausible value
        """
        try:
            # For simulation purposes, return a random depth value within valid range
            # In a real system, we would sample the actual depth image
            return np.random.uniform(self.min_depth + 0.1, self.max_depth - 1.0)

        except Exception as e:
            self.get_logger().error(f'Error estimating depth: {str(e)}')
            return None

    def pixel_to_world(self, u: float, v: float, depth: float, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray) -> tuple:
        """
        Convert pixel coordinates to world coordinates using camera parameters
        """
        try:
            # Apply inverse distortion
            undistorted_point = self.undistort_point(u, v, camera_matrix, distortion_coeffs)

            # Convert to world coordinates
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

            x = (undistorted_point[0] - cx) * depth / fx
            y = (undistorted_point[1] - cy) * depth / fy
            z = depth

            return (x, y, z)

        except Exception as e:
            self.get_logger().error(f'Error converting pixel to world: {str(e)}')
            return None

    def undistort_point(self, u: float, v: float, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray) -> tuple:
        """
        Undistort a point using camera distortion coefficients
        """
        try:
            # For simplicity, we'll use a basic distortion model
            # In practice, OpenCV's undistortPoints function would be used
            k1, k2, p1, p2, k3 = distortion_coeffs[:5]

            # Normalize point
            x = (u - camera_matrix[0, 2]) / camera_matrix[0, 0]
            y = (v - camera_matrix[1, 2]) / camera_matrix[1, 1]

            # Apply radial distortion
            r2 = x*x + y*y
            r4 = r2*r2
            r6 = r4*r2

            x_undistorted = x * (1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*x*y + p2*(r2 + 2*x*x)
            y_undisterted = y * (1 + k1*r2 + k2*r4 + k3*r6) + p1*(r2 + 2*y*y) + 2*p2*x*y

            # Convert back to pixel coordinates
            u_undistorted = x_undistorted * camera_matrix[0, 0] + camera_matrix[0, 2]
            v_undistorted = y_undisterted * camera_matrix[1, 1] + camera_matrix[1, 2]

            return (u_undistorted, v_undistorted)

        except Exception as e:
            self.get_logger().error(f'Error undistorting point: {str(e)}')
            # Return original point if undistortion fails
            return (u, v)

    def estimate_localization_error(self, depth: float, width: int, height: int) -> float:
        """
        Estimate the localization error based on depth and bounding box size
        """
        try:
            # Error increases with depth and decreases with object size
            # This is a simplified model - real systems would have more complex error models
            depth_error = 0.01 * depth  # 1% error per meter
            size_error = 0.05 / max(width, height)  # Error inversely proportional to object size

            return min(0.1, depth_error + size_error)  # Cap at 10cm

        except Exception as e:
            self.get_logger().error(f'Error estimating localization error: {str(e)}')
            return 0.05  # Default error

    def broadcast_object_transforms(self, detections: List[ObjectDetection], stamp):
        """
        Broadcast TF transforms for all detected objects
        """
        try:
            for detection in detections:
                transform = TransformStamped()

                transform.header.stamp = stamp
                transform.header.frame_id = 'camera_rgb_optical_frame'  # Camera frame
                transform.child_frame_id = f'object_{detection.object_id}'

                # Set translation
                transform.transform.translation.x = detection.position_3d.x
                transform.transform.translation.y = detection.position_3d.y
                transform.transform.translation.z = detection.position_3d.z

                # Set rotation (identity for point objects)
                transform.transform.rotation.x = 0.0
                transform.transform.rotation.y = 0.0
                transform.transform.rotation.z = 0.0
                transform.transform.rotation.w = 1.0

                # Broadcast the transform
                self.tf_broadcaster.sendTransform(transform)

        except Exception as e:
            self.get_logger().error(f'Error broadcasting object transforms: {str(e)}')


def main(args=None):
    rclpy.init(args=args)

    object_localizer = ObjectLocalizerNode()

    try:
        rclpy.spin(object_localizer)
    except KeyboardInterrupt:
        pass
    finally:
        object_localizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()