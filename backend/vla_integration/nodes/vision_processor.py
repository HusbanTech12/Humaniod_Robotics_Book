#!/usr/bin/env python3

"""
Vision Processor Node for Vision-Language-Action (VLA) Module
Handles Isaac ROS vision processing and object detection
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import cv2
import numpy as np
from cv_bridge import CvBridge
from typing import List, Dict, Any
import time

# Import ROS 2 message types
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from builtin_interfaces.msg import Time

# Import custom message types
from vla_integration.msg import VisionData, ObjectDetection
from vla_integration.srv import LocalizeObject


class VisionProcessorNode(Node):
    """
    ROS 2 Node for processing visual data and performing object detection
    """

    def __init__(self):
        super().__init__('vision_processor')

        # Declare parameters
        self.declare_parameter('camera_info_topic', '/camera/rgb/camera_info')
        self.declare_parameter('image_topic', '/camera/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('max_objects', 10)
        self.declare_parameter('object_classes', ['person', 'bottle', 'cup', 'chair'])

        # Get parameters
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.max_objects = self.get_parameter('max_objects').value
        self.object_classes = self.get_parameter('object_classes').value

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            QoSProfile(depth=10)
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            QoSProfile(depth=10)
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            QoSProfile(depth=10)
        )

        # Create publisher for vision data
        self.vision_data_pub = self.create_publisher(VisionData, 'vla/vision_data', 10)

        # Create service for object localization
        self.localize_service = self.create_service(
            LocalizeObject,
            'vla/localize_object',
            self.localize_object_callback
        )

        # Initialize camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.latest_image = None
        self.latest_depth = None

        self.get_logger().info('Vision Processor Node initialized')

    def image_callback(self, msg: Image):
        """Callback for processing incoming images"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Process the image for object detection
            detections = self.detect_objects(cv_image)

            # Create and publish vision data
            vision_data = self.create_vision_data(detections, msg.header)
            self.vision_data_pub.publish(vision_data)

            self.latest_image = cv_image

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def depth_callback(self, msg: Image):
        """Callback for processing depth images"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth = cv_depth

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def camera_info_callback(self, msg: CameraInfo):
        """Callback for camera calibration parameters"""
        try:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)

        except Exception as e:
            self.get_logger().error(f'Error processing camera info: {str(e)}')

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the image using a pre-trained model
        This is a simplified implementation - in reality, this would use Isaac ROS detection
        """
        try:
            # In a real implementation, this would use Isaac ROS detectnet
            # For now, we'll simulate object detection

            detections = []

            # Convert image to grayscale for simpler processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Simulate object detection by finding contours (as an example)
            # In reality, this would use a deep learning model
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process each contour as a potential object
            for i, contour in enumerate(contours[:self.max_objects]):
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate center of the object
                center_x = x + w // 2
                center_y = y + h // 2

                # Create a simulated detection
                detection = {
                    'object_id': f'obj_{int(time.time())}_{i}',
                    'class_name': np.random.choice(self.object_classes),
                    'bounding_box': [int(x), int(y), int(w), int(h)],
                    'center': (center_x, center_y),
                    'confidence': np.random.uniform(self.confidence_threshold, 1.0)
                }

                detections.append(detection)

            return detections

        except Exception as e:
            self.get_logger().error(f'Error in object detection: {str(e)}')
            return []

    def create_vision_data(self, detections: List[Dict[str, Any]], header) -> VisionData:
        """Create VisionData message from detections"""
        vision_data = VisionData()
        vision_data.header = header
        vision_data.data_id = f'vision_{int(time.time())}'
        vision_data.source_camera = 'front_camera'
        vision_data.timestamp = self.get_clock().now().to_msg()

        # Convert detections to ObjectDetection messages
        object_detections = []
        spatial_coords = []

        for det in detections:
            obj_det = ObjectDetection()
            obj_det.object_id = det['object_id']
            obj_det.class_name = det['class_name']
            obj_det.bounding_box = det['bounding_box']
            obj_det.confidence = det['confidence']

            # Create spatial coordinate (simplified - would use depth and camera matrix in reality)
            pos = PointStamped()
            pos.point.x = det['center'][0] * 0.01  # Scale to meters
            pos.point.y = det['center'][1] * 0.01  # Scale to meters
            pos.point.z = 1.0  # Default depth if no depth data available
            spatial_coords.append(pos)

            # Add attributes
            obj_det.attributes = ['color:red' if 'red' in det['class_name'] else 'color:unknown']

            object_detections.append(obj_det)

        vision_data.object_detections = object_detections
        vision_data.spatial_coordinates = spatial_coords
        vision_data.scene_context = 'indoor_environment'

        return vision_data

    def localize_object_callback(self, request, response):
        """Service callback for localizing objects in the environment"""
        try:
            self.get_logger().info(f'Localizing object: {request.object_description}')

            # In a real implementation, this would search the current vision data
            # For now, we'll simulate the localization

            # Create simulated detections based on the request
            simulated_detections = []

            # If the requested object type is in our known classes, create a detection
            if any(obj_class in request.object_description.lower() for obj_class in self.object_classes):
                detection = ObjectDetection()
                detection.object_id = f'localized_obj_{int(time.time())}'
                detection.class_name = request.object_description.split()[0] if request.object_description.split() else 'unknown'
                detection.bounding_box = [100, 100, 50, 50]  # x, y, width, height
                detection.position_3d.x = 1.0
                detection.position_3d.y = 0.5
                detection.position_3d.z = 0.9
                detection.confidence = 0.85
                detection.tracking_id = 1
                detection.attributes = ['color:red']

                simulated_detections.append(detection)

            response.detections = simulated_detections
            response.success = len(simulated_detections) > 0
            response.message = f'Located {len(simulated_detections)} object(s)' if simulated_detections else 'No objects found'
            response.timestamp = self.get_clock().now().to_msg()

            self.get_logger().info(f'Object localization completed: {response.message}')

        except Exception as e:
            self.get_logger().error(f'Error in object localization: {str(e)}')
            response.success = False
            response.message = f'Error localizing object: {str(e)}'
            response.timestamp = self.get_clock().now().to_msg()

        return response


def main(args=None):
    rclpy.init(args=args)

    vision_processor = VisionProcessorNode()

    try:
        rclpy.spin(vision_processor)
    except KeyboardInterrupt:
        pass
    finally:
        vision_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()