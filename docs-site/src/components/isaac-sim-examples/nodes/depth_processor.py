#!/usr/bin/env python3
# depth_processor.py
"""
Isaac ROS Depth Processing Node
This node processes depth information from stereo cameras or depth sensors,
performing depth refinement, filtering, and conversion to point clouds.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.duration import Duration

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import Point

import message_filters
from message_filters import ApproximateTimeSynchronizer


class DepthProcessorNode(Node):
    """
    Isaac ROS Depth Processing Node
    Processes depth images from stereo cameras or depth sensors
    """

    def __init__(self):
        super().__init__('depth_processor')

        # Declare parameters
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('fill_holes', True)
        self.declare_parameter('hole_size', 3)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('depth_unit', 'meters')  # 'meters' or 'millimeters'
        self.declare_parameter('enable_pointcloud', True)
        self.declare_parameter('pointcloud_resolution', 1)  # Every Nth pixel
        self.declare_parameter('enable_filtering', True)
        self.declare_parameter('median_filter_size', 3)
        self.declare_parameter('bilateral_filter', True)
        self.declare_parameter('bilateral_diameter', 5)
        self.declare_parameter('bilateral_sigma_color', 75)
        self.declare_parameter('bilateral_sigma_space', 75)
        self.declare_parameter('enable_edge_preservation', True)
        self.declare_parameter('edge_threshold', 0.1)

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.fill_holes = self.get_parameter('fill_holes').value
        self.hole_size = self.get_parameter('hole_size').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.depth_unit = self.get_parameter('depth_unit').value
        self.enable_pointcloud = self.get_parameter('enable_pointcloud').value
        self.pointcloud_resolution = self.get_parameter('pointcloud_resolution').value
        self.enable_filtering = self.get_parameter('enable_filtering').value
        self.median_filter_size = self.get_parameter('median_filter_size').value
        self.bilateral_filter = self.get_parameter('bilateral_filter').value
        self.bilateral_diameter = self.get_parameter('bilateral_diameter').value
        self.bilateral_sigma_color = self.get_parameter('bilateral_sigma_color').value
        self.bilateral_sigma_space = self.get_parameter('bilateral_sigma_space').value
        self.enable_edge_preservation = self.get_parameter('enable_edge_preservation').value
        self.edge_threshold = self.get_parameter('edge_threshold').value

        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()

        # Create QoS profile for sensor data
        self.qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Create publishers
        self.depth_pub = self.create_publisher(
            Image,
            'depth/processed',
            self.qos_profile
        )

        if self.enable_pointcloud:
            self.pointcloud_pub = self.create_publisher(
                PointCloud2,
                'depth/pointcloud',
                self.qos_profile
            )

        # Create subscribers
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            'depth/image_raw',
            qos_profile=self.qos_profile
        )

        self.camera_info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            'depth/camera_info',
            qos_profile=self.qos_profile
        )

        # Create approximate time synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.depth_sub, self.camera_info_sub],
            queue_size=5,
            slop=0.1
        )
        self.ts.registerCallback(self.depth_callback)

        self.get_logger().info('Isaac ROS Depth Processor Node initialized')
        self.get_logger().info(f'Parameters: min_depth={self.min_depth}, max_depth={self.max_depth}, fill_holes={self.fill_holes}')

    def depth_callback(self, depth_msg, camera_info_msg):
        """
        Process synchronized depth image and camera info
        """
        try:
            # Convert ROS Image to OpenCV
            cv_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Process depth image
            processed_depth = self.process_depth_image(cv_depth)

            # Create processed depth image message
            processed_depth_msg = self.cv_bridge.cv2_to_imgmsg(
                processed_depth,
                encoding=depth_msg.encoding
            )
            processed_depth_msg.header = depth_msg.header

            # Publish processed depth
            self.depth_pub.publish(processed_depth_msg)

            # Generate and publish point cloud if enabled
            if self.enable_pointcloud:
                pointcloud_msg = self.generate_pointcloud(
                    processed_depth,
                    camera_info_msg
                )
                self.pointcloud_pub.publish(pointcloud_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def process_depth_image(self, depth_image):
        """
        Apply various processing techniques to depth image
        """
        # Convert to float32 for processing
        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)

        # Apply min/max depth filtering
        depth_image = self.apply_depth_filtering(depth_image)

        # Apply filtering if enabled
        if self.enable_filtering:
            depth_image = self.apply_filtering(depth_image)

        # Fill holes if enabled
        if self.fill_holes:
            depth_image = self.fill_depth_holes(depth_image)

        # Apply edge preservation if enabled
        if self.enable_edge_preservation:
            depth_image = self.preserve_depth_edges(depth_image)

        return depth_image

    def apply_depth_filtering(self, depth_image):
        """
        Apply various filtering techniques to depth image
        """
        # Median filtering
        if self.median_filter_size > 1:
            depth_image = cv2.medianBlur(depth_image, self.median_filter_size)

        # Bilateral filtering for noise reduction while preserving edges
        if self.bilateral_filter:
            depth_image = cv2.bilateralFilter(
                depth_image,
                self.bilateral_diameter,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space
            )

        return depth_image

    def apply_depth_filtering(self, depth_image):
        """
        Apply min/max depth filtering
        """
        # Set values outside range to zero (invalid)
        depth_image = np.where(
            (depth_image >= self.min_depth) & (depth_image <= self.max_depth),
            depth_image,
            0.0
        )

        # Apply median filtering
        if self.median_filter_size > 1:
            # Only apply median filter to valid depth values
            valid_mask = depth_image > 0
            if np.any(valid_mask):
                filtered_depth = cv2.medianBlur(depth_image, self.median_filter_size)
                depth_image[valid_mask] = filtered_depth[valid_mask]

        # Apply bilateral filtering for noise reduction while preserving edges
        if self.bilateral_filter and np.any(depth_image > 0):
            depth_image = cv2.bilateralFilter(
                depth_image,
                self.bilateral_diameter,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space
            )

        return depth_image

    def fill_depth_holes(self, depth_image):
        """
        Fill holes in depth image using inpainting
        """
        # Create mask of invalid pixels (0 or NaN)
        invalid_mask = (depth_image == 0) | np.isnan(depth_image) | np.isinf(depth_image)

        if np.any(invalid_mask):
            # Convert mask to uint8 for inpainting
            mask_uint8 = invalid_mask.astype(np.uint8) * 255

            # Use inpainting to fill holes
            filled_depth = cv2.inpaint(
                depth_image.astype(np.float32),
                mask_uint8,
                self.hole_size,
                cv2.INPAINT_TELEA
            )

            # Only update invalid pixels
            depth_image[invalid_mask] = filled_depth[invalid_mask]

        return depth_image

    def preserve_depth_edges(self, depth_image):
        """
        Preserve depth edges while smoothing
        """
        # Calculate gradients to detect edges
        grad_x = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Create edge mask
        edge_mask = gradient_magnitude > self.edge_threshold

        # Apply different smoothing to edge vs non-edge regions
        non_edge_region = ~edge_mask
        if np.any(non_edge_region):
            # Apply more aggressive smoothing to non-edge regions
            smoothed = cv2.GaussianBlur(depth_image, (5, 5), 0)
            depth_image[non_edge_region] = smoothed[non_edge_region]

        return depth_image

    def generate_pointcloud(self, depth_image, camera_info):
        """
        Generate point cloud from depth image and camera info
        """
        # Extract camera parameters
        fx = camera_info.k[0]  # Focal length x
        fy = camera_info.k[4]  # Focal length y
        cx = camera_info.k[2]  # Principal point x
        cy = camera_info.k[5]  # Principal point y

        height, width = depth_image.shape

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Convert pixel coordinates to camera coordinates
        z = depth_image  # Depth values
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Flatten arrays and combine
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

        # Remove invalid points (where depth is 0 or invalid)
        valid_mask = (z.flatten() > 0) & (np.isfinite(points).all(axis=1))
        valid_points = points[valid_mask]

        # Sample points based on resolution
        if self.pointcloud_resolution > 1:
            sample_mask = np.arange(len(valid_points)) % self.pointcloud_resolution == 0
            valid_points = valid_points[sample_mask]

        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = camera_info.header.frame_id

        # Define point fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Create point cloud
        pointcloud_msg = point_cloud2.create_cloud(header, fields, valid_points)

        return pointcloud_msg


def main(args=None):
    """
    Main function to run the depth processor node
    """
    rclpy.init(args=args)

    depth_processor = DepthProcessorNode()

    try:
        rclpy.spin(depth_processor)
    except KeyboardInterrupt:
        pass
    finally:
        depth_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()