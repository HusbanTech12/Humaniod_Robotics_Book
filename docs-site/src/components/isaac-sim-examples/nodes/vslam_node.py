#!/usr/bin/env python3
# vslam_node.py

"""
Isaac ROS VSLAM Node
This node implements the Visual SLAM pipeline for mapping and localization
using Isaac ROS components and camera data from the humanoid robot.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.duration import Duration

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

import message_filters
from message_filters import ApproximateTimeSynchronizer
import tf2_ros


class VSLAMNode(Node):
    """
    Isaac ROS Visual SLAM Node
    Implements mapping and localization using visual data from cameras
    """

    def __init__(self):
        super().__init__('vslam_node')

        # Declare parameters
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('enable_mapping', True)
        self.declare_parameter('enable_localization', True)
        self.declare_parameter('enable_loop_closure', True)
        self.declare_parameter('map_resolution', 0.05)  # meters per cell
        self.declare_parameter('map_width', 200)  # cells
        self.declare_parameter('map_height', 200)  # cells
        self.declare_parameter('max_features', 1000)
        self.declare_parameter('min_feature_distance', 10.0)
        self.declare_parameter('feature_quality_level', 0.01)
        self.declare_parameter('max_num_landmarks', 10000)
        self.declare_parameter('min_tracked_landmarks', 10)
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('camera_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.enable_mapping = self.get_parameter('enable_mapping').value
        self.enable_localization = self.get_parameter('enable_localization').value
        self.enable_loop_closure = self.get_parameter('enable_loop_closure').value
        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        self.max_features = self.get_parameter('max_features').value
        self.min_feature_distance = self.get_parameter('min_feature_distance').value
        self.feature_quality_level = self.get_parameter('feature_quality_level').value
        self.max_num_landmarks = self.get_parameter('max_num_landmarks').value
        self.min_tracked_landmarks = self.get_parameter('min_tracked_landmarks').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value

        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()

        # TF broadcaster for transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create QoS profile for sensor data
        self.qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Initialize SLAM components
        self.initialize_slam_components()

        # Create publishers
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            'vslam/occupancy_grid',
            self.qos_profile
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            'vslam/pose',
            self.qos_profile
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            'vslam/odometry',
            self.qos_profile
        )

        if self.enable_visualization:
            self.visualization_pub = self.create_publisher(
                PointCloud2,
                'vslam/visualization',
                self.qos_profile
            )

        # Create subscribers
        self.image_sub = message_filters.Subscriber(
            self,
            Image,
            'camera/rgb/image_rect_color',
            qos_profile=self.qos_profile
        )

        self.camera_info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            'camera/rgb/camera_info',
            qos_profile=self.qos_profile
        )

        # Create approximate time synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.camera_info_sub],
            queue_size=5,
            slop=0.1
        )
        self.ts.registerCallback(self.vslam_callback)

        # Timer for publishing map and transforms
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_results
        )

        self.get_logger().info('Isaac ROS VSLAM Node initialized')
        self.get_logger().info(f'Parameters: map_resolution={self.map_resolution}, max_features={self.max_features}')

    def initialize_slam_components(self):
        """
        Initialize SLAM components including feature detectors, map, and pose tracking
        """
        # Initialize feature detector (ORB for good performance/stability balance)
        self.feature_detector = cv2.ORB_create(
            nfeatures=self.max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=19,
            patchSize=31,
            fastThreshold=20
        )

        # Initialize map as numpy array
        self.map_data = np.full((self.map_height, self.map_width), -1, dtype=np.int8)  # -1 = unknown
        self.map_origin_x = -self.map_width * self.map_resolution / 2.0
        self.map_origin_y = -self.map_height * self.map_resolution / 2.0

        # Initialize pose tracking
        self.current_pose = np.eye(4)  # 4x4 identity matrix (position and orientation)
        self.previous_features = None
        self.previous_image = None
        self.landmarks = {}  # Dictionary to store landmarks
        self.landmark_counter = 0

        # Initialize tracking variables
        self.frame_count = 0
        self.keyframe_count = 0
        self.loop_closure_detected = False

        self.get_logger().info('SLAM components initialized')

    def vslam_callback(self, image_msg, camera_info_msg):
        """
        Process synchronized camera image and camera info for VSLAM
        """
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

            # Get camera parameters from camera_info
            fx = camera_info_msg.k[0]  # Focal length x
            fy = camera_info_msg.k[4]  # Focal length y
            cx = camera_info_msg.k[2]  # Principal point x
            cy = camera_info_msg.k[5]  # Principal point y
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # Process the image for SLAM
            self.process_slam_frame(cv_image, camera_matrix)

            # Update frame count
            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f'Error processing VSLAM frame: {str(e)}')

    def process_slam_frame(self, image, camera_matrix):
        """
        Process a single frame for SLAM: feature detection, tracking, mapping
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Detect features in current image
        current_keypoints, current_descriptors = self.feature_detector.detectAndCompute(gray_image, None)

        if current_keypoints is not None and len(current_keypoints) > 0:
            # Extract feature coordinates
            current_points = np.float32([kp.pt for kp in current_keypoints]).reshape(-1, 1, 2)

            # If we have previous features, try to match them
            if self.previous_features is not None and self.previous_image is not None:
                # Track features using optical flow
                prev_points = self.previous_features.reshape(-1, 1, 2)

                # Calculate optical flow
                next_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.previous_image, gray_image, prev_points, None
                )

                # Filter good matches
                good_new = next_points[status == 1]
                good_old = prev_points[status == 1]

                if len(good_new) >= 10:  # Need minimum number of matches
                    # Estimate camera motion using essential matrix
                    E, mask = cv2.findEssentialMat(
                        good_new, good_old, camera_matrix,
                        method=cv2.RANSAC, threshold=1.0
                    )

                    if E is not None:
                        # Recover pose from essential matrix
                        _, R, t, _ = cv2.recoverPose(E, good_new, good_old, camera_matrix)

                        # Update current pose
                        pose_increment = np.eye(4)
                        pose_increment[:3, :3] = R
                        pose_increment[:3, 3] = t.flatten()

                        # Apply the transformation to current pose
                        self.current_pose = self.current_pose @ pose_increment

                        # Update landmarks if enabled
                        if self.enable_mapping:
                            self.update_landmarks(good_old, good_new, camera_matrix)

                        # Check for loop closure if enabled
                        if self.enable_loop_closure:
                            self.check_loop_closure(good_new)

            # Store current features for next iteration
            self.previous_features = current_points
            self.previous_image = gray_image

    def update_landmarks(self, old_points, new_points, camera_matrix):
        """
        Update landmark positions based on feature tracking
        """
        # In a real implementation, this would triangulate 3D positions of landmarks
        # For this simulation, we'll just keep track of feature persistence

        # Update landmark tracking counts
        for i in range(min(len(old_points), len(new_points))):
            # In a real system, we would triangulate the 3D position
            # For now, we'll just track feature persistence
            pass

    def check_loop_closure(self, current_features):
        """
        Check for loop closure by comparing current features with historical data
        """
        # In a real implementation, this would compare current features with
        # features from previous locations to detect if the robot has returned
        # to a previously visited location
        pass

    def create_occupancy_grid(self):
        """
        Create occupancy grid map from SLAM data
        """
        # In a real implementation, this would convert landmark data to occupancy grid
        # For this example, we'll return a simple map with current robot position

        # Create header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame

        # Create occupancy grid message
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header = header
        occupancy_grid.info.resolution = self.map_resolution
        occupancy_grid.info.width = self.map_width
        occupancy_grid.info.height = self.map_height
        occupancy_grid.info.origin.position.x = self.map_origin_x
        occupancy_grid.info.origin.position.y = self.map_origin_y
        occupancy_grid.info.origin.position.z = 0.0
        occupancy_grid.info.origin.orientation.x = 0.0
        occupancy_grid.info.origin.orientation.y = 0.0
        occupancy_grid.info.origin.orientation.z = 0.0
        occupancy_grid.info.origin.orientation.w = 1.0

        # Fill map data (in a real implementation, this would come from SLAM)
        # For now, set some cells as occupied around the robot position
        occupancy_grid.data = self.map_data.flatten().tolist()

        return occupancy_grid

    def publish_results(self):
        """
        Publish SLAM results including map, pose, and transforms
        """
        try:
            # Publish occupancy grid if mapping is enabled
            if self.enable_mapping:
                occupancy_grid = self.create_occupancy_grid()
                self.map_pub.publish(occupancy_grid)

            # Publish current pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.map_frame
            pose_msg.pose.position.x = self.current_pose[0, 3]
            pose_msg.pose.position.y = self.current_pose[1, 3]
            pose_msg.pose.position.z = self.current_pose[2, 3]

            # Convert rotation matrix to quaternion
            rotation_matrix = self.current_pose[:3, :3]
            qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation_matrix)
            pose_msg.pose.orientation.w = qw
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz

            self.pose_pub.publish(pose_msg)

            # Publish odometry
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = self.map_frame
            odom_msg.child_frame_id = self.base_frame
            odom_msg.pose.pose = pose_msg.pose

            # Set velocity to zero (would come from motion model in real implementation)
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0
            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0

            self.odom_pub.publish(odom_msg)

            # Broadcast transform from map to odom, and odom to base_link
            self.broadcast_transforms(pose_msg.pose)

            # Publish visualization if enabled
            if self.enable_visualization:
                self.publish_visualization()

        except Exception as e:
            self.get_logger().error(f'Error publishing VSLAM results: {str(e)}')

    def broadcast_transforms(self, pose):
        """
        Broadcast TF transforms for the SLAM system
        """
        # Transform from map to odom (SLAM provides map->base_link, but we need intermediate odom)
        # In a real system, this would come from odometry source
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = self.get_clock().now().to_msg()
        map_to_odom.header.frame_id = self.map_frame
        map_to_odom.child_frame_id = self.odom_frame
        map_to_odom.transform.translation.x = 0.0
        map_to_odom.transform.translation.y = 0.0
        map_to_odom.transform.translation.z = 0.0
        map_to_odom.transform.rotation.w = 1.0
        map_to_odom.transform.rotation.x = 0.0
        map_to_odom.transform.rotation.y = 0.0
        map_to_odom.transform.rotation.z = 0.0

        # Transform from odom to base_link (this comes from SLAM pose)
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = self.get_clock().now().to_msg()
        odom_to_base.header.frame_id = self.odom_frame
        odom_to_base.child_frame_id = self.base_frame
        odom_to_base.transform.translation.x = pose.position.x
        odom_to_base.transform.translation.y = pose.position.y
        odom_to_base.transform.translation.z = pose.position.z
        odom_to_base.transform.rotation = pose.orientation

        # Publish transforms
        self.tf_broadcaster.sendTransform(map_to_odom)
        self.tf_broadcaster.sendTransform(odom_to_base)

    def publish_visualization(self):
        """
        Publish visualization data for debugging and monitoring
        """
        # Create a simple point cloud for visualization
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame

        # Define point fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Create some visualization points (robot path and landmarks)
        points = []

        # Add current robot position
        points.append([self.current_pose[0, 3], self.current_pose[1, 3], self.current_pose[2, 3]])

        # Create point cloud
        pointcloud_msg = point_cloud2.create_cloud(header, fields, points)
        self.visualization_pub.publish(pointcloud_msg)

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a 3x3 rotation matrix to quaternion
        """
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        return qw, qx, qy, qz


def main(args=None):
    """
    Main function to run the VSLAM node
    """
    rclpy.init(args=args)

    vslam_node = VSLAMNode()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()