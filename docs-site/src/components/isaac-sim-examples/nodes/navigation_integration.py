#!/usr/bin/env python3
# navigation_integration.py

"""
Isaac ROS Navigation Integration Node
This node integrates perception, VSLAM, and navigation systems for the humanoid robot.
It combines sensor data from perception pipeline with VSLAM localization for enhanced navigation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.duration import Duration

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster, TransformListener, Buffer

import message_filters
from message_filters import ApproximateTimeSynchronizer
import tf2_ros
import tf2_geometry_msgs


class NavigationIntegrationNode(Node):
    """
    Isaac ROS Navigation Integration Node
    Integrates perception, VSLAM, and navigation systems for humanoid robot
    """

    def __init__(self):
        super().__init__('navigation_integration_node')

        # Declare parameters
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('integration_frequency', 10.0)  # Hz
        self.declare_parameter('perception_timeout', 1.0)  # seconds
        self.declare_parameter('vslam_timeout', 1.0)  # seconds
        self.declare_parameter('min_detection_confidence', 0.7)
        self.declare_parameter('detection_merge_distance', 0.5)  # meters
        self.declare_parameter('dynamic_obstacle_buffer', 0.3)  # meters
        self.declare_parameter('prediction_horizon', 2.0)  # seconds
        self.declare_parameter('max_detection_range', 5.0)  # meters
        self.declare_parameter('min_detection_size', 0.1)  # meters
        self.declare_parameter('enable_dynamic_filtering', True)
        self.declare_parameter('enable_perception_fusion', True)
        self.declare_parameter('enable_vslam_fusion', True)
        self.declare_parameter('publish_visualization', True)
        self.declare_parameter('visualization_topic', 'navigation_integration/visualization')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.integration_frequency = self.get_parameter('integration_frequency').value
        self.perception_timeout = self.get_parameter('perception_timeout').value
        self.vslam_timeout = self.get_parameter('vslam_timeout').value
        self.min_detection_confidence = self.get_parameter('min_detection_confidence').value
        self.detection_merge_distance = self.get_parameter('detection_merge_distance').value
        self.dynamic_obstacle_buffer = self.get_parameter('dynamic_obstacle_buffer').value
        self.prediction_horizon = self.get_parameter('prediction_horizon').value
        self.max_detection_range = self.get_parameter('max_detection_range').value
        self.min_detection_size = self.get_parameter('min_detection_size').value
        self.enable_dynamic_filtering = self.get_parameter('enable_dynamic_filtering').value
        self.enable_perception_fusion = self.get_parameter('enable_perception_fusion').value
        self.enable_vslam_fusion = self.get_parameter('enable_vslam_fusion').value
        self.publish_visualization = self.get_parameter('publish_visualization').value
        self.visualization_topic = self.get_parameter('visualization_topic').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value

        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()

        # TF buffer and listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create QoS profile for sensor data
        self.qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Initialize data storage
        self.perception_detections = []
        self.vslam_pose = None
        self.vslam_map = None
        self.odom_pose = None
        self.last_perception_time = self.get_clock().now()
        self.last_vslam_time = self.get_clock().now()
        self.integrated_costmap = None
        self.integrated_path = None

        # Create publishers
        self.integrated_costmap_pub = self.create_publisher(
            OccupancyGrid,
            'navigation_integration/costmap',
            self.qos_profile
        )

        self.integrated_path_pub = self.create_publisher(
            Path,
            'navigation_integration/path',
            self.qos_profile
        )

        if self.publish_visualization:
            self.visualization_pub = self.create_publisher(
                MarkerArray,
                self.visualization_topic,
                self.qos_profile
            )

        # Create subscribers
        # Perception detections (from detectnet)
        self.detection_sub = self.create_subscription(
            PointCloud2,
            '/detectnet/spatial_detections',
            self.detection_callback,
            self.qos_profile
        )

        # VSLAM pose and map
        self.vslam_pose_sub = self.create_subscription(
            PoseStamped,
            '/vslam/pose',
            self.vslam_pose_callback,
            self.qos_profile
        )

        self.vslam_map_sub = self.create_subscription(
            OccupancyGrid,
            '/vslam/occupancy_grid',
            self.vslam_map_callback,
            self.qos_profile
        )

        # Odometry (for motion prediction)
        self.odom_sub = self.create_subscription(
            PointCloud2,  # Using point cloud for velocity data from perception
            '/odometry/filtered',
            self.odom_callback,
            self.qos_profile
        )

        # Timer for integration processing
        self.integration_timer = self.create_timer(
            1.0 / self.integration_frequency,
            self.integration_callback
        )

        self.get_logger().info('Isaac ROS Navigation Integration Node initialized')
        self.get_logger().info(f'Parameters: integration_frequency={self.integration_frequency}, detection_merge_distance={self.detection_merge_distance}')

    def detection_callback(self, msg):
        """
        Process perception detections from detectnet
        """
        try:
            # Update timestamp
            self.last_perception_time = self.get_clock().now()

            # Parse detection data from point cloud
            detections = []
            for point in point_cloud2.read_points(msg, field_names=("x", "y", "z", "confidence"), skip_nans=True):
                x, y, z, confidence = point
                if confidence >= self.min_detection_confidence:
                    detection = {
                        'position': np.array([x, y, z]),
                        'confidence': confidence,
                        'timestamp': msg.header.stamp,
                        'frame_id': msg.header.frame_id
                    }
                    detections.append(detection)

            # Store detections
            self.perception_detections = detections

            # Log detection count
            self.get_logger().debug(f'Received {len(detections)} perception detections')

        except Exception as e:
            self.get_logger().error(f'Error processing perception detections: {str(e)}')

    def vslam_pose_callback(self, msg):
        """
        Process VSLAM pose estimates
        """
        try:
            # Update timestamp
            self.last_vslam_time = self.get_clock().now()

            # Store VSLAM pose
            self.vslam_pose = msg.pose

            # Log pose update
            self.get_logger().debug(f'VSLAM pose updated: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

        except Exception as e:
            self.get_logger().error(f'Error processing VSLAM pose: {str(e)}')

    def vslam_map_callback(self, msg):
        """
        Process VSLAM occupancy grid map
        """
        try:
            # Update timestamp
            self.last_vslam_time = self.get_clock().now()

            # Store VSLAM map
            self.vslam_map = msg

            # Log map update
            self.get_logger().debug(f'VSLAM map updated: {msg.info.width}x{msg.info.height} grid')

        except Exception as e:
            self.get_logger().error(f'Error processing VSLAM map: {str(e)}')

    def odom_callback(self, msg):
        """
        Process odometry data for motion prediction
        """
        try:
            # Parse odometry data from point cloud
            # In a real implementation, this would come from proper odometry topic
            # For this example, we'll simulate velocity from the point cloud
            velocities = []
            for point in point_cloud2.read_points(msg, field_names=("x", "y", "z", "velocity"), skip_nans=True):
                x, y, z, velocity = point
                velocities.append(velocity)

            if velocities:
                avg_velocity = np.mean(velocities)
                self.odom_pose = {
                    'linear_velocity': avg_velocity,
                    'timestamp': msg.header.stamp
                }

        except Exception as e:
            self.get_logger().error(f'Error processing odometry: {str(e)}')

    def integration_callback(self):
        """
        Main integration callback that combines perception, VSLAM, and navigation data
        """
        try:
            # Check if we have recent data
            current_time = self.get_clock().now()
            perception_age = (current_time - self.last_perception_time).nanoseconds / 1e9
            vslam_age = (current_time - self.last_vslam_time).nanoseconds / 1e9

            # Process perception data if available and recent
            if self.perception_detections and perception_age < self.perception_timeout:
                integrated_detections = self.process_perception_detections()
            else:
                integrated_detections = []

            # Process VSLAM data if available and recent
            if self.vslam_map is not None and vslam_age < self.vslam_timeout:
                integrated_map = self.process_vslam_map()
            else:
                integrated_map = self.create_empty_map() if self.vslam_map else None

            # Fuse perception and VSLAM data
            if integrated_map is not None and self.enable_perception_fusion:
                fused_map = self.fuse_perception_vslam(integrated_map, integrated_detections)
            else:
                fused_map = integrated_map

            # Publish integrated costmap if available
            if fused_map is not None:
                self.integrated_costmap_pub.publish(fused_map)
                self.integrated_costmap = fused_map

            # Generate integrated path if needed
            integrated_path = self.generate_integrated_path(fused_map)
            if integrated_path is not None:
                self.integrated_path_pub.publish(integrated_path)
                self.integrated_path = integrated_path

            # Publish visualization if enabled
            if self.publish_visualization:
                visualization = self.create_visualization_markers(integrated_detections)
                if visualization:
                    self.visualization_pub.publish(visualization)

            # Log integration status
            self.get_logger().debug(f'Integration completed: {len(integrated_detections)} detections, map updated')

        except Exception as e:
            self.get_logger().error(f'Error in integration callback: {str(e)}')

    def process_perception_detections(self):
        """
        Process and filter perception detections
        """
        try:
            # Filter detections based on confidence and range
            filtered_detections = []
            for detection in self.perception_detections:
                # Check confidence threshold
                if detection['confidence'] < self.min_detection_confidence:
                    continue

                # Check range threshold
                distance = np.linalg.norm(detection['position'])
                if distance > self.max_detection_range:
                    continue

                # Check minimum detection size (simulated)
                if distance < self.min_detection_size:
                    continue

                # Transform to map frame if needed
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.map_frame,
                        detection['frame_id'],
                        detection['timestamp'],
                        timeout=Duration(seconds=0.1)
                    )
                    # Apply transformation (simplified)
                    transformed_pos = detection['position']  # In a real system, apply full transform
                    detection['position_map'] = transformed_pos
                except Exception as e:
                    self.get_logger().warn(f'Could not transform detection: {str(e)}')
                    detection['position_map'] = detection['position']

                filtered_detections.append(detection)

            # Merge nearby detections
            merged_detections = self.merge_detections(filtered_detections)

            return merged_detections

        except Exception as e:
            self.get_logger().error(f'Error processing perception detections: {str(e)}')
            return []

    def process_vslam_map(self):
        """
        Process VSLAM occupancy grid map
        """
        try:
            # In a real implementation, this would process and validate the VSLAM map
            # For this example, we'll just return the original map
            return self.vslam_map

        except Exception as e:
            self.get_logger().error(f'Error processing VSLAM map: {str(e)}')
            return None

    def fuse_perception_vslam(self, vslam_map, detections):
        """
        Fuse perception detections with VSLAM map
        """
        try:
            # Create a copy of the VSLAM map to modify
            fused_map = OccupancyGrid()
            fused_map.header = vslam_map.header
            fused_map.info = vslam_map.info
            fused_map.data = list(vslam_map.data)  # Copy data

            # Convert detections to map coordinates and update map
            for detection in detections:
                # Calculate map coordinates
                pos = detection['position_map']
                map_x = int((pos[0] - fused_map.info.origin.position.x) / fused_map.info.resolution)
                map_y = int((pos[1] - fused_map.info.origin.position.y) / fused_map.info.resolution)

                # Check bounds
                if 0 <= map_x < fused_map.info.width and 0 <= map_y < fused_map.info.height:
                    # Calculate index
                    idx = map_y * fused_map.info.width + map_x

                    # Update cell based on detection confidence and type
                    if idx < len(fused_map.data):
                        # Increase cost based on detection confidence
                        base_cost = fused_map.data[idx]
                        detection_cost = min(100, int(detection['confidence'] * 100))
                        fused_map.data[idx] = max(base_cost, detection_cost)

            return fused_map

        except Exception as e:
            self.get_logger().error(f'Error fusing perception and VSLAM: {str(e)}')
            return vslam_map  # Return original if fusion fails

    def merge_detections(self, detections):
        """
        Merge nearby detections to reduce redundancy
        """
        try:
            if not detections:
                return []

            merged = []
            used = set()

            for i, det1 in enumerate(detections):
                if i in used:
                    continue

                # Find nearby detections to merge
                cluster = [det1]
                cluster_indices = [i]

                for j, det2 in enumerate(detections[i+1:], i+1):
                    if j in used:
                        continue

                    # Calculate distance between detections
                    dist = np.linalg.norm(det1['position'] - det2['position'])
                    if dist <= self.detection_merge_distance:
                        cluster.append(det2)
                        cluster_indices.append(j)

                # Mark indices as used
                used.update(cluster_indices)

                # Create merged detection (average position, max confidence)
                if len(cluster) == 1:
                    merged.append(det1)
                else:
                    # Calculate average position
                    avg_pos = np.mean([d['position'] for d in cluster], axis=0)
                    # Use max confidence
                    max_conf = max(d['confidence'] for d in cluster)
                    # Use most recent timestamp
                    latest_ts = max((d['timestamp'] for d in cluster), key=lambda x: x.sec + x.nanosec/1e9)

                    merged_detection = {
                        'position': avg_pos,
                        'confidence': max_conf,
                        'timestamp': latest_ts,
                        'frame_id': cluster[0]['frame_id'],
                        'position_map': avg_pos  # Simplified
                    }
                    merged.append(merged_detection)

            return merged

        except Exception as e:
            self.get_logger().error(f'Error merging detections: {str(e)}')
            return detections

    def generate_integrated_path(self, costmap):
        """
        Generate integrated path based on fused costmap
        """
        try:
            if costmap is None:
                return None

            # In a real implementation, this would perform path planning
            # For this example, we'll return an empty path
            path = Path()
            path.header.frame_id = self.map_frame
            path.header.stamp = self.get_clock().now().to_msg()

            # This is a placeholder - in a real system, this would call a path planner
            # that takes into account the integrated costmap

            return path

        except Exception as e:
            self.get_logger().error(f'Error generating integrated path: {str(e)}')
            return None

    def create_empty_map(self):
        """
        Create an empty occupancy grid map
        """
        try:
            # Create a basic empty map structure
            empty_map = OccupancyGrid()
            empty_map.header.frame_id = self.map_frame
            empty_map.header.stamp = self.get_clock().now().to_msg()
            empty_map.info.resolution = 0.05  # 5cm resolution
            empty_map.info.width = 100
            empty_map.info.height = 100
            empty_map.info.origin.position.x = -2.5
            empty_map.info.origin.position.y = -2.5
            empty_map.info.origin.position.z = 0.0
            empty_map.info.origin.orientation.x = 0.0
            empty_map.info.origin.orientation.y = 0.0
            empty_map.info.origin.orientation.z = 0.0
            empty_map.info.origin.orientation.w = 1.0

            # Initialize with unknown (-1) values
            total_cells = empty_map.info.width * empty_map.info.height
            empty_map.data = [-1] * total_cells

            return empty_map

        except Exception as e:
            self.get_logger().error(f'Error creating empty map: {str(e)}')
            return None

    def create_visualization_markers(self, detections):
        """
        Create visualization markers for detections and integration results
        """
        try:
            marker_array = MarkerArray()

            # Create markers for detections
            for i, detection in enumerate(detections):
                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "detections"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Position
                marker.pose.position.x = detection['position_map'][0]
                marker.pose.position.y = detection['position_map'][1]
                marker.pose.position.z = detection['position_map'][2] if len(detection['position_map']) > 2 else 0.0
                marker.pose.orientation.w = 1.0

                # Scale (radius)
                marker.scale.x = 0.2  # 20cm radius
                marker.scale.y = 0.2
                marker.scale.z = 0.2

                # Color based on confidence
                confidence = detection['confidence']
                marker.color.r = 1.0
                marker.color.g = 1.0 - confidence
                marker.color.b = 1.0 - confidence
                marker.color.a = 0.8

                marker_array.markers.append(marker)

            # Create a marker for the robot position
            if self.vslam_pose is not None:
                robot_marker = Marker()
                robot_marker.header.frame_id = self.map_frame
                robot_marker.header.stamp = self.get_clock().now().to_msg()
                robot_marker.ns = "robot"
                robot_marker.id = 999  # Robot marker ID
                robot_marker.type = Marker.CYLINDER
                robot_marker.action = Marker.ADD

                robot_marker.pose.position.x = self.vslam_pose.position.x
                robot_marker.pose.position.y = self.vslam_pose.position.y
                robot_marker.pose.position.z = 0.0
                robot_marker.pose.orientation.w = 1.0

                robot_marker.scale.x = 0.6  # 60cm diameter
                robot_marker.scale.y = 0.6
                robot_marker.scale.z = 0.8  # 80cm height

                robot_marker.color.r = 0.0
                robot_marker.color.g = 0.0
                robot_marker.color.b = 1.0
                robot_marker.color.a = 0.8

                marker_array.markers.append(robot_marker)

            return marker_array

        except Exception as e:
            self.get_logger().error(f'Error creating visualization markers: {str(e)}')
            return None


def main(args=None):
    """
    Main function to run the navigation integration node
    """
    rclpy.init(args=args)

    navigation_integration_node = NavigationIntegrationNode()

    try:
        rclpy.spin(navigation_integration_node)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_integration_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()