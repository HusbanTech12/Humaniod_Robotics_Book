# Isaac ROS Perception Pipelines

## Overview

The Isaac ROS Perception Pipeline is a hardware-accelerated system that processes multimodal sensor data in real-time to enable environment understanding for the humanoid robot. This pipeline integrates multiple Isaac ROS components to provide object detection, sensor fusion, depth processing, and spatial awareness capabilities.

## Architecture

The perception pipeline follows a modular architecture that allows for efficient processing of sensor data:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sensor Data   │    │  Processing     │    │  Output         │
│                 │───►│  Nodes          │───►│  Topics         │
│ • RGB Camera    │    │ • Image         │    │ • Detections    │
│ • Depth Camera  │    │   Rectification │    │ • Point Cloud   │
│ • LiDAR         │    │ • Object        │    │ • Spatial       │
│ • IMU           │    │   Detection     │    │   Detections    │
│                 │    │ • Sensor Fusion │    │ • Transforms    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components

### 1. Image Processing Pipeline

The image processing pipeline handles camera data rectification and preprocessing:

- **Image Rectification**: Corrects camera distortion using calibration parameters
- **Color Processing**: Converts raw camera images to rectified color images
- **Resolution Management**: Handles different input/output resolutions

#### Configuration
The image processing pipeline is configured in `config/image_processing.yaml` with parameters for:
- Output width and height
- Scaling factors for image resizing
- Distortion model parameters

### 2. Object Detection Pipeline

The object detection pipeline uses deep learning models to identify and classify objects in the environment:

- **DetectNet**: NVIDIA's optimized object detection network
- **Model**: SSD MobileNet v2 trained on COCO dataset
- **Confidence Threshold**: Configurable minimum confidence for detections
- **Spatial Detection**: Combines object detection with depth information

#### Parameters
- Model name: `ssd_mobilenet_v2_coco`
- Confidence threshold: 0.7 (configurable)
- Maximum objects: 10
- Input resolution: 300x300 for model processing

### 3. Sensor Fusion Pipeline

The sensor fusion pipeline combines data from multiple sensors to create a comprehensive understanding of the environment:

- **Camera-LiDAR Fusion**: Projects camera detections onto LiDAR point clouds
- **IMU Integration**: Incorporates inertial measurements for motion compensation
- **Multi-Modal Processing**: Combines RGB, depth, and LiDAR data

#### Fusion Methods
- Probabilistic fusion for uncertainty management
- Timestamp synchronization for multi-sensor data
- Data association for tracking objects across sensors

### 4. Depth Processing Pipeline

The depth processing pipeline handles depth information from stereo cameras or depth sensors:

- **Depth Conversion**: Converts raw depth images to metric units
- **Point Cloud Generation**: Creates 3D point clouds from RGB-D data
- **Depth Refinement**: Filters and interpolates depth data

## Launch Configuration

The perception pipeline is launched using the configuration in `launch/perception_launch_config.py` which includes:

### Composable Node Container
The pipeline uses a single container for efficient processing:
- **perception_container**: Contains all perception nodes for low-latency processing
- **vslam_container**: Separate container for Visual SLAM processing

### Node Configuration

#### Image Rectification Node
```yaml
- Package: `isaac_ros_image_proc`
- Plugin: `isaac_ros::ImageProc::RectifyNode`
- Parameters:
  - output_width: 640
  - output_height: 480
  - scale_to_fit_image: true
- Remappings:
  - image → camera/rgb/image_raw
  - camera_info → camera/rgb/camera_info
  - image_rect → camera/rgb/image_rect_color
```

#### Depth Image Processing Node
```yaml
- Package: `isaac_ros_depth_image_proc`
- Plugin: `nvidia::isaac_ros::depth_image_proc::ConvertMetricNode`
- Remappings:
  - image_raw → camera/depth/image_raw
  - image → camera/depth/image_metric
```

#### Point Cloud Creation Node
```yaml
- Package: `isaac_ros_pointcloud_utils`
- Plugin: `nvidia::isaac_ros::pointcloud_utils::PointCloudPclNode`
- Parameters:
  - use_color: true
  - fill_nan: false
  - min_height: 0.0
  - max_height: 3.0
- Remappings:
  - image → camera/rgb/image_rect_color
  - depth → camera/depth/image_metric
  - points → camera/depth/color/points
```

#### Object Detection Node
```yaml
- Package: `isaac_ros_detectnet`
- Plugin: `nvidia::isaac_ros::detectnet::DetectNetNode`
- Parameters:
  - model_name: ssd_mobilenet_v2_coco
  - confidence_threshold: 0.7
  - max_objects: 10
- Remappings:
  - image_input → camera/rgb/image_rect_color
  - detections_output → detectnet/detections
```

#### Spatial Detection Node
```yaml
- Package: `isaac_ros_detectnet`
- Plugin: `nvidia::isaac_ros::detectnet::SpatialDetectionNode`
- Parameters:
  - max_object_points: 500
  - depth_unit: meters
- Remappings:
  - detections → detectnet/detections
  - depth_image → camera/depth/image_rect_raw
  - spatial_detections → detectnet/spatial_detections
```

#### Isaac ROS AprilTag Node
```yaml
- Package: `isaac_ros_apriltag`
- Plugin: `nvidia::isaac_ros::apriltag::AprilTagNode`
- Parameters:
  - family: tag36h11
  - max_hamming: 1
- Remappings:
  - image → camera/rgb/image_rect_color
  - camera_info → camera/rgb/camera_info
  - detections → apriltag/detections
```

#### Visual SLAM Node
```yaml
- Package: `isaac_ros_visual_slam`
- Plugin: `nvidia::isaac_ros::visual_slam::VisualSlamNode`
- Parameters:
  - enable_occupancy_map: true
  - enable_mapper: true
  - enable_localization: true
  - enable_loop_closure: true
- Remappings:
  - visual_slam/imu → imu/data
  - visual_slam/left/image → camera/rgb/image_rect_color
```

## Configuration Files

### Object Detection Configuration
Located at `config/object_detection.yaml`, this file contains:
- Model parameters and paths
- Detection thresholds and limits
- Performance optimization settings
- Hardware acceleration settings

### Sensor Fusion Configuration
Located at `config/sensor_fusion.yaml`, this file contains:
- Fusion method parameters
- Sensor calibration data
- Multi-sensor synchronization settings
- Quality of service configurations

### Camera Calibration
Located at `config/camera_calibration.yaml`, this file contains:
- Intrinsic camera parameters
- Distortion coefficients
- Rectification parameters

## Performance Optimization

### Real-time Performance
The perception pipeline is optimized for real-time operation:
- Target: 30 FPS for camera processing
- Efficient memory management with pinned memory
- GPU acceleration using TensorRT
- Multi-threaded processing with thread pools

### Resource Management
- Memory pools for efficient allocation
- CUDA graph execution for consistent performance
- Adaptive processing rates based on system load

### Quality of Service Settings
- Best-effort reliability for sensor data
- Appropriate queue depths for different sensor types
- Configurable history policies for message handling

## Integration with Other Systems

### Navigation Integration
The perception pipeline provides input to the navigation system:
- Object detections for obstacle avoidance
- Spatial information for path planning
- Environmental maps for localization

### Control Integration
- Sensor data for state estimation
- Environmental awareness for decision making
- Feedback for closed-loop control

## Launch Parameters

The perception pipeline supports various launch parameters:

- `namespace`: Namespace for perception nodes (default: humanoid_robot)
- `enable_rectification`: Enable camera image rectification (default: true)
- `enable_object_detection`: Enable object detection pipeline (default: true)
- `enable_sensor_fusion`: Enable sensor fusion pipeline (default: true)
- `enable_visual_slam`: Enable visual SLAM pipeline (default: true)

## Troubleshooting

### Common Issues

#### Low Processing Performance
- Check GPU memory usage and availability
- Verify TensorRT installation and compatibility
- Adjust processing parameters for system capabilities

#### Sensor Synchronization Issues
- Verify timestamp accuracy across sensors
- Check QoS settings for appropriate queue depths
- Adjust synchronization tolerance parameters

#### Detection Quality Problems
- Verify camera calibration parameters
- Check lighting conditions for adequate illumination
- Adjust confidence thresholds as needed

#### Memory Issues
- Monitor GPU memory usage during operation
- Reduce processing resolution if needed
- Check for memory leaks in long-running operations

### Performance Monitoring
- Monitor processing rates for each pipeline stage
- Check memory usage and allocation patterns
- Verify sensor data quality and consistency

## Best Practices

### Configuration Management
- Use appropriate confidence thresholds for your application
- Configure sensor fusion weights based on sensor reliability
- Set appropriate processing rates for your hardware

### Deployment Considerations
- Test with representative environmental conditions
- Validate sensor calibration before deployment
- Monitor performance under expected operating conditions

### Development Workflow
- Use simulation for initial development and testing
- Validate perception outputs before integrating with other systems
- Implement comprehensive logging for debugging

## Testing Procedures

### 1. Unit Testing

#### Image Processing Pipeline
- Test image rectification with various distortion models
- Verify color space conversions work correctly
- Validate resolution scaling maintains aspect ratio

#### Object Detection Pipeline
- Test detection accuracy with known objects
- Verify confidence threshold filtering works
- Validate spatial detection with depth information
- Test AprilTag detection and pose estimation

#### Sensor Fusion Pipeline
- Test timestamp synchronization accuracy
- Verify multi-sensor data association
- Validate fusion algorithms with known inputs
- Test edge case scenarios with missing sensor data

#### Depth Processing Pipeline
- Test depth filtering and hole filling
- Validate point cloud generation accuracy
- Verify depth range filtering works correctly

### 2. Integration Testing

#### End-to-End Pipeline Test
```bash
# Launch the complete perception pipeline
ros2 launch isaac_ros_workspace perception_launch_config.py

# Monitor all output topics
ros2 topic echo /detectnet/detections
ros2 topic echo /sensor_fusion/fused_data
ros2 topic echo /camera/rgb/image_rect_color
```

#### Multi-Sensor Synchronization Test
```bash
# Monitor timestamp synchronization
ros2 run isaac_ros_utilities timestamp_analyzer --input-topics /camera/rgb/image_raw /scan /imu/data

# Check for synchronization accuracy within 10ms
```

#### Performance Integration Test
```bash
# Measure end-to-end processing performance
ros2 run isaac_ros_utilities performance_analyzer --pipeline perception --duration 60

# Expected: 30+ FPS with <50ms latency
```

### 3. Performance Testing

#### FPS Measurement
```bash
# Monitor processing rates for all key topics
ros2 topic hz /camera/rgb/image_rect_color  # Should be ~30 Hz
ros2 topic hz /detectnet/detections         # Should match camera rate
ros2 topic hz /sensor_fusion/fused_data     # Should be appropriate rate
```

#### Resource Utilization Test
```bash
# Monitor system resources during operation
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1
htop  # Monitor CPU usage
```

#### Latency Measurement
```bash
# Measure processing latency
ros2 run isaac_ros_utilities latency_analyzer --input-topic /camera/rgb/image_raw --output-topic /detectnet/detections
# Expected: <50ms end-to-end latency
```

### 4. Functional Testing

#### Object Detection Accuracy Test
1. Place known objects in the robot's field of view
2. Verify detection accuracy and confidence scores
3. Test with various lighting conditions
4. Validate detection consistency over time

#### Sensor Fusion Quality Test
1. Move robot through environment with known landmarks
2. Verify fused sensor data provides consistent information
3. Test with individual sensors disabled to verify fallback behavior
4. Validate multi-sensor consistency

#### Depth Processing Quality Test
1. Test with objects at various distances
2. Verify depth accuracy and precision
3. Validate point cloud generation quality
4. Test hole filling in depth maps

### 5. Stress Testing

#### High-Load Scenario
- Increase object density in the environment
- Test with maximum expected processing load
- Monitor for performance degradation or failures

#### Long-Running Test
- Run perception pipeline continuously for extended period
- Monitor for memory leaks or performance degradation
- Validate consistent processing rates over time

#### Edge Case Testing
- Test with poor lighting conditions
- Test with reflective or transparent objects
- Test with occluded or partially visible objects
- Test with sensor data dropouts

### 6. Validation Criteria

#### Performance Criteria
- **Processing Rate**: ≥30 FPS for camera processing
- **Latency**: <50ms end-to-end processing
- **Resource Usage**: GPU utilization <90%
- **Synchronization**: Multi-sensor sync within 10ms

#### Functional Criteria
- **Detection Accuracy**: ≥90% for known objects under good conditions
- **False Positive Rate**: <5% in static environment
- **Sensor Fusion**: Consistent data association across sensors
- **Depth Accuracy**: <5cm error at 1m distance

#### Robustness Criteria
- **Stability**: No crashes during 1-hour continuous operation
- **Recovery**: Automatic recovery from sensor data dropouts
- **Graceful Degradation**: Continued operation with reduced sensors

### 7. Automated Testing Script

Create an automated test script to validate the perception pipeline:

```bash
#!/bin/bash
# perception_test.sh
# Automated Isaac ROS perception pipeline validation

set -e  # Exit on any error

echo "Starting Isaac ROS Perception Pipeline Validation..."

# Check if ROS 2 is running
if ! pgrep -f "ros2" > /dev/null; then
    echo "ERROR: ROS 2 processes not found"
    exit 1
fi

# Define test parameters
TEST_DURATION=60  # seconds
FPS_THRESHOLD=25  # minimum acceptable FPS
LATENCY_THRESHOLD=0.05  # 50ms in seconds

echo "Running performance tests for ${TEST_DURATION}s..."

# Test 1: FPS measurement
echo "Test 1: Measuring processing FPS..."
CAMERA_FPS=$(timeout $TEST_DURATION ros2 topic hz /camera/rgb/image_rect_color 2>/dev/null | tail -1 | grep -o "[0-9.]*" || echo "0")
echo "Camera processing FPS: $CAMERA_FPS"

if (( $(echo "$CAMERA_FPS >= $FPS_THRESHOLD" | bc -l) )); then
    echo "✅ FPS test: PASS"
else
    echo "❌ FPS test: FAIL (threshold: $FPS_THRESHOLD, actual: $CAMERA_FPS)"
fi

# Test 2: Check for output topics
echo "Test 2: Checking for perception outputs..."
TOPICS_TO_CHECK=(
    "/detectnet/detections"
    "/sensor_fusion/fused_data"
    "/camera/depth/color/points"
)

for topic in "${TOPICS_TO_CHECK[@]}"; do
    if ros2 topic list | grep -q "$topic"; then
        # Check if topic has recent messages
        if timeout 5 ros2 topic echo "$topic" --field data --field header --field detections --field points --field fused_data 2>/dev/null | head -n 5 | grep -q .; then
            echo "✅ $topic: Available and publishing"
        else
            echo "⚠️ $topic: Available but not publishing"
        fi
    else
        echo "❌ $topic: Not available"
    fi
done

# Test 3: Resource monitoring
echo "Test 3: Resource utilization check..."
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}' | cut -d' ' -f1)
MEMORY_USAGE=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')

echo "CPU Usage: ${CPU_USAGE}%"
echo "Memory Usage: ${MEMORY_USAGE}%"

if (( $(echo "$CPU_USAGE < 80" | bc -l) )); then
    echo "✅ CPU usage: PASS"
else
    echo "⚠️ CPU usage: HIGH (${CPU_USAGE}%)"
fi

if (( $(echo "$MEMORY_USAGE < 80" | bc -l) )); then
    echo "✅ Memory usage: PASS"
else
    echo "⚠️ Memory usage: HIGH (${MEMORY_USAGE}%)"
fi

echo "Perception pipeline validation complete."
echo "Note: For comprehensive testing, run additional functional tests with known objects and scenarios."
```

### 8. Test Environment Setup

#### Simulation Environment
- Use Isaac Sim with calibrated sensor models
- Create test scenes with known objects and layouts
- Simulate various lighting and environmental conditions

#### Hardware-in-the-Loop Testing
- Connect real sensors to the perception pipeline
- Test with physical objects in controlled environment
- Validate simulation-to-reality transfer

## Next Steps

After implementing the perception pipeline:

1. **Navigation Integration**: Connect perception outputs to the Nav2 navigation system
2. **Control Integration**: Use perception data for humanoid robot control decisions
3. **Validation**: Test the complete perception pipeline in various scenarios
4. **Optimization**: Fine-tune parameters for specific applications