# Isaac ROS Perception Pipeline Performance Validation

## Overview

This document outlines the validation procedures to ensure that the Isaac ROS perception pipeline meets the real-time performance target of 30 FPS. The validation process includes measuring processing rates, latency, and resource utilization across all perception components.

## Performance Targets

### Primary Targets
- **Processing Rate**: 30+ FPS for camera processing pipeline
- **End-to-End Latency**: <50ms from sensor input to perception output
- **Component Latency**: Each perception node <20ms processing time
- **Resource Utilization**: GPU utilization <90% for stable operation

### Secondary Targets
- **Memory Usage**: Stable memory consumption without leaks
- **CPU Utilization**: <80% average CPU usage per core
- **ROS Topic Rates**: All topics maintain configured publishing rates
- **Synchronization Accuracy**: Multi-sensor data synchronized within 10ms

## Validation Environment

### Hardware Requirements
- **GPU**: NVIDIA RTX 3080 or better (for 30+ FPS target)
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB or more
- **Storage**: SSD with 50GB+ free space

### Software Environment
- **ROS 2 Humble Hawksbill**: With Isaac ROS packages
- **CUDA**: Version 11.8 or higher
- **TensorRT**: Compatible with Isaac ROS
- **Isaac Sim**: For simulation-based validation

## Validation Procedures

### 1. Processing Rate Measurement

#### Camera Processing Rate
```bash
# Monitor camera image publishing rate
ros2 topic hz /camera/rgb/image_rect_color

# Expected: ~30 Hz for 30 FPS processing
# Acceptable range: 25-35 Hz (allowing for minor variations)
```

#### Detection Processing Rate
```bash
# Monitor object detection rate
ros2 topic hz /detectnet/detections

# Expected: ~30 Hz (or same as camera input rate)
# Acceptable range: 25-35 Hz
```

#### Point Cloud Processing Rate
```bash
# Monitor point cloud generation rate
ros2 topic hz /sensor_fusion/fused_pointcloud

# Expected: ~10-30 Hz depending on configuration
# Acceptable range: 80% of input rate
```

### 2. Latency Measurement

#### End-to-End Latency
```bash
# Use ROS 2 tools to measure latency between sensor input and perception output
ros2 run topic_tools relay /camera/rgb/image_raw /timestamped_input &
ros2 run topic_tools relay /detectnet/detections /timestamped_output &

# Calculate time difference between input and output timestamps
# Target: <50ms average latency
```

#### Component Latency
```bash
# Monitor individual node processing times using Isaac ROS tools
ros2 run isaac_ros_utilities node_profiler --node-name depth_processor
ros2 run isaac_ros_utilities node_profiler --node-name detectnet_node
ros2 run isaac_ros_utilities node_profiler --node-name sensor_fusion_node
```

### 3. Resource Utilization Monitoring

#### GPU Utilization
```bash
# Monitor GPU usage during perception processing
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1

# Target: <90% GPU utilization for stable operation
# Monitor for thermal throttling indicators
```

#### CPU Utilization
```bash
# Monitor CPU usage during processing
htop
# or
vmstat 1

# Target: <80% average utilization per core
# Monitor for sustained high usage indicating bottlenecks
```

#### Memory Usage
```bash
# Monitor memory usage patterns
free -h
# Monitor for memory leaks over time
watch -n 1 'pmap $(pgrep -f depth_processor) | tail -1'
```

## Performance Test Scenarios

### Scenario 1: Static Environment
- **Description**: Robot stationary with static objects in view
- **Expected Performance**: Consistent 30+ FPS processing
- **Validation Criteria**: Stable processing rate with minimal variation

### Scenario 2: Dynamic Environment
- **Description**: Moving objects and changing scene complexity
- **Expected Performance**: 25+ FPS with adaptive processing
- **Validation Criteria**: Performance degradation <20% from static case

### Scenario 3: High-Complexity Scene
- **Description**: Dense object detection with complex backgrounds
- **Expected Performance**: 20+ FPS with maintained accuracy
- **Validation Criteria**: Processing rate >20 FPS with acceptable detection quality

### Scenario 4: Multi-Sensor Fusion
- **Description**: Simultaneous processing of RGB, depth, LiDAR, and IMU
- **Expected Performance**: 25+ FPS with synchronized fusion
- **Validation Criteria**: All sensor data processed with <50ms latency

## Validation Scripts

### Performance Monitoring Script
```python
#!/usr/bin/env python3
# performance_monitor.py
"""
Performance monitoring script for Isaac ROS perception pipeline
Measures FPS, latency, and resource utilization
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import time
from collections import deque
import psutil
import GPUtil

class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')

        # Parameters
        self.declare_parameter('monitor_topic', '/camera/rgb/image_rect_color')
        self.declare_parameter('window_size', 30)  # Number of frames for FPS calculation

        self.monitor_topic = self.get_parameter('monitor_topic').value
        self.window_size = self.get_parameter('window_size').value

        # FPS calculation
        self.frame_times = deque(maxlen=self.window_size)
        self.fps = 0.0

        # Resource monitoring
        self.gpu_monitoring = True
        try:
            GPUtil.getGPUs()
        except:
            self.gpu_monitoring = False

        # Subscribe to monitoring topic
        self.subscription = self.create_subscription(
            Image,
            self.monitor_topic,
            self.image_callback,
            10
        )

        # Performance reporting timer
        self.timer = self.create_timer(1.0, self.report_performance)

        self.get_logger().info(f'Performance monitor started on topic: {self.monitor_topic}')

    def image_callback(self, msg):
        current_time = time.time()
        self.frame_times.append(current_time)

        # Calculate FPS if we have enough frames
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps = (len(self.frame_times) - 1) / time_diff

    def report_performance(self):
        # Calculate FPS
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps = (len(self.frame_times) - 1) / time_diff
        else:
            self.fps = 0.0

        # Get system resources
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent

        # Get GPU info if available
        gpu_info = ""
        if self.gpu_monitoring:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    gpu_info = f", GPU: {gpu.load*100:.1f}% utilization, {gpu.memoryUtil*100:.1f}% memory"
            except:
                gpu_info = ", GPU: monitoring unavailable"

        # Log performance metrics
        self.get_logger().info(
            f'Performance: FPS={self.fps:.2f}, CPU={cpu_percent:.1f}%, '
            f'Memory={memory_percent:.1f}%{gpu_info}'
        )

        # Check performance targets
        if self.fps < 30.0:
            self.get_logger().warn(f'WARNING: FPS below target (30): {self.fps:.2f}')
        else:
            self.get_logger().info(f'OK: FPS meets target: {self.fps:.2f}')

def main(args=None):
    rclpy.init(args=args)
    performance_monitor = PerformanceMonitor()

    try:
        rclpy.spin(performance_monitor)
    except KeyboardInterrupt:
        pass
    finally:
        performance_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Automated Performance Test Script
```bash
#!/bin/bash
# performance_test.sh
# Automated performance validation script

echo "Starting Isaac ROS Perception Pipeline Performance Validation..."

# Function to check if ROS 2 is running
check_ros2() {
    if ! pgrep -f "ros2" > /dev/null; then
        echo "ERROR: ROS 2 processes not found. Please start Isaac ROS perception pipeline first."
        exit 1
    fi
}

# Function to measure FPS
measure_fps() {
    local topic=$1
    local duration=${2:-10}  # Default 10 seconds
    echo "Measuring FPS for topic: $topic over ${duration}s..."

    # Use ros2 topic hz to measure frequency
    timeout $duration ros2 topic hz $topic 2>/dev/null | tail -1 | grep -o "[0-9.]*" || echo "0"
}

# Function to monitor system resources
monitor_resources() {
    echo "Monitoring system resources..."
    echo "CPU Utilization:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1"%"}'

    echo "Memory Usage:"
    free -h | awk 'NR==2{printf "%s/%s (%.2f%%)\n", $3,$2,$3*100/$2}'

    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Utilization:"
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits
    fi
}

# Main validation process
check_ros2

echo "Validating perception pipeline performance..."
echo "Target: 30+ FPS for camera processing"

# Measure key topic rates
CAMERA_FPS=$(measure_fps "/camera/rgb/image_rect_color" 10)
DETECTION_FPS=$(measure_fps "/detectnet/detections" 10)
POINTCLOUD_FPS=$(measure_fps "/sensor_fusion/fused_pointcloud" 10)

echo "Results:"
echo "Camera Processing Rate: $CAMERA_FPS FPS"
echo "Detection Processing Rate: $DETECTION_FPS FPS"
echo "Point Cloud Rate: $POINTCLOUD_FPS FPS"

# Validate against targets
echo ""
echo "Validation Results:"
if (( $(echo "$CAMERA_FPS >= 30" | bc -l) )); then
    echo "✅ Camera processing: PASS (Target: 30+, Actual: $CAMERA_FPS)"
else
    echo "❌ Camera processing: FAIL (Target: 30+, Actual: $CAMERA_FPS)"
fi

if (( $(echo "$DETECTION_FPS >= 25" | bc -l) )); then
    echo "✅ Detection processing: PASS (Target: 25+, Actual: $DETECTION_FPS)"
else
    echo "❌ Detection processing: FAIL (Target: 25+, Actual: $DETECTION_FPS)"
fi

# System resource check
echo ""
monitor_resources

echo ""
echo "Performance validation complete."
echo "Note: For actual validation, run this script while the Isaac ROS perception pipeline is actively processing data."
```

## Performance Optimization Guidelines

### If FPS is Below Target
1. **Reduce Rendering Quality**: Lower resolution or switch to rasterized mode
2. **Simplify Scene**: Reduce number of objects/lighting complexity
3. **Adjust Physics**: Increase time step (may affect stability)
4. **Sensor Optimization**: Reduce sensor update rates if not critical
5. **GPU Memory**: Check for and resolve GPU memory issues

### Memory Optimization
- Use memory pools and caching
- Implement proper cleanup of temporary data
- Monitor for memory leaks over extended runs
- Use pinned memory for transfers where appropriate

### Processing Optimization
- Enable multithreading for parallel processing
- Use hardware acceleration (GPU/TensorRT) where available
- Optimize queue depths for different sensor types
- Implement adaptive processing rates based on system load

## Validation Report Template

### Performance Validation Report
- **Test Environment**: [System specifications]
- **Test Date**: [Date of testing]
- **Pipeline Configuration**: [Active configuration]
- **Test Scenarios**: [List of scenarios tested]
- **Results Summary**: [Key performance metrics]
- **Pass/Fail Status**: [Overall validation status]
- **Recommendations**: [Any optimization recommendations]

## Expected Outcomes

With the current configuration and appropriate hardware, the Isaac ROS perception pipeline should achieve:
- **Minimum**: 30 FPS sustained camera processing
- **Target**: 30-60 FPS for optimal performance
- **Acceptable**: 25-30 FPS for basic functionality (with reduced complexity)

## Next Steps for Validation

1. Deploy perception pipeline configuration to Isaac Sim environment
2. Run performance validation scripts
3. Document actual performance achieved
4. Adjust configuration if needed to meet 30 FPS target
5. Update this report with actual performance metrics

## Performance Troubleshooting

### Common Performance Issues
- **Low FPS**: Check GPU memory usage and availability
- **High Latency**: Verify QoS settings and network configuration
- **Memory Leaks**: Monitor memory usage over time
- **CPU Bottlenecks**: Check for single-threaded processing blocks

### Performance Monitoring Tools
- ROS 2 built-in tools: `ros2 topic hz`, `ros2 topic delay`
- System monitoring: `htop`, `nvidia-smi`, `iotop`
- Isaac ROS utilities: Node profilers and performance monitors