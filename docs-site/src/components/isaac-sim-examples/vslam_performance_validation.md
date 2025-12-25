# VSLAM Performance Validation

## Overview

This document outlines the validation procedures to ensure that the Isaac ROS Visual SLAM pipeline meets the performance target of 10 Hz map update rate. The validation process includes measuring map publishing rates, processing performance, and system resource utilization.

## Performance Targets

### Primary Target
- **Map Update Rate**: 10+ Hz for occupancy grid updates
- **Pose Update Rate**: 30+ Hz for pose estimation
- **Feature Processing Rate**: 30+ Hz for feature detection and tracking

### Secondary Targets
- **Processing Latency**: <100ms from image input to map update
- **Resource Utilization**: GPU utilization <90% for stable operation
- **Memory Usage**: Stable memory consumption without leaks
- **CPU Utilization**: <80% average CPU usage per core

## Validation Environment

### Hardware Requirements
- **GPU**: NVIDIA RTX 3080 or better (for 10+ Hz map updates)
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB or more
- **Storage**: SSD with 50GB+ free space

### Software Environment
- **ROS 2 Humble Hawksbill**: With Isaac ROS packages
- **CUDA**: Version 11.8 or higher
- **TensorRT**: Compatible with Isaac ROS
- **Isaac Sim**: For simulation-based validation

## Validation Procedures

### 1. Map Update Rate Measurement

#### Occupancy Grid Publishing Rate
```bash
# Monitor occupancy grid publishing rate
ros2 topic hz /vslam/occupancy_grid

# Expected: ~10+ Hz for map updates
# Acceptable range: 10-30 Hz (allowing for variation based on processing load)
```

#### Pose Estimation Rate
```bash
# Monitor pose estimation rate
ros2 topic hz /vslam/pose

# Expected: ~30 Hz (or same as camera input rate)
# Acceptable range: 20+ Hz for pose updates
```

### 2. Processing Latency Measurement

#### End-to-End Latency
```bash
# Use ROS 2 tools to measure latency between image input and map output
ros2 run topic_tools relay /camera/rgb/image_rect_color /timestamped_input &
ros2 run topic_tools relay /vslam/occupancy_grid /timestamped_output &

# Calculate time difference between input and output timestamps
# Target: <100ms average latency for map updates
```

#### Component Latency
```bash
# Monitor individual node processing times using Isaac ROS tools
ros2 run isaac_ros_utilities node_profiler --node-name vslam_node
```

### 3. Resource Utilization Monitoring

#### GPU Utilization
```bash
# Monitor GPU usage during VSLAM processing
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
watch -n 1 'pmap $(pgrep -f vslam_node) | tail -1'
```

## Performance Test Scenarios

### Scenario 1: Static Environment
- **Description**: Robot stationary with static objects in view
- **Expected Performance**: Consistent 10+ Hz map updates
- **Validation Criteria**: Stable map update rate with minimal variation

### Scenario 2: Slow Movement
- **Description**: Robot moving slowly through environment
- **Expected Performance**: 10+ Hz with occasional drops during motion
- **Validation Criteria**: Average rate >10 Hz with brief dips acceptable

### Scenario 3: Fast Movement
- **Description**: Robot moving quickly through environment
- **Expected Performance**: Variable rate, but average >10 Hz
- **Validation Criteria**: Recovery to >10 Hz after motion stops

### Scenario 4: Feature-Rich Environment
- **Description**: Environment with many visual features
- **Expected Performance**: Consistent 10+ Hz map updates
- **Validation Criteria**: Processing rate maintains target with high feature count

### Scenario 5: Feature-Poor Environment
- **Description**: Environment with few visual features
- **Expected Performance**: Variable rate, but system remains stable
- **Validation Criteria**: System continues operation without crashes

## Validation Scripts

### Performance Monitoring Script
```python
#!/usr/bin/env python3
# vslam_performance_monitor.py
"""
Performance monitoring script for Isaac ROS VSLAM
Measures map update rate, latency, and resource utilization
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import time
from collections import deque
import psutil
import GPUtil

class VSLAMPerformanceMonitor(Node):
    def __init__(self):
        super().__init__('vslam_performance_monitor')

        # Parameters
        self.declare_parameter('monitor_topic', '/vslam/occupancy_grid')
        self.declare_parameter('window_size', 10)  # Number of maps for rate calculation

        self.monitor_topic = self.get_parameter('monitor_topic').value
        self.window_size = self.get_parameter('window_size').value

        # Rate calculation
        self.map_times = deque(maxlen=self.window_size)
        self.map_rate = 0.0

        # Resource monitoring
        self.gpu_monitoring = True
        try:
            GPUtil.getGPUs()
        except:
            self.gpu_monitoring = False

        # Subscribe to monitoring topic
        self.subscription = self.create_subscription(
            OccupancyGrid,
            self.monitor_topic,
            self.map_callback,
            10
        )

        # Performance reporting timer
        self.timer = self.create_timer(1.0, self.report_performance)

        self.get_logger().info(f'VSLAM performance monitor started on topic: {self.monitor_topic}')

    def map_callback(self, msg):
        current_time = time.time()
        self.map_times.append(current_time)

        # Calculate map rate if we have enough samples
        if len(self.map_times) >= 2:
            time_diff = self.map_times[-1] - self.map_times[0]
            if time_diff > 0:
                self.map_rate = (len(self.map_times) - 1) / time_diff

    def report_performance(self):
        # Calculate map rate
        if len(self.map_times) >= 2:
            time_diff = self.map_times[-1] - self.map_times[0]
            if time_diff > 0:
                self.map_rate = (len(self.map_times) - 1) / time_diff
        else:
            self.map_rate = 0.0

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
            f'VSLAM Performance: Map Rate={self.map_rate:.2f} Hz, CPU={cpu_percent:.1f}%, '
            f'Memory={memory_percent:.1f}%{gpu_info}'
        )

        # Check performance targets
        if self.map_rate >= 10.0:
            self.get_logger().info(f'✅ Map update rate meets target: {self.map_rate:.2f} Hz')
        else:
            self.get_logger().warn(f'⚠️ Map update rate below target (10 Hz): {self.map_rate:.2f} Hz')

def main(args=None):
    rclpy.init(args=args)
    performance_monitor = VSLAMPerformanceMonitor()

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
# vslam_performance_test.sh
# Automated VSLAM performance validation script

echo "Starting Isaac ROS VSLAM Performance Validation..."

# Function to check if ROS 2 is running
check_ros2() {
    if ! pgrep -f "ros2" > /dev/null; then
        echo "ERROR: ROS 2 processes not found. Please start Isaac ROS VSLAM pipeline first."
        exit 1
    fi
}

# Function to measure rate
measure_rate() {
    local topic=$1
    local duration=${2:-10}  # Default 10 seconds
    echo "Measuring rate for topic: $topic over ${duration}s..."

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

echo "Validating VSLAM performance..."
echo "Target: 10+ Hz for map updates"

# Measure key topic rates
MAP_RATE=$(measure_rate "/vslam/occupancy_grid" 10)
POSE_RATE=$(measure_rate "/vslam/pose" 10)
FEATURE_RATE=$(measure_rate "/visual_slam/features" 10)

echo "Results:"
echo "Map Update Rate: $MAP_RATE Hz"
echo "Pose Update Rate: $POSE_RATE Hz"
echo "Feature Processing Rate: $FEATURE_RATE Hz"

# Validate against targets
echo ""
echo "Validation Results:"
if (( $(echo "$MAP_RATE >= 10.0" | bc -l) )); then
    echo "✅ Map update rate: PASS (Target: 10+, Actual: $MAP_RATE)"
else
    echo "❌ Map update rate: FAIL (Target: 10+, Actual: $MAP_RATE)"
fi

if (( $(echo "$POSE_RATE >= 20.0" | bc -l) )); then
    echo "✅ Pose update rate: PASS (Target: 20+, Actual: $POSE_RATE)"
else
    echo "⚠️ Pose update rate: WARNING (Target: 20+, Actual: $POSE_RATE)"
fi

# System resource check
echo ""
monitor_resources

echo ""
echo "VSLAM performance validation complete."
echo "Note: For actual validation, run this script while the Isaac ROS VSLAM pipeline is actively processing data."
echo "Required: Robot moving in environment to generate map updates."
```

## Performance Optimization Guidelines

### If Map Update Rate is Below Target
1. **Reduce Map Resolution**: Increase cell size (e.g., from 0.05m to 0.1m)
2. **Optimize Feature Detection**: Reduce max features to process
3. **Adjust Tracking Parameters**: Reduce landmark count to track
4. **Hardware Upgrade**: Use more powerful GPU for processing

### Memory Optimization
- Use memory pools and caching for image data
- Implement proper cleanup of landmark data
- Monitor for memory leaks over extended runs
- Use efficient data structures for map representation

### Processing Optimization
- Enable multithreading for parallel processing
- Use hardware acceleration (GPU) where available
- Optimize queue depths for different processing stages
- Implement adaptive processing based on system load

## Validation Report Template

### VSLAM Performance Validation Report
- **Test Environment**: [System specifications]
- **Test Date**: [Date of testing]
- **VSLAM Configuration**: [Active configuration]
- **Test Scenarios**: [List of scenarios tested]
- **Results Summary**: [Key performance metrics]
- **Pass/Fail Status**: [Overall validation status]
- **Recommendations**: [Any optimization recommendations]

## Expected Outcomes

With the current configuration and appropriate hardware, the Isaac ROS VSLAM system should achieve:
- **Minimum**: 10 Hz map update rate
- **Target**: 10-30 Hz for optimal performance
- **Acceptable**: 8-10 Hz for basic functionality (with reduced complexity)

## Next Steps for Validation

1. Deploy VSLAM pipeline configuration to Isaac Sim environment
2. Run performance validation scripts with robot moving in environment
3. Document actual performance achieved
4. Adjust configuration if needed to meet 10 Hz target
5. Update this report with actual performance metrics

## Performance Troubleshooting

### Common Performance Issues
- **Low Map Rate**: Check GPU memory usage and available features
- **High Latency**: Verify QoS settings and processing pipeline
- **Memory Leaks**: Monitor memory usage over time
- **CPU Bottlenecks**: Check for single-threaded processing blocks

### Performance Monitoring Tools
- ROS 2 built-in tools: `ros2 topic hz`, `ros2 topic delay`
- System monitoring: `htop`, `nvidia-smi`, `iotop`
- Isaac ROS utilities: Node profilers and performance monitors