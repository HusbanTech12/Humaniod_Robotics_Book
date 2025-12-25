# Isaac ROS Perception Pipeline Architecture

## Overview
This document describes the architecture of the Isaac ROS Perception Pipeline, which processes multimodal sensor data to enable environment understanding for the humanoid robot.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Isaac ROS Perception Pipeline                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Sensors   │  │  Isaac ROS  │  │  Isaac ROS  │  │  Outputs    │  │
│  │             │  │  Processing │  │  Processing │  │             │  │
│  │ • RGB Cam   │  │ • Image     │  │ • Object    │  │ • Detections│  │
│  │ • Depth Cam │  │   Rectifier │  │   Detection │  │ • Point     │  │
│  │ • LiDAR     │  │ • Depth     │  │ • Spatial   │  │   Clouds    │  │
│  │ • IMU       │  │   Processing│  │   Detection │  │ • Transforms│  │
│  │             │  │ • Sensor    │  │ • SLAM      │  │ • Maps      │  │
│  └─────────────┘  │   Fusion    │  │             │  └─────────────┘  │
│         │         └─────────────┘  └─────────────┘         │          │
│         ▼                ▼                  ▼              ▼          │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                   ROS 2 Communication Layer                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────┐  │  │
│  │  │ /camera/*   │  │ /detectnet/*│  │ /sensor_    │  │ /map │  │  │
│  │  │             │  │             │  │ fusion/*    │  │      │  │  │
│  │  │ • image_raw │  │ • detections│  │ • fused_    │  │ •    │  │  │
│  │  │ • camera_   │  │ • spatial_  │  │   data      │  │ occupancy ││
│  │  │   info      │  │   detections│  │ • tracked_  │  │ •    │  │  │
│  │  │ • image_    │  │             │  │   objects   │  │ grid │  │  │
│  │  │   rect      │  └─────────────┘  └─────────────┘  └──────┘  │  │
│  │  │ • image_    │                                              │  │
│  │  │   metric    │                                              │  │
│  │  └─────────────┘                                              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Detailed Processing Pipeline

### 1. Input Stage: Sensor Data Acquisition
```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Sensor Data Input                              │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│ │ RGB Camera  │  │ Depth Cam   │  │ LiDAR       │  │ IMU         │   │
│ │             │  │             │  │             │  │             │   │
│ │ • 640x480   │  │ • Metric    │  │ • 360° Scan │  │ • Accel/    │   │
│ │ • 30Hz      │  │ • 30Hz      │  │ • 10Hz      │  │   Gyro Data │   │
│ │ • Distorted │  │ • 16-bit    │  │ • Range:    │  │ • 100Hz     │   │
│ │ • Raw Data  │  │ • Metric    │  │   0.1-30m   │  │ • Orientation│  │
│ └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│         │               │               │               │              │
│         ▼               ▼               ▼               ▼              │
│    /camera/rgb     /camera/depth      /scan         /imu/data        │
│    /image_raw      /image_raw                         /orientation    │
│    /camera_info    /camera_info                       /angular_vel    │
│                                                         /linear_acc   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. Preprocessing Stage: Image Rectification and Depth Processing
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Preprocessing Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│ │ Image           │    │ Depth           │    │ Camera          │     │
│ │ Rectification   │    │ Processing      │    │ Calibration     │     │
│ │                 │    │                 │    │                 │     │
│ │ • Distortion    │    │ • Unit          │    │ • Intrinsic     │     │
│ │   Correction    │    │   Conversion    │    │   Parameters    │     │
│ │ • Resolution    │    │ • Filtering     │    │ • Extrinsics    │     │
│ │   Adjustment    │    │ • Hole Filling  │    │ • Validation    │     │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│         │                       │                       │              │
│         ▼                       ▼                       ▼              │
│ /camera/rgb/image_    /camera/depth/        /camera/rgb/              │
│ rect_color            image_metric          camera_info               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. Processing Stage: Object Detection and Spatial Analysis
```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Core Processing Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│ │ Object          │    │ Spatial         │    │ AprilTag        │     │
│ │ Detection       │    │ Detection       │    │ Detection       │     │
│ │ (DetectNet)     │    │ (3D Location)   │    │ (Pose Estimation│     │
│ │                 │    │                 │    │  & Localization)│     │
│ │ • SSD MobileNet │    │ • Depth         │    │ • Tag36h11      │     │
│ │ • COCO trained  │    │   Integration   │    │ • Pose          │     │
│ │ • 2D Bounding   │    │ • 3D Bounding   │    │   Estimation    │     │
│ │   Boxes         │    │   Boxes         │    │ • Transform     │     │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│         │                       │                       │              │
│         ▼                       ▼                       ▼              │
│ /detectnet/         /detectnet/           /apriltag/                  │
│ detections          spatial_detections    detections                  │
│                     /detectnet/           /apriltag/                  │
│                     mask_image            transforms                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4. Fusion Stage: Multi-Sensor Data Integration
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Sensor Fusion Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│ │ Camera-LiDAR    │    │ IMU Integration │    │ Object Tracking │     │
│ │ Fusion          │    │                 │    │                 │     │
│ │                 │    │ • Orientation   │    │ • Data          │     │
│ │ • Projection    │    │ • Velocity      │    │   Association   │     │
│ │ • Point Cloud   │    │ • Acceleration  │    │ • Tracking      │     │
│ │   Generation    │    │ • Sensor        │    │ • Prediction    │     │
│ │ • 3D Object     │    │   Fusion        │    │                 │     │
│ │   Enhancement   │    │                 │    │                 │     │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│         │                       │                       │              │
│         ▼                       ▼                       ▼              │
│ /fused_camera_      /fused_imu/           /tracked_objects/          │
│ lidar               data                    /sensor_fusion/           │
│ /fused_pointcloud                             tracked_objects         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5. Output Stage: Processed Perception Results
```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Perception Outputs                                │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│ │ Object      │  │ Point Cloud │  │ Spatial     │  │ Semantic    │   │
│ │ Detections  │  │ Data        │  │ Maps        │  │ Segmentation│   │
│ │             │  │             │  │             │  │             │   │
│ │ • 2D/3D     │  │ • 3D Points │  │ • Occupancy │  │ • Pixel-    │   │
│ │   Bounding  │  │ • RGB       │  │ • Grid Maps │  │   level     │   │
│ │   Boxes     │  │ • Intensity │  │ • Free Space│  │   Labels    │   │
│ │ • Class IDs │  │ • Normals   │  │ • Obstacles │  │ • Instance  │   │
│ └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│         │               │               │               │              │
│         ▼               ▼               ▼               ▼              │
│ /detectnet/       /sensor_fusion/   /occupancy_map/   /semantic_     │
│ detections        fused_pointcloud    /map              segmentation │
│ /detectnet/       /points_raw         /projected_map                  │
│ spatial_                              /grid                           │
│ detections                                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

## ROS 2 Topic Flow

```
Sensor Data Flow:
/camera/rgb/image_raw → /camera/rgb/image_rect_color → /detectnet/image_input
/camera/depth/image_raw → /camera/depth/image_metric → /detectnet/depth_image
/scan → /sensor_fusion/lidar_input
/imu/data → /sensor_fusion/imu_input

Processing Flow:
/detectnet/image_input → [DetectNet Node] → /detectnet/detections
/detectnet/detections + /camera/depth/image_rect_raw → [Spatial Detection] → /detectnet/spatial_detections
/camera/rgb/image_rect_color → [AprilTag Node] → /apriltag/detections

Fusion Flow:
/detectnet/spatial_detections + /sensor_fusion/* → [Sensor Fusion] → /sensor_fusion/fused_data
/tracked_objects → /sensor_fusion/tracked_objects
```

## Component Integration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    System Integration View                            │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│ │ Isaac Sim       │    │ Isaac ROS       │    │ Perception      │     │
│ │ (Simulation)    │    │ (Processing)    │    │ (Pipeline)      │     │
│ │                 │    │                 │    │                 │     │
│ │ • Physics       │    │ • Image         │    │ • Object        │     │
│ │   Simulation    │    │   Processing    │    │   Detection     │     │
│ │ • Sensor        │    │ • Depth         │    │ • Sensor        │     │
│ │   Simulation    │    │   Processing    │    │   Fusion        │     │
│ │ • Rendering     │    │ • Detection     │    │ • SLAM          │     │
│ └─────────────────┘    │ • Calibration   │    │ • Mapping       │     │
│         │               └─────────────────┘    └─────────────────┘     │
│         │                       │                       │              │
│         └───────────────────────┼───────────────────────┘              │
│                                 │                                      │
│                    ┌─────────────────────────┐                         │
│                    │ ROS 2 Communication   │                         │
│                    │                       │                         │
│                    │ • Real-time topics    │                         │
│                    │ • QoS policies        │                         │
│                    │ • Message filtering   │                         │
│                    │ • Synchronization     │                         │
│                    └─────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

- **Processing Rate**: 30+ FPS for camera processing
- **Latency**: <50ms end-to-end processing
- **Memory Usage**: Optimized with memory pools and caching
- **Hardware Acceleration**: GPU and TensorRT acceleration
- **Multi-threading**: Parallel processing for real-time performance

## Quality of Service Settings

- **Reliability**: Best-effort for sensor data
- **Durability**: Volatile for real-time processing
- **History**: Keep-last with configurable depth
- **Synchronization**: Approximate time synchronization for multi-sensor fusion