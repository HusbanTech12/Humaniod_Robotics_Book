# Isaac Sim Setup for Humanoid Robot

## Overview

This document provides instructions for setting up NVIDIA Isaac Sim with a humanoid robot for simulation. The setup includes installing Isaac Sim, configuring the humanoid robot model, and establishing the necessary simulation environment with physics, sensors, and rendering.

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA GPU with compute capability 6.0 or higher (RTX series recommended)
- **RAM**: 16GB or more
- **Storage**: 50GB+ free disk space
- **CPU**: Multi-core processor (8+ cores recommended)

### Software Dependencies
- **NVIDIA GPU Drivers**: Version 470 or higher
- **CUDA**: Version 11.8 or higher
- **Docker**: (Optional) For containerized deployment
- **ROS 2 Humble Hawksbill**: For ROS bridge functionality

## Installation

### 1. Install Isaac Sim

#### Option A: Download from NVIDIA Developer Portal (Recommended)
1. Visit [NVIDIA Isaac Sim Downloads](https://developer.nvidia.com/isaac-sim)
2. Download the latest version compatible with your system
3. Extract the archive to your desired location (e.g., `~/isaac-sim`)

#### Option B: Using Omniverse Launcher
1. Download and install the Omniverse Launcher from NVIDIA
2. Search for "Isaac Sim" in the app store
3. Install the latest stable version

### 2. Verify Installation

```bash
# Navigate to Isaac Sim directory
cd ~/isaac-sim

# Launch Isaac Sim to verify installation
./isaac-sim.sh
```

### 3. Install Isaac Lab (Optional but Recommended)

```bash
# Clone Isaac Lab repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install Isaac Lab
./isaaclab.sh -i

# Source the environment
source ./isaaclab.sh -a
```

## Robot Configuration

### 1. Humanoid Robot Model Setup

The humanoid robot model is configured using the Unitree A1 quadruped model adapted for bipedal locomotion. The robot configuration includes:

- **Base Link**: Main body with dimensions 0.193m x 0.09m x 0.055m
- **Leg Configuration**: Four legs with hip, thigh, and calf joints
- **Sensors**: RGB camera, depth camera, LiDAR, IMU, and force/torque sensors

### 2. URDF Configuration

The robot model is defined in URDF (Unified Robot Description Format) and located at:
`docs-site/src/components/isaac-sim-examples/robot/unitree_a1.urdf`

Key URDF elements:
- **Links**: Base, hip, thigh, and calf links for each leg
- **Joints**: Revolute joints with position limits and effort constraints
- **Inertial Properties**: Mass, center of mass, and inertia tensors
- **Visual/Collision**: Mesh definitions for rendering and collision detection

### 3. Sensor Configuration

The robot is equipped with multiple sensors as defined in:
`docs-site/src/components/isaac-sim-examples/robot/sensors_config.yaml`

#### Camera Sensors
- **RGB Camera**: 640x480 resolution, 60° FOV
- **Depth Camera**: Aligned with RGB camera, 640x480 resolution
- **Topics**: `/camera/rgb/image_raw`, `/camera/depth/image_raw`

#### LiDAR Sensor
- **Type**: Velodyne VLP-16 equivalent
- **Parameters**: 16 channels, 10Hz rotation rate, 100m max range
- **Topic**: `/scan`

#### IMU Sensor
- **Type**: 9-axis IMU simulation
- **Frequency**: 100Hz
- **Topic**: `/imu/data`

## Environment Setup

### 1. Scene Configuration

The simulation environment is configured in:
`docs-site/src/components/isaac-sim-examples/scene_config.yaml`

The default scene includes:
- **Ground Plane**: 10m x 10m with configurable material properties
- **Lighting**: Dome light and directional lights with adjustable intensity
- **Obstacles**: Static obstacles for navigation testing
- **Interactive Objects**: Movable objects for manipulation tasks

### 2. Physics Configuration

Physics parameters are defined in:
`docs-site/src/components/isaac-sim-examples/physics_config.yaml`

Key physics settings:
- **Gravity**: [0.0, 0.0, -9.81] m/s²
- **Solver**: TGS (Time-Stepping Gauss-Seidel) solver
- **Time Step**: 0.005s simulation step
- **Collision Detection**: Enabled with configurable offsets

### 3. Environment Randomization

For sim-to-real transfer, environment randomization is configured in:
`docs-site/src/components/isaac-sim-examples/randomization_config.yaml`

Randomization includes:
- **Lighting**: Intensity and color temperature variation
- **Materials**: Albedo, roughness, and metallic property variation
- **Objects**: Position, rotation, and scale variation
- **Physics**: Friction and restitution coefficient variation

## Launching the Simulation

### 1. Using the Launch Script

The simulation can be launched using the provided launch script:

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Navigate to the project directory
cd ~/humanoid-robotic-book/docs-site

# Launch Isaac Sim with the humanoid robot
python3 -m docs-site/src/components/isaac-sim-examples/launch/isaac_sim_launch.py
```

### 2. Manual Launch

Alternatively, you can launch Isaac Sim manually:

```bash
# Start Isaac Sim application
cd ~/isaac-sim
./isaac-sim.sh

# In Isaac Sim, load the humanoid robot USD file:
# Window -> Quick Access -> Assets Browser
# Navigate to Isaac/Robots/Unitree/A1/unitree_a1.usd
# Drag and drop onto the stage
```

## Verification and Testing

### 1. Basic Functionality Check

After launching the simulation, verify the following:

1. **Robot Model**: The humanoid robot appears in the scene
2. **Sensors**: All sensors are publishing data to ROS topics
3. **Physics**: The robot responds correctly to physics simulation
4. **Rendering**: Visual rendering appears correctly

### 2. Sensor Data Verification

Check that sensor data is being published:

```bash
# Check available topics
ros2 topic list | grep -E "(camera|scan|imu|joint)"

# View camera images
ros2 run image_view image_view __ns:=/humanoid_robot __params_file:=path/to/camera_params.yaml

# Check LiDAR data
ros2 topic echo /scan
```

### 3. Performance Verification

Monitor simulation performance:

- **FPS**: Should maintain 30+ FPS for real-time operation
- **Physics**: Should run at 200+ Hz physics updates
- **Sensors**: Should publish at configured rates (e.g., 30Hz for cameras)

## Troubleshooting

### Common Issues and Solutions

#### 1. Low Rendering Performance
- **Issue**: Rendering FPS below 30
- **Solution**:
  - Reduce rendering quality in Isaac Sim settings
  - Check GPU memory usage
  - Close other GPU-intensive applications

#### 2. Physics Instability
- **Issue**: Robot behaving erratically or falling through surfaces
- **Solution**:
  - Verify physics parameters in `physics_config.yaml`
  - Check joint limits and constraints
  - Adjust solver parameters if needed

#### 3. Sensor Data Issues
- **Issue**: Missing or incorrect sensor data
- **Solution**:
  - Verify sensor configuration in `sensors_config.yaml`
  - Check ROS topic connections
  - Confirm Isaac Sim bridge is running

#### 4. GPU Memory Issues
- **Issue**: "Out of memory" errors
- **Solution**:
  - Reduce simulation complexity
  - Lower rendering resolution
  - Close other GPU applications

## Configuration Files Reference

- `env_config.yaml`: Main environment configuration
- `sensors_config.yaml`: Sensor setup and parameters
- `scene_config.yaml`: Scene and environment settings
- `physics_config.yaml`: Physics simulation parameters
- `randomization_config.yaml`: Domain randomization settings
- `isaac_sim_launch.py`: ROS 2 launch configuration

## Next Steps

After completing the Isaac Sim setup:

1. **Perception Pipeline**: Configure Isaac ROS perception nodes
2. **Navigation**: Set up Nav2 for humanoid navigation
3. **Control**: Implement reinforcement learning for locomotion
4. **Testing**: Validate sim-to-real transfer capabilities

## Performance Optimization Tips

- **Reduce rendering complexity** when not needed for training
- **Use appropriate physics substeps** based on simulation requirements
- **Optimize sensor update rates** for your application needs
- **Batch multiple environments** for parallel training
- **Use domain randomization** to improve sim-to-real transfer