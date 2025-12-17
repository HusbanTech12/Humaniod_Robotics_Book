# Humanoid Model Configuration Guide

This guide explains how to configure and set up a humanoid robot model for digital twin simulation in Gazebo and Unity environments.

## Overview

The humanoid robot model is configured using Unified Robot Description Format (URDF) which defines the robot's physical properties, kinematic structure, and sensor placements. The configuration includes:

- Physical properties (mass, inertia, collision geometry)
- Joint definitions with limits and dynamics
- Visual geometry and materials
- Sensor placements and configurations
- Gazebo-specific plugins

## URDF Model Structure

The humanoid model is defined in `src/ros2_packages/humanoid_control/urdf/basic_humanoid.urdf` and follows a hierarchical structure:

```
base_link
├── torso
│   ├── head
│   ├── left_arm
│   └── right_arm
└── legs
    ├── left_leg
    └── right_leg
```

### Key Components

1. **Base Link**: The root of the robot with mass and inertia properties
2. **Torso**: Central body with head and arm attachments
3. **Limbs**: Arms and legs with appropriate joint constraints
4. **End Effectors**: Hands and feet for interaction

## Physical Properties Configuration

### Mass and Inertia

Each link in the robot model has defined mass and inertia properties:

```xml
<link name="link_name">
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0"
             iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>
```

### Collision and Visual Geometry

Collision and visual geometries are defined separately:

```xml
<link name="link_name">
  <collision>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
  <visual>
    <geometry>
      <mesh filename="package://humanoid_control/meshes/link_name.dae"/>
    </geometry>
    <material name="blue">
      <color rgba="0.0 0.0 1.0 1.0"/>
    </material>
  </visual>
</link>
```

## Joint Configuration

### Joint Types

The humanoid model uses several joint types:

- **Revolute Joints**: For rotational movement with limits
- **Fixed Joints**: For rigid connections
- **Continuous Joints**: For unlimited rotation (e.g., neck)

### Joint Limits

Joint limits are critical for realistic movement:

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="3.0"/>
</joint>
```

## Sensor Configuration

Sensors are configured as plugins within the URDF:

### IMU Sensor
```xml
<gazebo reference="torso">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
      </angular_velocity>
    </imu>
  </sensor>
</gazebo>
```

### LiDAR Sensor
```xml
<gazebo reference="head">
  <sensor name="lidar_sensor" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1.0</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
  </sensor>
</gazebo>
```

## Gazebo Plugins

### ROS Control Plugin

The ros_control plugin enables joint control:

```xml
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/humanoid</robotNamespace>
  </plugin>
</gazebo>
```

### Joint State Publisher

The joint state publisher plugin publishes joint states to ROS:

```xml
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <robotNamespace>/humanoid</robotNamespace>
    <jointName>joint_name</jointName>
  </plugin>
</gazebo>
```

## Loading the Model in Simulation

### Using Launch Files

The model can be loaded using the provided launch files:

```bash
# Load humanoid model in Gazebo
ros2 launch humanoid_control load_humanoid.launch.py

# Launch the complete humanoid system
ros2 launch humanoid_control humanoid_system.launch.py
```

### Parameters

The launch files accept several parameters:

- `use_gui`: Whether to use Gazebo GUI (default: true)
- `world_file`: Path to the world file (default: simple_room.world)
- `robot_model`: Path to the robot URDF file (default: basic_humanoid.urdf)

## Physics Configuration

Physics parameters are configured in the world file (`src/ros2_packages/humanoid_control/worlds/simple_room.world`):

```xml
<physics name="1ms" type="ode">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
</physics>
```

## Validation and Testing

### Physics Validation

Use the physics validation test to verify the robot responds to gravity:

```bash
ros2 run humanoid_control physics_validation_test
```

### Joint Validation

Validate joint limits and range of motion:

```bash
ros2 run humanoid_control joint_validation_test
```

## Troubleshooting

### Common Issues

1. **Model not loading**: Check URDF syntax with `check_urdf` command
2. **Joints not moving**: Verify joint controller configuration
3. **Physics instability**: Adjust physics parameters in world file
4. **TF transforms missing**: Ensure robot_state_publisher is running

### Debugging Commands

```bash
# Check URDF validity
check_urdf src/ros2_packages/humanoid_control/urdf/basic_humanoid.urdf

# View TF tree
ros2 run tf2_tools view_frames

# Monitor joint states
ros2 topic echo /joint_states
```

## Best Practices

1. **Realistic Mass Properties**: Use realistic mass and inertia values for stable simulation
2. **Joint Limits**: Define appropriate limits to prevent unrealistic poses
3. **Collision Geometry**: Use simplified collision geometry for better performance
4. **Visual Quality**: Balance visual quality with performance requirements
5. **Testing**: Regularly validate physics behavior and joint limits

## Next Steps

- Configure additional sensors (LiDAR, cameras, force/torque sensors)
- Set up Unity visualization for the humanoid model
- Implement perception and control pipelines
- Validate simulation accuracy against physical models