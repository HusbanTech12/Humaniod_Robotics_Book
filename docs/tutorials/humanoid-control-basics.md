# Humanoid Control Basics Tutorial

## Overview

This tutorial covers the fundamentals of controlling humanoid robots using ROS 2. You'll learn how to create, configure, and control a basic humanoid robot model with emphasis on proper URDF modeling, joint control, and integration with AI systems.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Create a complete humanoid robot URDF model
- Set up joint control systems for humanoid robots
- Integrate sensor systems into your humanoid model
- Control humanoid robot joints using ROS 2
- Connect AI agents to your humanoid robot control system

## Prerequisites

Before starting this tutorial, you should:
- Have completed the ROS 2 architecture module
- Understand basic ROS 2 concepts (nodes, topics, services)
- Have a working ROS 2 Humble Hawksbill installation
- Have completed the Python integration module

## Step 1: Understanding Humanoid Robot Structure

A humanoid robot typically consists of:
- A torso (trunk)
- A head
- Two arms (with shoulders, elbows, wrists, and hands)
- Two legs (with hips, knees, ankles, and feet)

The joints in a humanoid robot are designed to mimic human movement patterns. Each joint has specific ranges of motion and capabilities.

## Step 2: Creating the URDF Model

### Basic Structure

Start with a basic URDF structure that includes all major body parts:

```xml
<?xml version="1.0"?>
<robot name="my_humanoid">
  <!-- Base link (torso) -->
  <link name="base_link">
    <!-- Visual, collision, and inertial properties -->
  </link>

  <!-- Head -->
  <link name="head">
    <!-- Head properties -->
  </link>

  <!-- Joints connecting body parts -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <!-- Joint properties -->
  </joint>

  <!-- Continue for arms, legs, etc. -->
</robot>
```

### Joint Types for Humanoid Robots

Different joints in a humanoid robot require different joint types:

- **Revolute joints**: Used for shoulders, elbows, hips, knees, ankles (limited rotation)
- **Continuous joints**: Used for joints that can rotate indefinitely (rare in humanoid robots)
- **Fixed joints**: Used for permanent connections (e.g., sensor mounting)

### Example Joint Definition

```xml
<joint name="left_elbow_joint" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_lower_arm"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0" upper="2.35" effort="8.0" velocity="1.0"/>
</joint>
```

## Step 3: Adding Sensors to Your Humanoid Robot

### IMU Sensor

An IMU (Inertial Measurement Unit) is crucial for humanoid balance and orientation:

```xml
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="head"/>
  <child link="imu_link"/>
  <origin xyz="0.05 0 0" rpy="0 0 0"/>
</joint>

<!-- Gazebo plugin for IMU -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <!-- IMU configuration -->
  </sensor>
</gazebo>
```

### Joint Encoders

Joint encoders are automatically provided by the joint state publisher:

```xml
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <update_rate>30</update_rate>
    <joint_name>left_shoulder_joint</joint_name>
    <!-- Add all other joint names -->
  </plugin>
</gazebo>
```

## Step 4: Setting Up Control Systems

### ROS 2 Control Interface

For simulation and real robot control, use the ROS 2 Control interface:

```xml
<ros2_control name="GazeboSystem" type="system">
  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>
  <joint name="left_shoulder_joint">
    <command_interface name="position"/>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
  </joint>
  <!-- Add interfaces for all joints -->
</ros2_control>
```

## Step 5: Controlling Your Humanoid Robot

### Joint Position Control

To control joint positions, publish to the appropriate topics:

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

    def send_trajectory_command(self, joint_names, positions, duration=1.0):
        msg = JointTrajectory()
        msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration), nanosec=0)

        msg.points = [point]
        self.publisher.publish(msg)
```

### Using Joint State Publisher for Testing

For initial testing, you can use the joint state publisher to manually control joints:

```bash
# Launch the visualization with joint control
ros2 launch humanoid_control display_humanoid.launch.py

# In another terminal, use rqt to control joints
ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller
```

## Step 6: Integration with AI Systems

### Connecting to AI Bridge

Connect your humanoid control system to the AI bridge node:

```python
class HumanoidAIController(Node):
    def __init__(self):
        super().__init__('humanoid_ai_controller')

        # Subscribe to AI commands
        self.ai_subscription = self.create_subscription(
            Float64MultiArray,
            'ai_joint_commands',
            self.ai_command_callback,
            10
        )

        # Publisher for joint trajectories
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

    def ai_command_callback(self, msg):
        # Convert AI commands to joint trajectory
        joint_names = [
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            # ... add all joints
        ]

        self.send_trajectory_command(joint_names, msg.data)
```

## Step 7: Testing Your Humanoid Robot

### Basic Functionality Test

1. Launch your robot model:
   ```bash
   ros2 launch humanoid_control display_humanoid.launch.py
   ```

2. Verify all links are visible and connected properly

3. Test joint movement using the joint state publisher GUI

4. Check that sensor data is being published correctly

### Integration Test

1. Run your AI bridge node:
   ```bash
   ros2 run ai_bridge ai_bridge
   ```

2. Verify that AI commands are received and converted to joint movements

3. Test the complete pipeline from sensor input to control output

## Best Practices

### 1. Proper Inertial Properties

Always define realistic inertial properties for each link:
- Mass values should reflect the actual weight of the part
- Inertia tensors should be physically plausible
- Center of mass should be correctly positioned

### 2. Realistic Joint Limits

Set appropriate joint limits based on human anatomy or your specific design:
- Shoulder joints: typically ±90° to ±180°
- Elbow joints: 0° to ~170° (flexion only)
- Knee joints: 0° to ~170° (flexion only)
- Wrist joints: ±45° for rotation

### 3. Safety in Control

Implement safety measures in your control system:
- Joint position limits
- Velocity limits
- Torque/effort limits
- Emergency stop capabilities

### 4. Modular Design

Structure your URDF model in a modular way:
- Separate files for different body parts
- Use xacro for parameterized models
- Include proper naming conventions

## Troubleshooting Common Issues

### Robot Falls Over in Simulation

- Check that inertial properties are properly defined
- Verify that the center of mass is within the support polygon
- Ensure sufficient friction coefficients in contact properties

### Joints Don't Move as Expected

- Verify joint axes are correctly defined
- Check that joint limits are appropriate
- Ensure command interfaces are properly configured

### AI Control Not Responding

- Check topic names match between nodes
- Verify message types are compatible
- Ensure timing requirements are met

## Advanced Topics

Once you've mastered the basics, consider exploring:

- **Inverse Kinematics**: Calculate joint angles for desired end-effector positions
- **Walking Pattern Generation**: Create stable walking gaits for your humanoid
- **Balance Control**: Implement feedback control for maintaining balance
- **Motion Planning**: Plan complex movements while avoiding self-collisions
- **Learning-based Control**: Use reinforcement learning for complex behaviors

## Summary

This tutorial covered the fundamentals of humanoid robot control in ROS 2, including:
- Creating a complete URDF model with proper joints and sensors
- Setting up control systems for joint actuation
- Integrating with AI systems for autonomous control
- Testing and validating your humanoid robot model

With this foundation, you're ready to explore more advanced humanoid robotics concepts and applications.