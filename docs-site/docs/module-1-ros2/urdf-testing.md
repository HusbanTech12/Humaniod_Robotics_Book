# Testing URDF Model Loading

## Overview

This document explains how to test that your URDF model loads correctly in RViz and Gazebo environments. Testing is crucial to ensure your humanoid robot model is properly defined and ready for simulation or real-world use.

## Prerequisites

Before testing, ensure that:
1. ROS 2 Humble Hawksbill is installed and sourced
2. The `humanoid_control` package is built
3. You have terminal access to run RViz or Gazebo
4. Your URDF file is properly formatted and located in the correct directory

## Testing URDF Model Validity

### Step 1: Validate URDF Syntax

First, check that your URDF file is syntactically correct:

```bash
# Check URDF syntax
check_urdf src/ros2_packages/humanoid_control/urdf/basic_humanoid.urdf
```

This command will report any XML syntax errors or structural issues in your URDF.

### Step 2: Visualize URDF Structure

Visualize the kinematic structure of your robot:

```bash
# Generate a graph of the URDF structure
urdf_to_graphiz src/ros2_packages/humanoid_control/urdf/basic_humanoid.urdf
```

This creates `.dot` and `.pdf` files showing the link-joint relationships in your robot.

## Testing in RViz

### Step 1: Launch the Visualization

Use the launch file created for URDF visualization:

```bash
cd ~/ros2_ws  # or wherever your ROS 2 workspace is located
source install/setup.bash
ros2 launch humanoid_control display_humanoid.launch.py
```

### Step 2: Verify Visualization

In RViz, you should see:
- The complete humanoid robot model with all links (torso, head, arms, legs)
- Proper joint connections between links
- Correct colors and materials as defined in the URDF
- The ability to move joints using the joint state publisher GUI

### Step 3: Check Robot State

Verify that the robot model is properly configured:
- All links should be visible and connected
- Joint axes should be correctly oriented
- No links should be floating or disconnected
- The model should look like the intended humanoid robot

## Testing Joint Movement

### Step 1: Use Joint State Publisher GUI

The launch file includes the joint state publisher GUI, which allows you to manually control joint positions:

1. In the joint state publisher GUI, you should see sliders for all joints in your humanoid model
2. Move the sliders to test that all joints are properly connected and move as expected
3. Verify that joint limits are respected (joints shouldn't move beyond their defined limits)

### Step 2: Check Joint Ranges

Ensure that:
- Shoulder joints move in the expected range
- Elbow joints bend in the correct direction
- Hip, knee, and ankle joints move properly
- The head rotates as expected

## Testing with Robot State Publisher

### Step 1: Command Line Testing

You can also publish joint states from the command line:

```bash
# Publish a joint state message
ros2 topic pub /joint_states sensor_msgs/msg/JointState "{
  name: ['left_shoulder_joint', 'right_shoulder_joint'],
  position: [0.5, -0.5]
}"
```

### Step 2: Verify Response

Check that RViz updates to reflect the commanded joint positions.

## Testing in Gazebo (Simulation)

### Step 1: Launch with Gazebo

To test in Gazebo simulation:

```bash
# First, make sure you have Gazebo installed
sudo apt install ros-humble-gazebo-ros2-control ros-humble-gazebo-dev

# Launch Gazebo with your robot
ros2 launch gazebo_ros gazebo.launch.py
```

In another terminal:

```bash
# Spawn your robot in Gazebo
ros2 run gazebo_ros spawn_entity.py -entity basic_humanoid -file src/ros2_packages/humanoid_control/urdf/basic_humanoid.urdf -x 0 -y 0 -z 1
```

### Step 2: Verify Simulation

In Gazebo, you should see:
- The robot model loaded correctly
- Proper physics properties (mass, inertia)
- Correct joint behavior when actuated
- IMU sensor properly positioned in the head

## Common Issues and Troubleshooting

### 1. URDF Validation Errors

If `check_urdf` reports errors:
- Check for missing closing tags
- Verify all joint and link names are unique
- Ensure parent-child relationships are valid
- Check that all required elements are present

### 2. Visualization Issues

If the robot doesn't appear correctly in RViz:
- Verify the robot_description parameter is set correctly
- Check that the URDF file path is correct
- Ensure all mesh files (if used) are in the correct location
- Verify that the robot model is not too large or too small

### 3. Joint Issues

If joints don't move properly:
- Check joint types (revolute, continuous, prismatic, fixed)
- Verify joint axes are correctly defined
- Ensure joint limits are appropriate
- Check that parent and child links are correctly specified

### 4. Material/Color Issues

If colors don't appear:
- Verify material definitions in the URDF
- Check that material names are unique
- Ensure the material properties are correctly defined

## Expected Results

When the URDF model is loaded successfully:
- The complete humanoid robot appears in RViz
- All joints can be moved using the joint state publisher
- The robot maintains its structural integrity
- No errors appear in the terminal output
- Joint limits are respected
- The IMU sensor is properly positioned

## Next Steps

Once your URDF model loads correctly:
- Add controllers for more sophisticated joint control
- Integrate with your AI bridge node for autonomous control
- Test in simulation with Gazebo
- Add more sophisticated sensors to your robot model
- Implement inverse kinematics for more natural movement