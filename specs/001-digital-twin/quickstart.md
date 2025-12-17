# Quickstart Guide: Digital Twin Simulation for Humanoid Robots

## Prerequisites

- Ubuntu 22.04 LTS (recommended for ROS 2 Humble Hawksbill)
- ROS 2 Humble Hawksbill installed
- Gazebo 11+ installed
- Unity 2021+ installed
- Python 3.8+ installed
- Git installed

## Installation Steps

### 1. Install ROS 2 Humble Hawksbill
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-rosdep2
sudo apt install -y python3-colcon-common-extensions

# Source ROS 2 environment
source /opt/ros/humble/setup.bash
```

### 2. Install Gazebo
```bash
sudo apt install -y gazebo
# Verify installation
gazebo --version
```

### 3. Set up workspace
```bash
# Create workspace directory
mkdir -p ~/digital_twin_ws/src
cd ~/digital_twin_ws

# Clone the project repository
git clone https://github.com/your-organization/humanoid-digital-twin.git src/humanoid_digital_twin

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
colcon build --packages-select humanoid_control ai_bridge
source install/setup.bash
```

### 4. Install Unity ROS 2 Bridge (if needed)
```bash
# The Unity ROS 2 Bridge package can be imported into Unity via the Unity Package Manager
# More details in the Unity visualization section
```

## Running the Simulation

### 1. Launch the basic humanoid simulation
```bash
# Source the workspace
source ~/digital_twin_ws/install/setup.bash

# Launch the humanoid robot simulation in Gazebo
ros2 launch humanoid_control humanoid_system.launch.py
```

### 2. Launch the Unity visualization (separate terminal)
```bash
# After starting Gazebo simulation, in a new terminal:
# Start Unity project for visualization
# (Unity project located at unity_projects/humanoid_visualization)
```

### 3. Publish sensor data to ROS 2 topics
```bash
# In a new terminal, source the workspace
source ~/digital_twin_ws/install/setup.bash

# Run the sensor processing node
ros2 run humanoid_control sensor_processing_node
```

### 4. Monitor sensor topics
```bash
# View LiDAR data
ros2 topic echo /humanoid/lidar_scan sensor_msgs/msg/LaserScan

# View IMU data
ros2 topic echo /humanoid/imu sensor_msgs/msg/Imu

# View camera data
ros2 topic echo /humanoid/camera/depth/image_raw sensor_msgs/msg/Image
```

## Basic Commands

### Load a robot model into simulation
```bash
ros2 service call /load_robot_model humanoid_control_interfaces/srv/LoadRobotModel "{
  model_file_path: 'package://humanoid_control/urdf/basic_humanoid.urdf',
  model_name: 'basic_humanoid',
  initial_pose: {
    position: {x: 0.0, y: 0.0, z: 1.0},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}"
```

### Send a joint command
```bash
ros2 topic pub /joint_commands humanoid_control_msgs/msg/JointCommand "{
  joint_name: 'hip_joint',
  command_type: 'position',
  command_value: 0.5
}"
```

## Troubleshooting

### Common Issues:

1. **Gazebo fails to start**
   - Check if GPU drivers are properly installed
   - Try running with `gazebo --verbose` for detailed logs

2. **ROS 2 nodes cannot communicate**
   - Ensure all terminals have sourced the workspace: `source ~/digital_twin_ws/install/setup.bash`
   - Check if ROS_DOMAIN_ID is consistent across terminals

3. **Sensor data not publishing**
   - Verify that sensor plugins are correctly configured in URDF
   - Check Gazebo plugins are loaded using `gz topic -l`

4. **Unity visualization not syncing**
   - Ensure ROS TCP Connector is properly configured in Unity
   - Check network connectivity between Unity and ROS 2 nodes