# Setting up ROS 2 Humble Hawksbill

This guide will help you install ROS 2 Humble Hawksbill, which is the recommended version for this educational module.

## Prerequisites

- Ubuntu 22.04 LTS (recommended) or a compatible Linux distribution
- At least 4GB of RAM
- At least 20GB of free disk space

## Installation Steps

### 1. Set up the ROS 2 apt repository

```bash
# Add the ROS 2 GPG key
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros- humble.repos | sudo tee /etc/ros/ros.repos
sudo apt update
```

### 2. Install ROS 2 packages

```bash
# Install the desktop variant (includes Gazebo)
sudo apt install -y ros-humble-desktop
```

### 3. Install additional dependencies

```bash
# Install Python and development tools
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 4. Initialize rosdep

```bash
sudo rosdep init
rosdep update
```

### 5. Source the ROS 2 environment

```bash
# Add to your bashrc to source automatically
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source /opt/ros/humble/setup.bash
```

## Verify Installation

Test that ROS 2 is properly installed by running a simple example:

```bash
# Terminal 1
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker

# Terminal 2
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```

You should see messages being published by the talker and received by the listener.

## Next Steps

After installing ROS 2, you can proceed with creating your first ROS 2 workspace and packages for the humanoid robotics examples in this module.