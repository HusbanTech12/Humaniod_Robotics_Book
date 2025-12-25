# Quickstart Guide: NVIDIA Isaac AI-Robot Brain

## Prerequisites

### System Requirements
- Ubuntu 22.04 LTS
- NVIDIA GPU with CUDA support (RTX 3080 or better recommended)
- 16GB+ RAM
- Multi-core CPU (8+ cores recommended)
- 50GB+ free disk space

### Software Dependencies
- ROS 2 Humble Hawksbill
- NVIDIA Isaac Sim and Isaac Lab
- Isaac ROS packages
- Nav2 navigation stack
- Python 3.8+

## Installation

### 1. Install ROS 2 Humble
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-ros-base
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-build
```

### 2. Install NVIDIA Isaac Sim and Isaac Lab
```bash
# Follow NVIDIA Isaac Sim installation guide:
# 1. Download Isaac Sim from NVIDIA Developer website
# 2. Extract to desired location (e.g., ~/isaac-sim)
# 3. Install dependencies as per documentation

# Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh -i

# Source Isaac Lab environment
source ./isaaclab.sh -a
```

### 3. Install Isaac ROS
```bash
# Install Isaac ROS dependencies
sudo apt install -y python3-pip
pip3 install rospkg catkin_pkg

# Clone Isaac ROS repositories
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_perceptor.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git

# Build Isaac ROS packages
cd ~/isaac_ros_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select isaac_ros_common isaac_ros_perceptor isaac_ros_visual_slam
source install/setup.bash
```

### 4. Install Nav2
```bash
sudo apt update
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup
```

## Basic Usage

### 1. Launch Isaac Sim with Humanoid Robot
```bash
# Source all required environments
source /opt/ros/humble/setup.bash
source ~/isaac_ros_ws/install/setup.bash
source ~/IsaacLab/packages/setup/isaac-sim.sh

# Launch Isaac Sim with a humanoid robot
cd ~/IsaacLab
python3 source/standalone/run.py --env=Isaac-Locomotion-Unitree-A1-Scene-v0
```

### 2. Start Perception Pipeline
```bash
# Terminal 1: Launch Isaac ROS perception nodes
source ~/isaac_ros_ws/install/setup.bash
ros2 launch isaac_ros_perceptor isaac_ros_perceptor.launch.py

# Terminal 2: Verify perception topics
source ~/isaac_ros_ws/install/setup.bash
ros2 topic list | grep perception
```

### 3. Run VSLAM
```bash
# Terminal 1: Launch Visual SLAM
source ~/isaac_ros_ws/install/setup.bash
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam.launch.py

# Terminal 2: Check SLAM output
source ~/isaac_ros_ws/install/setup.bash
ros2 topic echo /slam_map
```

### 4. Start Navigation
```bash
# Terminal 1: Launch Nav2
source /opt/ros/humble/setup.bash
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true

# Terminal 2: Send navigation goal
source /opt/ros/humble/setup.bash
ros2 run nav2_msgs nav2_msgs__action__NavigateToPose --goal "pose: {position: {x: 1.0, y: 1.0, z: 0.0}, orientation: {w: 1.0}}"
```

## Example: Perception → Navigation Pipeline

```python
#!/usr/bin/env python3
# example_perception_navigation.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

class PerceptionNavigationNode(Node):
    def __init__(self):
        super().__init__('perception_navigation_node')

        # Subscriptions for perception data
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10)

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def image_callback(self, msg):
        # Process image to detect goal position
        # This is a simplified example
        goal_pose = self.process_image_for_goal(msg)
        self.send_navigation_goal(goal_pose)

    def process_image_for_goal(self, image_msg):
        # Placeholder for actual perception processing
        # In real implementation, this would use Isaac ROS perception nodes
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = 1.0  # Example goal
        pose.pose.position.y = 1.0
        pose.pose.orientation.w = 1.0
        return pose

    def send_navigation_goal(self, pose):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        self.get_logger().info('Goal accepted')

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Reinforcement Learning Example

```bash
# Train a simple locomotion policy
cd ~/IsaacLab
python3 source/standalone/run.py --task=Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs=16 --save_metrics

# The trained policy will be saved and can be used for sim-to-real transfer
```

## Troubleshooting

### Common Issues
1. **Isaac Sim won't launch**: Ensure NVIDIA GPU drivers are properly installed and CUDA is working
2. **Perception nodes not publishing**: Check that Isaac Sim is publishing sensor data
3. **Navigation fails**: Verify that the robot's URDF and navigation parameters are correctly configured
4. **Performance issues**: Reduce simulation complexity or upgrade hardware

### Performance Optimization
- Use Isaac Lab's environment randomization features
- Optimize robot URDF for simulation performance
- Adjust Isaac Sim rendering quality settings based on requirements

## Next Steps

1. Follow the detailed module documentation for in-depth tutorials
2. Experiment with different humanoid robots and environments
3. Implement custom perception and navigation algorithms
4. Explore sim-to-real transfer techniques