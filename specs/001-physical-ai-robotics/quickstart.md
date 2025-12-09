# Quickstart: The Robotic Nervous System (ROS 2)

## Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill installed
- Python 3.8 or higher
- Basic understanding of Python and robotics concepts

## Setup Environment

### 1. Install ROS 2 Humble
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros- humble.repos | sudo tee /etc/ros/ros.repos
sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 2. Initialize rosdep
```bash
sudo rosdep init
rosdep update
```

### 3. Source ROS 2 environment
```bash
source /opt/ros/humble/setup.bash
```

## Create Your First ROS 2 Package

### 1. Create workspace
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 2. Create a simple publisher/subscriber package
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python simple_humanoid_control --dependencies rclpy std_msgs
```

### 3. Create a basic publisher node
Edit `~/ros2_ws/src/simple_humanoid_control/simple_humanoid_control/joint_command_publisher.py`:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        self.publisher = self.create_publisher(Float64MultiArray, 'joint_commands', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = [0.0, 0.1 * self.i, 0.0, -0.1 * self.i]  # Example joint positions
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing joint commands: {msg.data}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    joint_command_publisher = JointCommandPublisher()
    rclpy.spin(joint_command_publisher)
    joint_command_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 4. Make the script executable and update setup.py
```bash
chmod +x simple_humanoid_control/simple_humanoid_control/joint_command_publisher.py
```

Update `setup.py` to include the entry point:
```python
entry_points={
    'console_scripts': [
        'joint_command_publisher = simple_humanoid_control.joint_command_publisher:main',
    ],
},
```

### 5. Build and run
```bash
cd ~/ros2_ws
colcon build --packages-select simple_humanoid_control
source install/setup.bash
ros2 run simple_humanoid_control joint_command_publisher
```

## Create a Simple URDF Model

Create `~/ros2_ws/src/simple_humanoid_control/urdf/simple_humanoid.urdf`:
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.15" radius="0.02"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.15" radius="0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>
</robot>
```

## Visualize Your Robot Model

### 1. Install visualization tools
```bash
sudo apt install ros-humble-joint-state-publisher-gui ros-humble-robot-state-publisher ros-humble-xacro
```

### 2. Launch the robot in RViz
```bash
cd ~/ros2_ws
ros2 launch simple_humanoid_control display.launch.py
```

## Connecting AI Agent to ROS 2

### 1. Create AI bridge node
Create `~/ros2_ws/src/simple_humanoid_control/simple_humanoid_control/ai_bridge.py`:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np

class AIBridge(Node):
    def __init__(self):
        super().__init__('ai_bridge')

        # Subscribe to sensor data
        self.sensor_subscription = self.create_subscription(
            Float64MultiArray,
            'sensor_data',
            self.sensor_callback,
            10)

        # Publish control commands
        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            'joint_commands',
            10)

        # Timer for AI processing
        self.timer = self.create_timer(0.1, self.ai_processing_callback)

        self.latest_sensor_data = None

    def sensor_callback(self, msg):
        self.latest_sensor_data = msg.data
        self.get_logger().info(f'Received sensor data: {msg.data}')

    def ai_processing_callback(self):
        if self.latest_sensor_data is not None:
            # Simple AI logic - in practice, this would be your ML model
            control_commands = self.simple_ai_logic(self.latest_sensor_data)

            # Publish the commands
            cmd_msg = Float64MultiArray()
            cmd_msg.data = control_commands
            self.command_publisher.publish(cmd_msg)
            self.get_logger().info(f'AI output: {control_commands}')

    def simple_ai_logic(self, sensor_data):
        # Placeholder for AI logic
        # In practice, this would call your ML model
        return [val * 0.1 for val in sensor_data]  # Simple transformation

def main(args=None):
    rclpy.init(args=args)
    ai_bridge = AIBridge()
    rclpy.spin(ai_bridge)
    ai_bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Multi-Node System

Create `~/ros2_ws/src/simple_humanoid_control/launch/simple_humanoid_system.launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_directory('simple_humanoid_control'),
        'urdf',
        'simple_humanoid.urdf'
    )

    return LaunchDescription([
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': open(urdf_path).read()}]
        ),

        # Joint state publisher (GUI for testing)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui'
        ),

        # AI Bridge Node
        Node(
            package='simple_humanoid_control',
            executable='ai_bridge',
            name='ai_bridge'
        ),

        # Joint Command Publisher (for testing)
        Node(
            package='simple_humanoid_control',
            executable='joint_command_publisher',
            name='joint_command_publisher'
        )
    ])
```

## Run the Complete System

```bash
cd ~/ros2_ws
colcon build --packages-select simple_humanoid_control
source install/setup.bash
ros2 launch simple_humanoid_control simple_humanoid_system.launch.py
```

## Next Steps

1. Explore the ROS 2 tutorials at https://docs.ros.org/en/humble/Tutorials.html
2. Learn more about URDF at https://wiki.ros.org/urdf
3. Experiment with different robot models and control strategies
4. Integrate your own AI models with the ROS 2 communication layer