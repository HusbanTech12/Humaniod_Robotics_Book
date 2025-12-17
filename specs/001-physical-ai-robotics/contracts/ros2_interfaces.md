# ROS 2 Interface Contracts: The Robotic Nervous System (ROS 2)

## Node: AI Bridge Node

### Publishers

#### `/joint_commands` (sensor_msgs/JointState)
- **Purpose**: Publishes joint position, velocity, and effort commands to control the humanoid robot
- **Message Format**:
  ```python
  std_msgs/Header header
  string[] name          # Joint names
  float64[] position     # Joint positions (radians)
  float64[] velocity     # Joint velocities (rad/s)
  float64[] effort       # Joint efforts (Nm)
  ```
- **QoS**: Reliable, transient local durability
- **Frequency**: 50-100 Hz for real-time control

#### `/robot_behavior_status` (std_msgs/String)
- **Purpose**: Publishes current behavior state of the robot
- **Message Format**:
  ```python
  string data            # Current behavior state (e.g., "walking", "standing", "idle")
  ```
- **QoS**: Best effort
- **Frequency**: On state change

### Subscribers

#### `/sensor_data` (sensor_msgs/JointState)
- **Purpose**: Receives sensor data from robot joints
- **Message Format**:
  ```python
  std_msgs/Header header
  string[] name          # Joint names
  float64[] position     # Joint positions (radians)
  float64[] velocity     # Joint velocities (rad/s)
  float64[] effort       # Joint efforts (Nm)
  ```
- **QoS**: Reliable
- **Expected Frequency**: 50-100 Hz

#### `/imu_data` (sensor_msgs/Imu)
- **Purpose**: Receives inertial measurement unit data
- **Message Format**:
  ```python
  std_msgs/Header header
  geometry_msgs/Quaternion orientation
  float64[9] orientation_covariance
  geometry_msgs/Vector3 angular_velocity
  float64[9] angular_velocity_covariance
  geometry_msgs/Vector3 linear_acceleration
  float64[9] linear_acceleration_covariance
  ```
- **QoS**: Reliable
- **Expected Frequency**: 100-200 Hz

### Services

#### `/configure_ai_behavior` (std_srvs/SetBool)
- **Purpose**: Enable/disable AI control of the robot
- **Request**:
  ```python
  bool data              # True to enable AI control, False to disable
  ```
- **Response**:
  ```python
  bool success           # True if command succeeded
  string message         # Status message
  ```

#### `/get_robot_state` (custom service: GetRobotState)
- **Purpose**: Retrieve complete robot state
- **Request**: Empty
- **Response**:
  ```python
  sensor_msgs/JointState joint_state
  sensor_msgs/Imu imu_data
  geometry_msgs/PoseStamped base_pose
  std_msgs/String behavior_status
  ```

### Actions

#### `/execute_walking_pattern` (control_msgs/FollowJointTrajectory)
- **Purpose**: Execute a predefined walking pattern
- **Goal**:
  ```python
  trajectory_msgs/JointTrajectory trajectory
  ```
- **Feedback**:
  ```python
  control_msgs/FollowJointTrajectoryFeedback feedback
  ```
- **Result**:
  ```python
  control_msgs/FollowJointTrajectoryResult result
  ```

## Node: Joint Control Node

### Publishers

#### `/joint_states` (sensor_msgs/JointState)
- **Purpose**: Publishes current joint states
- **Message Format**: Same as sensor_msgs/JointState
- **QoS**: Best effort
- **Frequency**: 50-100 Hz

### Subscribers

#### `/joint_position_commands` (std_msgs/Float64MultiArray)
- **Purpose**: Receives position commands for joints
- **Message Format**:
  ```python
  float64[] data         # Position commands for each joint
  ```
- **QoS**: Reliable
- **Expected Frequency**: 50-100 Hz

#### `/joint_velocity_commands` (std_msgs/Float64MultiArray)
- **Purpose**: Receives velocity commands for joints
- **Message Format**:
  ```python
  float64[] data         # Velocity commands for each joint
  ```
- **QoS**: Reliable
- **Expected Frequency**: 50-100 Hz

## Node: Sensor Processing Node

### Publishers

#### `/processed_sensor_data` (custom message: ProcessedSensorData)
- **Purpose**: Publishes processed sensor information
- **Message Format**:
  ```python
  std_msgs/Header header
  float64[] joint_angles
  float64[] joint_velocities
  float64[] imu_orientations
  float64[] imu_angular_velocities
  float64[] imu_linear_accelerations
  float64 center_of_mass_x
  float64 center_of_mass_y
  float64 center_of_mass_z
  bool balance_stable
  ```
- **QoS**: Reliable
- **Frequency**: 50-100 Hz

### Subscribers

#### `/raw_joint_states` (sensor_msgs/JointState)
- **Purpose**: Receives raw joint state data from hardware
- **Message Format**: Same as sensor_msgs/JointState
- **QoS**: Reliable
- **Expected Frequency**: 100-200 Hz

#### `/raw_imu_data` (sensor_msgs/Imu)
- **Purpose**: Receives raw IMU data from hardware
- **Message Format**: Same as sensor_msgs/Imu
- **QoS**: Reliable
- **Expected Frequency**: 200-500 Hz

## Validation Requirements

### Message Format Validation
- All messages must conform to ROS 2 message definitions
- Header timestamps must be current (within 100ms of system time)
- Data arrays must match expected sizes and ranges

### Communication Validation
- Publishers must maintain minimum required frequencies
- Subscribers must handle message loss gracefully
- Services must respond within 1 second
- Actions must provide feedback within 100ms of goal acceptance

### Error Handling
- Invalid message formats must be rejected with appropriate error messages
- Nodes must continue operating when connected nodes fail
- Parameter validation must occur at startup and during runtime