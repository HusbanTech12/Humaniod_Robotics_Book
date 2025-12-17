# Custom message for sensor data in digital twin simulation

# Header
std_msgs/Header header

# Sensor information
string sensor_type  # Type of sensor (lidar, camera, imu, etc.)
string sensor_name  # Name of the sensor
string data_format  # Format of the data (depends on sensor type)

# Sensor data (variable length array of floats)
float64[] data      # Raw sensor data values

# Additional metadata
float64 timestamp   # Timestamp of the sensor reading
string frame_id    # Coordinate frame of the sensor