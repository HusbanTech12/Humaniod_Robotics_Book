# Custom message for joint commands in digital twin simulation

# Header
std_msgs/Header header

# Joint information
string joint_name    # Name of the joint to control
string command_type  # Type of command ("position", "velocity", "effort")

# Command value
float64 command_value  # Desired value for the command

# Additional parameters
float64 duration      # Duration for the command to complete (optional)
float64 max_effort    # Maximum effort allowed (optional)