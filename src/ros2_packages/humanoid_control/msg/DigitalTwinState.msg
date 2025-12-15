# Custom message for digital twin state in simulation

# Header
std_msgs/Header header

# Twin identification
string twin_id      # Unique identifier for the digital twin
string model_name   # Name of the robot model

# State information
string state        # Current state (unloaded, loaded, configured, simulating, paused, stopped)
bool is_active      # Whether the twin is currently active in simulation

# Simulation parameters
float64 simulation_time    # Current simulation time
float64 real_time_factor   # Real-time factor of the simulation
bool is_synchronized      # Whether synchronized with visualization

# Additional metadata
string description        # Description of the twin state