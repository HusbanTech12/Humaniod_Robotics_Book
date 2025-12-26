# Quickstart Guide: Vision-Language-Action (VLA) Module

## Prerequisites

- Ubuntu 22.04 LTS (recommended)
- ROS 2 Humble Hawksbill installed
- NVIDIA GPU with CUDA support (RTX series recommended)
- Isaac Sim installed and licensed
- Isaac ROS packages installed
- Python 3.8 or higher
- OpenAI Whisper model files (or access to Whisper API)

## Installation Steps

### 1. Environment Setup
```bash
# Create workspace
mkdir -p ~/vla_ws/src
cd ~/vla_ws

# Install Isaac Sim and Isaac ROS
# Follow official NVIDIA installation guides for Isaac Sim and Isaac ROS packages
```

### 2. Clone Required Repositories
```bash
cd ~/vla_ws/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_perceptor.git
# Additional Isaac ROS packages as needed
```

### 3. Install Python Dependencies
```bash
pip3 install openai-whisper
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers
pip3 install rclpy
```

### 4. Build the Workspace
```bash
cd ~/vla_ws
colcon build --symlink-install
source install/setup.bash
```

## Basic Usage

### 1. Launch Isaac Sim Environment
```bash
# Terminal 1: Launch Isaac Sim with humanoid robot
ros2 launch isaac_sim_launcher humanoid_robot.launch.py
```

### 2. Start VLA System Components
```bash
# Terminal 2: Launch VLA pipeline
ros2 launch vla_examples vla_pipeline.launch.py
```

### 3. Run Voice Command Demo
```bash
# Terminal 3: Run voice command processing
ros2 run vla_examples voice_command_node
```

## Running the Complete VLA Pipeline

### 1. Launch All Components
```bash
# Use the combined launch file
ros2 launch vla_examples complete_vla_system.launch.py
```

### 2. Issue Voice Commands
The system will listen for voice commands and process them through the VLA pipeline:
- "Pick up the red cup" - Vision grounding and manipulation
- "Go to the kitchen" - Navigation and path planning
- "Clean the table" - Multi-step task execution

### 3. Monitor System Status
```bash
# Check ROS 2 topics
ros2 topic list

# Monitor system performance
ros2 run rqt_plot rqt_plot
```

## Configuration

### Voice Processing Configuration
Edit `~/vla_ws/src/vla_examples/config/voice_config.yaml`:
```yaml
voice_processing:
  model: "base"  # Whisper model size: tiny, base, small, medium, large
  language: "en"
  sample_rate: 16000
  chunk_size: 1024
```

### Vision Pipeline Configuration
Edit `~/vla_ws/src/vla_examples/config/vision_config.yaml`:
```yaml
vision_pipeline:
  object_detection:
    model: "isaac_ros_detectnet"
    confidence_threshold: 0.7
    max_objects: 10
  camera_topics:
    rgb: "/camera/rgb/image_rect_color"
    depth: "/camera/depth/image_rect_raw"
```

### Planning Configuration
Edit `~/vla_ws/src/vla_examples/config/planning_config.yaml`:
```yaml
planning:
  llm_model: "gpt-4"  # or other compatible LLM
  max_plan_steps: 20
  task_timeout: 30.0  # seconds
  safety_constraints:
    max_velocity: 1.0
    max_force: 50.0
```

## Testing the System

### 1. Unit Tests
```bash
# Run Python unit tests
cd ~/vla_ws
source install/setup.bash
python3 -m pytest src/vla_examples/tests/
```

### 2. Integration Tests
```bash
# Run ROS 2 integration tests
colcon test --packages-select vla_examples
colcon test-result --all
```

### 3. Simulation Tests
```bash
# Run simulation-based tests
ros2 launch vla_examples test_scenarios.launch.py
```

## Troubleshooting

### Common Issues

1. **Audio Input Not Working**
   - Check microphone permissions
   - Verify audio device in configuration
   - Test with `arecord -d 3 test.wav`

2. **Vision Pipeline Not Detecting Objects**
   - Verify camera topics are publishing
   - Check lighting conditions in simulation
   - Adjust detection confidence thresholds

3. **LLM Connection Issues**
   - Verify API key configuration
   - Check network connectivity
   - Ensure rate limits are not exceeded

### Performance Optimization

- Reduce simulation complexity if FPS is low
- Use smaller Whisper models for faster processing
- Adjust detection confidence thresholds for performance
- Limit number of tracked objects in vision pipeline