# VLA Quickstart Guide

## Overview

This quickstart guide will help you set up and run the Vision-Language-Action (VLA) system for humanoid robots. By the end of this guide, you'll have a complete VLA system running that can process voice commands and execute multi-step tasks.

## Prerequisites

Before starting, ensure you have:

- ROS 2 Humble Hawksbill installed
- Isaac Sim and Isaac ROS packages
- Python 3.8+ with required dependencies
- OpenAI API key (for Whisper and LLM services)
- Compatible hardware (GPU recommended for Isaac ROS)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd humanoid-robotic-book
```

### 2. Install Dependencies

```bash
# Install ROS 2 dependencies
sudo apt update
sudo apt install ros-humble-desktop

# Install Python dependencies
pip install openai numpy scipy

# Install Isaac ROS packages
# Follow Isaac ROS installation guide
```

### 3. Set Up Environment

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Source workspace
source install/setup.bash
```

## Configuration

### 1. Set API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Configure Components

The system is configured through YAML files in `backend/vla_integration/config/`:

- `voice_config.yaml` - Voice processing settings
- `object_detection_config.yaml` - Vision processing settings
- `isaac_sim_config.yaml` - Isaac Sim environment settings
- `vla_pipeline_config.yaml` - Pipeline configuration

## Running the System

### 1. Launch Isaac Sim Environment

```bash
# Launch Isaac Sim with humanoid robot
ros2 launch backend/vla_integration/launch/isaac_sim.launch.py
```

### 2. Start VLA Components

The system can be launched in parts or all together:

#### Launch Complete System:

```bash
ros2 launch backend/vla_integration/launch/vla_complete_system.launch.py
```

#### Launch Components Separately:

```bash
# Voice processing pipeline
ros2 launch backend/vla_integration/launch/voice_processing.launch.py

# Vision processing pipeline
ros2 launch backend/vla_integration/launch/vision_pipeline.launch.py

# Action execution
ros2 launch backend/vla_integration/launch/action_execution.launch.py

# Complete VLA pipeline
ros2 launch backend/vla_integration/launch/vla_voice_pipeline.launch.py
```

## Basic Usage

### 1. Voice Command Processing

Once the system is running, you can send voice commands:

```bash
# Send a voice command via service call
ros2 service call /voice/process_command vla_integration/srv/ProcessVoiceCommand "{
  audio_file_path: '/path/to/audio.wav'
}"
```

### 2. Direct Plan Generation

You can also generate action plans directly:

```bash
# Generate a plan from text command
ros2 service call /vla/generate_plan vla_integration/srv/GenerateActionPlan "{
  command: 'Navigate to the kitchen and find a red cup'
}"
```

### 3. Execute Action Plans

Execute a generated plan:

```bash
# Execute an action plan
ros2 service call /actions/execute_plan vla_integration/srv/ExecutePlan "{
  plan: {
    plan_id: 'example_plan',
    tasks: [
      {
        task_id: 'task_1',
        type: 'navigate',
        parameters: [
          {key: 'target_location', value: 'kitchen'}
        ],
        description: 'Navigate to kitchen'
      }
    ]
  }
}"
```

## Example Scenarios

### Scenario 1: Simple Navigation

```bash
# Command: "Go to the kitchen"
ros2 service call /voice/process_command vla_integration/srv/ProcessVoiceCommand "{
  command_text: 'Go to the kitchen'
}"
```

### Scenario 2: Object Detection

```bash
# Command: "Find the red cup"
ros2 service call /voice/process_command vla_integration/srv/ProcessVoiceCommand "{
  command_text: 'Find the red cup'
}"
```

### Scenario 3: Multi-step Task

```bash
# Command: "Go to the kitchen, find the red cup, and bring it to the table"
ros2 service call /voice/process_command vla_integration/srv/ProcessVoiceCommand "{
  command_text: 'Go to the kitchen, find the red cup, and bring it to the table'
}"
```

## Monitoring

### Check System Status

Monitor the system through various topics:

```bash
# View execution status
ros2 topic echo /vla/execution_status

# View detected objects
ros2 topic echo /vla/localized_objects

# View action plans
ros2 topic echo /vla/action_plan
```

### View System Logs

```bash
# View logs for specific nodes
ros2 run backend/vla_integration nodes/voice_processor.py
ros2 run backend/vla_integration nodes/llm_planner.py
ros2 run backend/vla_integration nodes/action_executor.py
```

## Troubleshooting

### Common Issues

#### 1. Voice Processing Errors
- Check microphone access and permissions
- Verify OpenAI API key is set
- Ensure Whisper model is downloaded

#### 2. Vision Processing Errors
- Verify Isaac Sim is running
- Check camera topics are publishing
- Ensure Isaac ROS packages are installed

#### 3. Action Execution Errors
- Verify robot is properly simulated
- Check joint limits and constraints
- Ensure navigation stack is available

### Debugging Commands

```bash
# Check all active nodes
ros2 node list

# Check all active topics
ros2 topic list

# Check all active services
ros2 service list

# Get detailed information about a specific node
ros2 node info <node_name>
```

## Development Workflow

### 1. Testing Individual Components

Test components in isolation:

```bash
# Test voice processing only
ros2 run backend/vla_integration nodes/voice_processor.py

# Test vision processing only
ros2 run backend/vla_integration nodes/vision_processor.py

# Test action execution only
ros2 run backend/vla_integration nodes/action_executor.py
```

### 2. Integration Testing

Test the complete pipeline:

```bash
# Launch complete system
ros2 launch backend/vla_integration/launch/vla_complete_system.launch.py

# Send end-to-end commands
ros2 service call /voice/process_command vla_integration/srv/ProcessVoiceCommand "{
  command_text: 'Perform a simple task'
}"
```

## Performance Tuning

### Configuration Options

Adjust performance in configuration files:

- Increase/decrease detection thresholds for speed/accuracy trade-off
- Adjust planning complexity based on computational resources
- Tune safety margins based on operational requirements

### Resource Management

Monitor system resources:

```bash
# Monitor CPU and memory usage
htop

# Monitor GPU usage (if using GPU acceleration)
nvidia-smi

# Monitor ROS 2 communication
ros2 topic hz /vla/action_plan
```

## Next Steps

After successfully running the basic VLA system:

1. Explore more complex multi-step tasks
2. Customize the system for your specific use case
3. Add new task types and capabilities
4. Integrate with additional sensors or actuators
5. Enhance safety and reliability features

## Support

For additional support:

- Check the detailed documentation for each component
- Review the architecture documentation
- Examine the example implementations
- Consult the troubleshooting guides

This quickstart guide provides the foundation to run and understand the VLA system. The complete documentation contains more detailed information about each component and advanced usage scenarios.