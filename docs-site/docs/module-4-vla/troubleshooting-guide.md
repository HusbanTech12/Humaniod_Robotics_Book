# Troubleshooting Guide for VLA Module

## Overview

This guide provides solutions to common issues encountered when setting up and running the Vision-Language-Action (VLA) system. Use this guide to diagnose and resolve problems with voice processing, cognitive planning, vision grounding, and action execution components.

## Common Issues and Solutions

### 1. Voice Processing Issues

#### Issue: Whisper service not responding
**Symptoms**: Voice commands are not being processed or converted to text.

**Solutions**:
- Verify OpenAI API key is set in environment variables
- Check network connectivity to OpenAI services
- Ensure Whisper model is properly downloaded and cached
- Verify audio input device permissions and availability

**Commands to diagnose**:
```bash
# Check if voice processing node is running
ros2 node list | grep voice

# Check voice processing topics
ros2 topic list | grep -i voice

# Check voice processing logs
ros2 run backend/vla_integration nodes/voice_processor.py
```

#### Issue: Poor voice recognition accuracy
**Symptoms**: Commands are misinterpreted or not recognized.

**Solutions**:
- Ensure quiet environment with minimal background noise
- Check microphone quality and positioning
- Adjust audio input gain settings in `voice_config.yaml`
- Verify language setting matches spoken command language

### 2. Cognitive Planning Issues

#### Issue: LLM service timeout
**Symptoms**: Action plans are not generated or take too long to generate.

**Solutions**:
- Verify OpenAI API key has sufficient quota
- Check network connectivity and latency
- Adjust timeout settings in `llm_planner.py`
- Verify prompt formatting and complexity

**Commands to diagnose**:
```bash
# Check LLM planner status
ros2 service list | grep generate_plan

# Test plan generation service
ros2 service call /vla/generate_plan vla_integration/srv/GenerateActionPlan "{
  command: 'test command'
}"
```

#### Issue: Invalid action plans
**Symptoms**: Generated plans contain invalid task types or parameters.

**Solutions**:
- Review and update LLM prompt templates
- Verify action plan schema validation
- Check for proper JSON formatting in LLM responses
- Ensure task type constraints are properly enforced

### 3. Vision Processing Issues

#### Issue: Object detection failures
**Symptoms**: Objects are not detected or localization fails.

**Solutions**:
- Verify Isaac Sim environment is properly configured
- Check camera topics are publishing data
- Ensure Isaac ROS packages are installed and configured
- Verify detection thresholds in `object_detection_config.yaml`

**Commands to diagnose**:
```bash
# Check camera topics
ros2 topic list | grep camera

# Verify Isaac ROS nodes
ros2 node list | grep isaac

# Test vision service
ros2 service call /vla/vision/localize_object vla_integration/srv/LocalizeObject "{
  object_description: 'cup'
}"
```

#### Issue: Poor localization accuracy
**Symptoms**: Objects detected but positioned inaccurately in 3D space.

**Solutions**:
- Calibrate camera intrinsic and extrinsic parameters
- Verify depth sensor data quality
- Adjust spatial accuracy thresholds
- Check coordinate frame transformations

### 4. Action Execution Issues

#### Issue: Task execution failures
**Symptoms**: Tasks in action plans fail to execute or complete.

**Solutions**:
- Verify robot simulation environment is running
- Check joint limits and constraints
- Ensure navigation stack is available
- Verify manipulation capabilities

**Commands to diagnose**:
```bash
# Check action execution status
ros2 service call /vla/actions/execute_plan vla_integration/srv/ExecutePlan "{
  plan: {
    plan_id: 'test_plan',
    tasks: []
  }
}"

# Monitor execution status
ros2 topic echo /vla/actions/execution_status
```

#### Issue: Safety validation blocking execution
**Symptoms**: Plans are rejected by safety validation.

**Solutions**:
- Review safety constraint settings
- Check environmental hazard detection
- Verify robot state and capabilities
- Adjust safety thresholds appropriately

### 5. System Integration Issues

#### Issue: Pipeline timeouts
**Symptoms**: End-to-end VLA pipeline fails with timeout errors.

**Solutions**:
- Increase timeout values in configuration files
- Check for blocking operations in pipeline
- Verify all required services are available
- Monitor system resource usage

**Commands to diagnose**:
```bash
# Check all VLA services
ros2 service list | grep vla

# Monitor pipeline status
ros2 topic echo /vla/orchestration_status

# Check system resources
htop
nvidia-smi  # if using GPU
```

#### Issue: Message synchronization problems
**Symptoms**: Components receive outdated or mismatched data.

**Solutions**:
- Verify QoS profiles are properly configured
- Check message timestamps and sequence numbers
- Adjust buffer sizes for message queues
- Synchronize clock sources if needed

## Performance Optimization

### Memory Usage
- Monitor memory consumption with `htop`
- Reduce history buffer sizes in configuration
- Implement proper garbage collection for message queues

### Processing Speed
- Use GPU acceleration for vision processing
- Optimize LLM prompt complexity
- Implement caching for repeated operations
- Adjust processing rates in configuration files

### Network Communication
- Use appropriate QoS settings for real-time requirements
- Minimize message size through efficient serialization
- Implement message compression if needed

## Debugging Strategies

### Enable Verbose Logging
```bash
# Set logging level to debug
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=DEBUG

# Or set for specific nodes
ros2 run backend/vla_integration nodes/vla_orchestrator.py --ros-args --log-level debug
```

### Monitor System Performance
```bash
# Monitor CPU and memory usage
ros2 run performance_test performance_test_node

# Monitor ROS 2 communication
ros2 topic hz /vla/action_plan
```

### Test Individual Components
```bash
# Test voice processing separately
ros2 launch backend/vla_integration/launch/voice_processing.launch.py

# Test vision processing separately
ros2 launch backend/vla_integration/launch/vision_pipeline.launch.py

# Test action execution separately
ros2 launch backend/vla_integration/launch/action_execution.launch.py
```

## Common Error Messages

### "Service not available"
- Ensure the target service node is running
- Check for network connectivity issues
- Verify service names are correct

### "Message queue overflow"
- Increase queue size in publisher/subscriber configuration
- Reduce message publishing rate
- Implement message filtering

### "Insufficient permissions"
- Verify ROS 2 domain settings
- Check user permissions for required resources
- Ensure proper authentication for external services

## Support Resources

For additional help:
- Check ROS 2 documentation for communication issues
- Review Isaac ROS documentation for vision processing
- Consult OpenAI documentation for API issues
- Examine system logs in `/tmp/` or ROS 2 log directory