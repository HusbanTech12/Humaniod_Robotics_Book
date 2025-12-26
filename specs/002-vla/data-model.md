# Data Model: Vision-Language-Action (VLA) Module

## Core Entities

### VoiceCommand
- **id**: Unique identifier for the command
- **raw_audio**: Audio data from user input
- **transcribed_text**: Text from speech-to-text processing
- **intent**: Parsed intent from language understanding
- **timestamp**: When the command was received
- **confidence**: Confidence score of transcription
- **status**: Processing status (received, processing, completed, failed)

### ActionPlan
- **id**: Unique identifier for the plan
- **command_id**: Reference to originating VoiceCommand
- **tasks**: Array of structured task objects
- **parameters**: Execution parameters for the plan
- **dependencies**: Task dependency relationships
- **status**: Execution status (planned, executing, completed, failed)
- **created_at**: Timestamp of plan creation
- **updated_at**: Last update timestamp

### Task
- **id**: Unique identifier for the task
- **type**: Task type (navigate, grasp, detect, etc.)
- **description**: Human-readable description
- **parameters**: Task-specific parameters
- **requirements**: Prerequisites for task execution
- **success_criteria**: Conditions for task completion
- **priority**: Execution priority level
- **status**: Task status (pending, executing, completed, failed)

### VisionData
- **id**: Unique identifier for vision data
- **timestamp**: When data was captured
- **object_detections**: Array of detected objects
- **spatial_coordinates**: 3D coordinates of objects
- **confidence_scores**: Confidence for each detection
- **scene_context**: Environmental context information
- **source_camera**: Camera/feed that generated data

### ObjectDetection
- **id**: Unique identifier for detection
- **class_name**: Object class (cup, chair, etc.)
- **bounding_box**: 2D coordinates of bounding box
- **position_3d**: 3D world coordinates
- **confidence**: Detection confidence score
- **tracking_id**: ID for object tracking across frames

### RobotState
- **id**: Unique identifier for state snapshot
- **position**: Current position (x, y, z)
- **orientation**: Current orientation (quaternion)
- **joint_states**: Current joint positions and velocities
- **gripper_state**: Current gripper status
- **battery_level**: Current battery percentage
- **timestamp**: When state was recorded

### TaskExecutionContext
- **id**: Unique identifier for execution context
- **plan_id**: Reference to the ActionPlan
- **current_task**: Currently executing task
- **task_history**: Completed tasks in sequence
- **errors**: Any errors encountered during execution
- **recovery_attempts**: Number of recovery attempts
- **execution_state**: Current state of execution
- **start_time**: When execution began
- **end_time**: When execution completed (if applicable)

## Relationships

- VoiceCommand → ActionPlan (one-to-many): One command can generate multiple plan variations
- ActionPlan → Task (one-to-many): One plan contains multiple tasks
- Task → TaskExecutionContext (one-to-one): Each task has execution context
- VisionData → ObjectDetection (one-to-many): Vision data contains multiple detections
- RobotState → TaskExecutionContext (many-to-one): Multiple robot states during execution

## Validation Rules

### VoiceCommand Validation
- transcribed_text must not be empty
- confidence must be between 0.0 and 1.0
- timestamp must be within the last 10 seconds

### ActionPlan Validation
- tasks array must not be empty
- all tasks must have valid types from predefined set
- dependencies must reference valid task IDs
- parameters must match expected schema for task type

### Task Validation
- type must be from predefined set (navigate, grasp, detect, approach, manipulate)
- parameters must be valid for the task type
- priority must be between 1 (highest) and 10 (lowest)
- success_criteria must be defined

### ObjectDetection Validation
- class_name must be from recognized object classes
- confidence must be between 0.0 and 1.0
- position_3d must have valid coordinates within environment bounds

## State Transitions

### ActionPlan States
- planned → executing: When plan execution begins
- executing → completed: When all tasks complete successfully
- executing → failed: When a task fails and no recovery possible
- executing → paused: When external intervention occurs

### Task States
- pending → executing: When task execution begins
- executing → completed: When task success criteria met
- executing → failed: When task fails to complete
- failed → retrying: When recovery attempt initiated

### TaskExecutionContext States
- initialized → running: When first task starts
- running → completed: When final task completes
- running → error: When unrecoverable error occurs
- error → recovery: When recovery procedure initiated