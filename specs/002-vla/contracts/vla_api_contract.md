# API Contract: Vision-Language-Action (VLA) Module

## Voice Command Processing Service

### `/voice/process_command`
- **Method**: POST
- **Purpose**: Process a voice command and generate an action plan
- **Request**:
  - Content-Type: `multipart/form-data` or `application/json`
  - Body: Audio data or transcribed text
  - Example:
    ```json
    {
      "audio_data": "base64_encoded_audio",
      "command_text": "Pick up the red cup",
      "timestamp": "2025-12-25T10:00:00Z"
    }
    ```

- **Response**:
  - Success: 200 OK
    ```json
    {
      "command_id": "cmd_12345",
      "status": "processing",
      "estimated_completion": "2.5",
      "action_plan": {
        "plan_id": "plan_67890",
        "tasks": [
          {
            "task_id": "task_001",
            "type": "detect_object",
            "parameters": {
              "object_type": "cup",
              "color": "red"
            },
            "priority": 1
          },
          {
            "task_id": "task_002",
            "type": "navigate",
            "parameters": {
              "target_position": {"x": 1.5, "y": 2.0, "z": 0.0}
            },
            "priority": 2
          }
        ]
      }
    }
    ```
  - Error: 400 Bad Request (invalid input), 500 Internal Server Error

## Vision Grounding Service

### `/vision/localize_object`
- **Method**: POST
- **Purpose**: Localize objects in the environment based on description
- **Request**:
  ```json
  {
    "object_description": "red cup",
    "camera_view": "front_camera",
    "search_area": {
      "x_range": [-2.0, 2.0],
      "y_range": [-2.0, 2.0],
      "z_range": [0.0, 1.5]
    }
  }
  ```

- **Response**:
  ```json
  {
    "detections": [
      {
        "object_id": "obj_001",
        "class": "cup",
        "color": "red",
        "position": {"x": 1.2, "y": 0.8, "z": 0.95},
        "confidence": 0.89,
        "bounding_box": {
          "x": 120,
          "y": 85,
          "width": 45,
          "height": 60
        }
      }
    ],
    "timestamp": "2025-12-25T10:00:05Z"
  }
  ```

## Action Execution Service

### `/actions/execute_plan`
- **Method**: POST
- **Purpose**: Execute a structured action plan
- **Request**:
  ```json
  {
    "plan_id": "plan_67890",
    "tasks": [
      {
        "task_id": "task_001",
        "type": "detect_object",
        "parameters": {
          "object_type": "cup",
          "color": "red"
        }
      }
    ],
    "execution_context": {
      "robot_id": "humanoid_001",
      "environment_id": "kitchen_01"
    }
  }
  ```

- **Response**:
  ```json
  {
    "execution_id": "exec_54321",
    "status": "started",
    "estimated_duration": 15.0,
    "task_sequence": [
      {
        "task_id": "task_001",
        "status": "pending",
        "estimated_time": 5.0
      }
    ]
  }
  ```

### `/actions/execution_status/{execution_id}`
- **Method**: GET
- **Purpose**: Get status of an executing action plan
- **Response**:
  ```json
  {
    "execution_id": "exec_54321",
    "overall_status": "executing",
    "completed_tasks": 1,
    "total_tasks": 3,
    "current_task": {
      "task_id": "task_002",
      "type": "navigate",
      "status": "executing"
    },
    "progress": 0.33,
    "timestamp": "2025-12-25T10:00:10Z"
  }
  ```

## LLM Planning Service

### `/planning/generate_plan`
- **Method**: POST
- **Purpose**: Generate structured action plan from natural language command
- **Request**:
  ```json
  {
    "command": "Clean the table by picking up all cups and placing them in the sink",
    "context": {
      "environment": "kitchen",
      "robot_capabilities": ["navigation", "manipulation", "object_detection"],
      "available_objects": ["cups", "plates", "forks"]
    }
  }
  ```

- **Response**:
  ```json
  {
    "plan_id": "plan_11223",
    "command": "Clean the table by picking up all cups and placing them in the sink",
    "tasks": [
      {
        "task_id": "task_001",
        "type": "detect_objects",
        "description": "Find all cups on the table",
        "parameters": {
          "object_type": "cup",
          "search_location": "table"
        },
        "dependencies": []
      },
      {
        "task_id": "task_002",
        "type": "navigate",
        "description": "Move to first cup location",
        "parameters": {
          "target_location": {"x": 1.0, "y": 0.5, "z": 0.0}
        },
        "dependencies": ["task_001"]
      },
      {
        "task_id": "task_003",
        "type": "grasp_object",
        "description": "Pick up the cup",
        "parameters": {
          "object_id": "cup_001"
        },
        "dependencies": ["task_002"]
      }
    ],
    "estimated_steps": 12,
    "confidence": 0.92
  }
  ```

## Safety and Validation Service

### `/safety/validate_action`
- **Method**: POST
- **Purpose**: Validate an action plan for safety constraints
- **Request**:
  ```json
  {
    "action_plan": {
      "tasks": [
        {
          "task_id": "task_001",
          "type": "navigate",
          "parameters": {
            "target_position": {"x": 50.0, "y": 50.0, "z": 0.0}
          }
        }
      ]
    },
    "robot_state": {
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "battery_level": 0.85
    }
  }
  ```

- **Response**:
  ```json
  {
    "is_safe": false,
    "violations": [
      {
        "type": "boundary_violation",
        "description": "Target position outside safe operating area",
        "severity": "high"
      }
    ],
    "suggested_alternatives": [
      {
        "task_id": "task_001",
        "modified_parameters": {
          "target_position": {"x": 2.0, "y": 1.5, "z": 0.0}
        }
      }
    ]
  }
  ```

## Message Formats

### VoiceCommand Message
```json
{
  "header": {
    "timestamp": "2025-12-25T10:00:00Z",
    "sequence": 123
  },
  "command_id": "cmd_12345",
  "audio_data": "base64...",
  "transcribed_text": "Pick up the red cup",
  "confidence": 0.95,
  "language": "en"
}
```

### ActionPlan Message
```json
{
  "header": {
    "timestamp": "2025-12-25T10:00:02Z",
    "sequence": 124
  },
  "plan_id": "plan_67890",
  "command_id": "cmd_12345",
  "tasks": [
    {
      "task_id": "task_001",
      "type": "detect_object",
      "parameters": {},
      "dependencies": [],
      "timeout": 10.0
    }
  ],
  "status": "pending",
  "created_by": "llm_planner"
}
```

### ExecutionStatus Message
```json
{
  "header": {
    "timestamp": "2025-12-25T10:00:05Z",
    "sequence": 125
  },
  "execution_id": "exec_54321",
  "plan_id": "plan_67890",
  "current_task_id": "task_001",
  "status": "executing",
  "progress": 0.25,
  "error": null
}
```