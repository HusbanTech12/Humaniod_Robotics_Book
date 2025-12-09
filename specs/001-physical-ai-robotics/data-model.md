# Data Model: The Robotic Nervous System (ROS 2)

## Key Entities

### Humanoid Robot Model
- **Attributes**:
  - joints: List of joint definitions (name, type, limits, position, velocity, effort)
  - links: List of link definitions (name, geometry, material, inertial properties)
  - sensors: List of sensor definitions (type, position, orientation, parameters)
  - actuators: List of actuator definitions (type, joint, control parameters)
- **Relationships**: Contains multiple joints, links, sensors, and actuators
- **Validation**: Must form valid kinematic chain, pass URDF validation

### ROS 2 Node
- **Attributes**:
  - name: Unique identifier for the node
  - namespace: Optional namespace for organization
  - parameters: Configuration values for the node
  - publishers: List of topics the node publishes to
  - subscribers: List of topics the node subscribes to
  - services: List of services the node provides
  - clients: List of services the node calls
  - actions: List of action servers/clients
- **Relationships**: Communicates with other nodes via topics, services, and actions
- **Validation**: Must have unique name within namespace, valid parameter types

### ROS 2 Message
- **Attributes**:
  - type: Message type (e.g., std_msgs/String, sensor_msgs/JointState)
  - fields: Data fields contained in the message
  - timestamp: When the message was created
  - header: Metadata including frame_id and timestamp
- **Relationships**: Exchanged between nodes via topics, services, and actions
- **Validation**: Must conform to message definition schema

### Launch File Configuration
- **Attributes**:
  - nodes: List of nodes to launch
  - parameters: Global or node-specific parameters
  - remappings: Topic/service remapping rules
  - conditions: Conditional launch logic
  - event_handlers: Actions triggered by node events
- **Relationships**: Configures multiple nodes and their interactions
- **Validation**: Must be valid XML/YAML according to launch format

### AI Agent Interface
- **Attributes**:
  - input_topics: Topics from which the agent receives sensor data
  - output_topics: Topics to which the agent publishes control commands
  - service_interfaces: Services the agent can call for specific actions
  - action_interfaces: Actions the agent can send goals for
  - control_frequency: How often the agent updates commands
- **Relationships**: Connects AI logic to ROS 2 communication layer
- **Validation**: Must match expected message types and timing requirements

## State Transitions

### Node Lifecycle
- Unconfigured → Inactive: Node created but not yet configured
- Inactive → Active: Node configured and ready to process data
- Active → Inactive: Node deactivated but still exists
- Active/Inactive → Finalized: Node destroyed and cleaned up

### Action Goal States
- Pending → Active: Goal accepted and execution started
- Active → Succeeded: Goal completed successfully
- Active → Canceled: Goal canceled by client
- Active → Aborted: Goal failed during execution
- Pending → Rejected: Goal rejected by action server

## Communication Patterns

### Sensor Data Flow
- Sensors → Sensor Processing Nodes → AI Agent → Control Nodes → Actuators
- Uses topics for real-time streaming with appropriate QoS settings

### Configuration Flow
- External Configuration → Services → Node Parameters
- Uses services for reliable one-time configuration changes

### Behavior Execution
- AI Behavior Planner → Action Goals → Behavior Execution Nodes → Results
- Uses actions for multi-step behaviors with feedback