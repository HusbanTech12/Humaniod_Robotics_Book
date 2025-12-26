# ExecutionStatus.msg
# Message definition for execution status in the VLA system

std_msgs/Header header
string execution_id
string plan_id
string current_task_id
string overall_status  # running, completed, error, paused
int32 completed_tasks
int32 total_tasks
float32 progress  # 0.0 to 1.0
string error
builtin_interfaces/Time start_time
builtin_interfaces/Time end_time