# ActionPlan.msg
# Message definition for action plans in the VLA system

std_msgs/Header header
string plan_id
string command_id
Task[] tasks
builtin_interfaces/Time created_at
builtin_interfaces/Time updated_at
string status  # planned, executing, completed, failed
string[] dependencies