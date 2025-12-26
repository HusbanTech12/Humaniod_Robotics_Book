# Task.msg
# Message definition for individual tasks in the VLA system

string task_id
string type  # navigate, grasp, detect, approach, manipulate, etc.
string description
KeyValue[] parameters
string[] requirements
string success_criteria
int8 priority  # 1 (highest) to 10 (lowest)
string status  # pending, executing, completed, failed
string[] dependencies

# Key-value pair for parameters
message KeyValue
{
  string key
  string value
}