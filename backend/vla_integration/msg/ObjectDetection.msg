# ObjectDetection.msg
# Message definition for object detections in the VLA system

string object_id
string class_name
int32[] bounding_box  # [x, y, width, height]
geometry_msgs/Point position_3d
float32 confidence
int32 tracking_id
string[] attributes  # color, size, etc.