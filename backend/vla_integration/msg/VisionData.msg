# VisionData.msg
# Message definition for vision data in the VLA system

std_msgs/Header header
string data_id
ObjectDetection[] object_detections
geometry_msgs/PointStamped[] spatial_coordinates
float32[] confidence_scores
string scene_context
string source_camera
time timestamp