# VoiceCommand.msg
# Message definition for voice commands in the VLA system

std_msgs/Header header
string command_id
string audio_data  # base64 encoded audio data
string transcribed_text
string language
float32 confidence
string intent
time timestamp