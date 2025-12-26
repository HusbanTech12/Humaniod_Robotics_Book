#!/usr/bin/env python3

"""
Voice Processor Node for Vision-Language-Action (VLA) Module
Handles speech recognition and voice command processing
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import speech_recognition as sr
import whisper
import torch
import numpy as np
from threading import Thread
import queue
import time

# Import custom message types
from vla_integration.msg import VoiceCommand
from std_msgs.msg import String


class VoiceProcessorNode(Node):
    """
    ROS 2 Node for processing voice commands using OpenAI Whisper
    """

    def __init__(self):
        super().__init__('voice_processor')

        # Declare parameters
        self.declare_parameter('model', 'base')
        self.declare_parameter('language', 'en')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_size', 1024)
        self.declare_parameter('energy_threshold', 300)
        self.declare_parameter('pause_threshold', 0.8)
        self.declare_parameter('timeout', 5.0)

        # Get parameters
        self.model_name = self.get_parameter('model').value
        self.language = self.get_parameter('language').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.chunk_size = self.get_parameter('chunk_size').value
        self.energy_threshold = self.get_parameter('energy_threshold').value
        self.pause_threshold = self.pause_threshold = self.get_parameter('pause_threshold').value
        self.timeout = self.get_parameter('timeout').value

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = self.energy_threshold
        self.recognizer.pause_threshold = self.pause_threshold

        # Initialize microphone
        self.microphone = sr.Microphone(sample_rate=self.sample_rate)

        # Load Whisper model
        self.get_logger().info(f'Loading Whisper model: {self.model_name}')
        self.whisper_model = whisper.load_model(self.model_name)

        # Create publishers and subscribers
        self.voice_cmd_pub = self.create_publisher(VoiceCommand, 'vla/voice/command', 10)
        self.transcription_pub = self.create_publisher(String, 'vla/voice/transcription', 10)

        # Create timer for continuous listening
        self.listen_timer = self.create_timer(0.1, self.listen_callback)

        self.get_logger().info('Voice Processor Node initialized')

    def listen_callback(self):
        """Callback to continuously listen for voice commands"""
        try:
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)

                # Listen for audio
                self.get_logger().info('Listening for voice command...')
                audio = self.recognizer.listen(source, timeout=self.timeout, phrase_time_limit=10)

                # Process audio with Whisper
                self.get_logger().info('Processing audio with Whisper...')
                result = self.whisper_model.transcribe(
                    audio.get_wav_data(),
                    language=self.language,
                    temperature=0.0
                )

                # Create and publish VoiceCommand message
                cmd_msg = VoiceCommand()
                cmd_msg.header.stamp = self.get_clock().now().to_msg()
                cmd_msg.header.frame_id = 'voice_processor'
                cmd_msg.command_id = f'cmd_{int(time.time())}'
                cmd_msg.transcribed_text = result['text']
                cmd_msg.language = self.language
                cmd_msg.confidence = min(1.0, result.get('avg_logprob', -0.5) + 1.0)  # Normalize confidence
                cmd_msg.timestamp = self.get_clock().now().to_msg()

                # Publish the voice command
                self.voice_cmd_pub.publish(cmd_msg)
                self.transcription_pub.publish(String(data=result['text']))

                self.get_logger().info(f'Voice command processed: "{result["text"]}"')

        except sr.WaitTimeoutError:
            # No speech detected, continue listening
            pass
        except Exception as e:
            self.get_logger().error(f'Error in voice processing: {str(e)}')

    def process_audio_data(self, audio_data):
        """
        Process raw audio data through Whisper model
        """
        try:
            # Convert audio data to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Process with Whisper
            result = self.whisper_model.transcribe(
                audio_np,
                language=self.language,
                temperature=0.0
            )

            return result['text'], min(1.0, result.get('avg_logprob', -0.5) + 1.0)
        except Exception as e:
            self.get_logger().error(f'Error processing audio data: {str(e)}')
            return "", 0.0


def main(args=None):
    rclpy.init(args=args)

    voice_processor = VoiceProcessorNode()

    try:
        rclpy.spin(voice_processor)
    except KeyboardInterrupt:
        pass
    finally:
        voice_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()