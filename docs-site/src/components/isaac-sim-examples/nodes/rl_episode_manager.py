#!/usr/bin/env python3
# rl_episode_manager.py

"""
Isaac Lab RL Episode Management System for Humanoid Robot Control

This module manages RL training episodes, including episode tracking,
statistics collection, logging, and curriculum learning progression.
"""

import os
import json
import pickle
import numpy as np
import torch
import datetime
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import csv
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class EpisodeInfo:
    """
    Data class to store information about a single episode
    """
    episode_id: int
    start_time: float
    end_time: float
    total_reward: float
    episode_length: int
    success: bool
    termination_reason: str
    final_position: Tuple[float, float, float]
    initial_position: Tuple[float, float, float]
    target_position: Optional[Tuple[float, float, float]] = None
    metrics: Optional[Dict[str, float]] = None


class EpisodeManager:
    """
    Manages RL training episodes, statistics, and logging
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the episode manager with configuration

        Args:
            config: Configuration dictionary with episode management parameters
        """
        self.config = config
        self.episode_count = 0
        self.total_timesteps = 0
        self.current_episode_start_time = None

        # Episode statistics
        self.episode_rewards = deque(maxlen=config.get('episode_history_length', 100))
        self.episode_lengths = deque(maxlen=config.get('episode_history_length', 100))
        self.episode_successes = deque(maxlen=config.get('episode_history_length', 100))
        self.episode_times = deque(maxlen=config.get('episode_history_length', 100))

        # Episode info storage
        self.episode_history: List[EpisodeInfo] = []

        # Metrics tracking
        self.metrics_history = defaultdict(lambda: deque(maxlen=config.get('episode_history_length', 100)))

        # Curriculum learning parameters
        self.curriculum_enabled = config.get('curriculum', {}).get('enabled', False)
        self.curriculum_difficulty = config.get('curriculum', {}).get('initial_difficulty', 0.0)
        self.difficulty_threshold = config.get('curriculum', {}).get('difficulty_threshold', 0.8)
        self.difficulty_increment = config.get('curriculum', {}).get('difficulty_increment', 0.01)

        # Logging setup
        self.log_dir = config.get('log_dir', './logs')
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        self.stats_file = os.path.join(self.log_dir, 'episode_stats.json')
        self.csv_file = os.path.join(self.log_dir, 'episode_data.csv')

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize CSV file with headers
        self._init_csv_file()

        # Setup logging
        self.logger = logging.getLogger('RL_Episode_Manager')
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'episode_manager.log'))
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def _init_csv_file(self):
        """
        Initialize the CSV file with appropriate headers
        """
        headers = [
            'episode_id', 'timestamp', 'total_reward', 'episode_length',
            'success', 'duration', 'final_x', 'final_y', 'final_z',
            'initial_x', 'initial_y', 'initial_z', 'difficulty'
        ]

        # Add any additional metrics from config
        if 'tracked_metrics' in self.config:
            headers.extend(self.config['tracked_metrics'])

        with open(self.csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    def start_episode(self, initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> int:
        """
        Start a new episode and return the episode ID

        Args:
            initial_position: Initial position of the robot (x, y, z)

        Returns:
            Episode ID
        """
        self.current_episode_start_time = time.time()
        self.episode_count += 1

        self.logger.info(f"Starting episode {self.episode_count} at position {initial_position}")

        return self.episode_count

    def end_episode(self,
                   total_reward: float,
                   episode_length: int,
                   success: bool,
                   termination_reason: str,
                   final_position: Tuple[float, float, float],
                   initial_position: Tuple[float, float, float],
                   target_position: Optional[Tuple[float, float, float]] = None,
                   metrics: Optional[Dict[str, float]] = None) -> EpisodeInfo:
        """
        End the current episode and record statistics

        Args:
            total_reward: Total reward accumulated during the episode
            episode_length: Number of timesteps in the episode
            success: Whether the episode was successful
            termination_reason: Reason for episode termination
            final_position: Final position of the robot (x, y, z)
            initial_position: Initial position of the robot (x, y, z)
            target_position: Target position if applicable (x, y, z)
            metrics: Additional metrics to track

        Returns:
            EpisodeInfo object containing episode details
        """
        if self.current_episode_start_time is None:
            raise ValueError("No episode is currently running")

        end_time = time.time()
        duration = end_time - self.current_episode_start_time

        # Create episode info
        episode_info = EpisodeInfo(
            episode_id=self.episode_count,
            start_time=self.current_episode_start_time,
            end_time=end_time,
            total_reward=total_reward,
            episode_length=episode_length,
            success=success,
            termination_reason=termination_reason,
            final_position=final_position,
            initial_position=initial_position,
            target_position=target_position,
            metrics=metrics or {}
        )

        # Store episode info
        self.episode_history.append(episode_info)

        # Update statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.episode_successes.append(int(success))
        self.episode_times.append(duration)

        # Update metrics history
        if metrics:
            for key, value in metrics.items():
                self.metrics_history[key].append(value)

        # Update total timesteps
        self.total_timesteps += episode_length

        # Log episode data to CSV
        self._log_episode_to_csv(episode_info)

        # Log to console periodically
        if self.episode_count % 10 == 0:
            self.log_statistics()

        # Update curriculum difficulty if enabled
        if self.curriculum_enabled:
            self._update_curriculum_difficulty()

        self.current_episode_start_time = None

        self.logger.info(
            f"Episode {self.episode_count} ended: "
            f"Reward={total_reward:.2f}, Length={episode_length}, "
            f"Success={success}, Duration={duration:.2f}s"
        )

        return episode_info

    def _log_episode_to_csv(self, episode_info: EpisodeInfo):
        """
        Log episode data to CSV file

        Args:
            episode_info: EpisodeInfo object containing episode details
        """
        row = [
            episode_info.episode_id,
            datetime.datetime.fromtimestamp(episode_info.end_time).isoformat(),
            episode_info.total_reward,
            episode_info.episode_length,
            episode_info.success,
            episode_info.end_time - episode_info.start_time,
            episode_info.final_position[0],
            episode_info.final_position[1],
            episode_info.final_position[2],
            episode_info.initial_position[0],
            episode_info.initial_position[1],
            episode_info.initial_position[2],
            self.curriculum_difficulty
        ]

        # Add additional metrics if available
        if episode_info.metrics:
            for metric_name in self.config.get('tracked_metrics', []):
                row.append(episode_info.metrics.get(metric_name, 0.0))

        with open(self.csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    def log_statistics(self):
        """
        Log current statistics to console and file
        """
        if not self.episode_rewards:
            self.logger.warning("No episodes completed yet")
            return

        # Calculate statistics
        avg_reward = np.mean(self.episode_rewards)
        std_reward = np.std(self.episode_rewards)
        min_reward = np.min(self.episode_rewards)
        max_reward = np.max(self.episode_rewards)

        avg_length = np.mean(self.episode_lengths)
        std_length = np.std(self.episode_lengths)

        success_rate = np.mean(self.episode_successes) if self.episode_successes else 0.0

        avg_time = np.mean(self.episode_times) if self.episode_times else 0.0

        # Log to console
        stats_str = (
            f"\nEpisode Statistics (Last {len(self.episode_rewards)} episodes):\n"
            f"  Episode Count: {self.episode_count}\n"
            f"  Total Timesteps: {self.total_timesteps}\n"
            f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f} "
            f"[Min: {min_reward:.2f}, Max: {max_reward:.2f}]\n"
            f"  Average Length: {avg_length:.2f} ± {std_length:.2f}\n"
            f"  Success Rate: {success_rate:.2%}\n"
            f"  Average Episode Time: {avg_time:.2f}s\n"
            f"  Current Difficulty: {self.curriculum_difficulty:.2f}\n"
        )

        self.logger.info(stats_str)

        # Save statistics to JSON
        stats = {
            'episode_count': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'average_reward': float(avg_reward),
            'std_reward': float(std_reward),
            'min_reward': float(min_reward),
            'max_reward': float(max_reward),
            'average_length': float(avg_length),
            'std_length': float(std_length),
            'success_rate': float(success_rate),
            'average_episode_time': float(avg_time),
            'curriculum_difficulty': self.curriculum_difficulty,
            'timestamp': datetime.datetime.now().isoformat()
        }

        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics

        Returns:
            Dictionary containing current statistics
        """
        if not self.episode_rewards:
            return {}

        return {
            'episode_count': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'average_reward': float(np.mean(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'min_reward': float(np.min(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'average_length': float(np.mean(self.episode_lengths)),
            'success_rate': float(np.mean(self.episode_successes)) if self.episode_successes else 0.0,
            'curriculum_difficulty': self.curriculum_difficulty
        }

    def _update_curriculum_difficulty(self):
        """
        Update curriculum difficulty based on recent performance
        """
        if len(self.episode_rewards) < 10:
            return  # Need enough episodes to calculate meaningful statistics

        # Calculate recent success rate
        recent_successes = list(self.episode_successes)[-10:]
        recent_success_rate = np.mean(recent_successes)

        # Update difficulty based on performance
        if recent_success_rate >= self.difficulty_threshold:
            # Performance is good, increase difficulty
            self.curriculum_difficulty = min(
                1.0,
                self.curriculum_difficulty + self.difficulty_increment
            )
            self.logger.info(f"Increased curriculum difficulty to {self.curriculum_difficulty:.2f}")
        elif recent_success_rate < self.difficulty_threshold * 0.5:
            # Performance is poor, decrease difficulty
            self.curriculum_difficulty = max(
                0.0,
                self.curriculum_difficulty - self.difficulty_increment
            )
            self.logger.info(f"Decreased curriculum difficulty to {self.curriculum_difficulty:.2f}")

    def should_save_checkpoint(self, iteration: int, save_interval: int = 500) -> bool:
        """
        Determine if a checkpoint should be saved

        Args:
            iteration: Current training iteration
            save_interval: How often to save checkpoints

        Returns:
            True if checkpoint should be saved
        """
        return iteration % save_interval == 0

    def save_checkpoint(self, model_state: Dict[str, Any], iteration: int):
        """
        Save a training checkpoint

        Args:
            model_state: State dictionary of the model to save
            iteration: Current training iteration
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{iteration:06d}.pth"
        )

        # Add episode manager state to the checkpoint
        checkpoint_data = {
            'model_state': model_state,
            'episode_manager_state': self.get_state(),
            'iteration': iteration,
            'episode_count': self.episode_count,
            'total_timesteps': self.total_timesteps
        }

        torch.save(checkpoint_data, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a training checkpoint

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Dictionary containing model state and metadata
        """
        checkpoint_data = torch.load(checkpoint_path)

        # Restore episode manager state
        if 'episode_manager_state' in checkpoint_data:
            self.set_state(checkpoint_data['episode_manager_state'])

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the episode manager for checkpointing

        Returns:
            Dictionary containing the state
        """
        return {
            'episode_count': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'episode_successes': list(self.episode_successes),
            'episode_times': list(self.episode_times),
            'episode_history': [asdict(ep_info) for ep_info in self.episode_history],
            'curriculum_difficulty': self.curriculum_difficulty,
            'metrics_history': {k: list(v) for k, v in self.metrics_history.items()}
        }

    def set_state(self, state: Dict[str, Any]):
        """
        Set the state of the episode manager from a checkpoint

        Args:
            state: Dictionary containing the state to restore
        """
        self.episode_count = state['episode_count']
        self.total_timesteps = state['total_timesteps']
        self.episode_rewards = deque(state['episode_rewards'], maxlen=self.config.get('episode_history_length', 100))
        self.episode_lengths = deque(state['episode_lengths'], maxlen=self.config.get('episode_history_length', 100))
        self.episode_successes = deque(state['episode_successes'], maxlen=self.config.get('episode_history_length', 100))
        self.episode_times = deque(state['episode_times'], maxlen=self.config.get('episode_history_length', 100))

        # Restore episode history
        self.episode_history = []
        for ep_data in state['episode_history']:
            ep_info = EpisodeInfo(**ep_data)
            self.episode_history.append(ep_info)

        self.curriculum_difficulty = state['curriculum_difficulty']

        # Restore metrics history
        for k, v in state['metrics_history'].items():
            self.metrics_history[k] = deque(v, maxlen=self.config.get('episode_history_length', 100))

    def reset(self):
        """
        Reset the episode manager to initial state
        """
        self.episode_count = 0
        self.total_timesteps = 0
        self.current_episode_start_time = None

        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_successes.clear()
        self.episode_times.clear()
        self.episode_history.clear()

        # Reset metrics history
        for key in list(self.metrics_history.keys()):
            self.metrics_history[key].clear()

        # Reset curriculum
        self.curriculum_difficulty = self.config.get('curriculum', {}).get('initial_difficulty', 0.0)

    def get_curriculum_difficulty(self) -> float:
        """
        Get the current curriculum difficulty level

        Returns:
            Current difficulty level (0.0 to 1.0)
        """
        return self.curriculum_difficulty

    def set_curriculum_difficulty(self, difficulty: float):
        """
        Set the curriculum difficulty level

        Args:
            difficulty: New difficulty level (0.0 to 1.0)
        """
        self.curriculum_difficulty = max(0.0, min(1.0, difficulty))
        self.logger.info(f"Set curriculum difficulty to {self.curriculum_difficulty:.2f}")

    def get_recent_performance(self, window_size: int = 100) -> Dict[str, float]:
        """
        Get performance metrics for the most recent episodes

        Args:
            window_size: Number of recent episodes to consider

        Returns:
            Dictionary with recent performance metrics
        """
        recent_rewards = list(self.episode_rewards)[-window_size:]
        recent_successes = list(self.episode_successes)[-window_size:]

        if not recent_rewards:
            return {}

        return {
            'average_reward': float(np.mean(recent_rewards)),
            'std_reward': float(np.std(recent_rewards)),
            'success_rate': float(np.mean(recent_successes)) if recent_successes else 0.0,
            'episode_count': len(recent_rewards)
        }

    def save_episode_history(self, filepath: str):
        """
        Save the complete episode history to a file

        Args:
            filepath: Path to save the episode history
        """
        history_data = [asdict(ep_info) for ep_info in self.episode_history]

        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)

        self.logger.info(f"Saved episode history to {filepath}")

    def load_episode_history(self, filepath: str):
        """
        Load episode history from a file

        Args:
            filepath: Path to load the episode history from
        """
        with open(filepath, 'r') as f:
            history_data = json.load(f)

        self.episode_history = []
        for ep_data in history_data:
            ep_info = EpisodeInfo(**ep_data)
            self.episode_history.append(ep_info)

        # Update episode count based on history
        if self.episode_history:
            self.episode_count = max(ep.episode_id for ep in self.episode_history)

        self.logger.info(f"Loaded episode history from {filepath}")


class CurriculumManager:
    """
    Manages curriculum learning for humanoid robot control
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize curriculum manager

        Args:
            config: Configuration dictionary for curriculum learning
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        self.current_stage = 0
        self.difficulty_levels = config.get('difficulty_levels', [])
        self.performance_thresholds = config.get('performance_thresholds', [])
        self.stage_progress = 0.0

        if not self.difficulty_levels:
            # Default difficulty levels if none provided
            self.difficulty_levels = [
                {'name': 'beginner', 'difficulty': 0.0, 'description': 'Basic locomotion'},
                {'name': 'intermediate', 'difficulty': 0.5, 'description': 'Complex terrain'},
                {'name': 'advanced', 'difficulty': 0.8, 'description': 'Dynamic obstacles'}
            ]

        self.logger = logging.getLogger('Curriculum_Manager')
        self.logger.setLevel(logging.INFO)

    def update_stage(self, performance_metric: float) -> bool:
        """
        Update curriculum stage based on performance

        Args:
            performance_metric: Current performance metric (0.0 to 1.0)

        Returns:
            True if stage was updated
        """
        if not self.enabled:
            return False

        if self.current_stage >= len(self.difficulty_levels) - 1:
            # Already at the highest stage
            return False

        # Check if performance is high enough to advance
        if performance_metric >= self.performance_thresholds[self.current_stage]:
            self.current_stage += 1
            self.stage_progress = 0.0
            self.logger.info(f"Advanced to curriculum stage {self.current_stage}: "
                           f"{self.difficulty_levels[self.current_stage]['name']}")
            return True

        return False

    def get_current_difficulty(self) -> float:
        """
        Get the current difficulty level

        Returns:
            Current difficulty (0.0 to 1.0)
        """
        if not self.enabled or self.current_stage >= len(self.difficulty_levels):
            return 0.0

        return self.difficulty_levels[self.current_stage]['difficulty']

    def get_current_stage_info(self) -> Dict[str, Any]:
        """
        Get information about the current stage

        Returns:
            Dictionary with current stage information
        """
        if not self.enabled or self.current_stage >= len(self.difficulty_levels):
            return {}

        return {
            'stage': self.current_stage,
            'name': self.difficulty_levels[self.current_stage]['name'],
            'difficulty': self.difficulty_levels[self.current_stage]['difficulty'],
            'description': self.difficulty_levels[self.current_stage]['description']
        }


def main():
    """
    Example usage of the EpisodeManager
    """
    # Example configuration
    config = {
        'episode_history_length': 100,
        'log_dir': './logs',
        'checkpoint_dir': './checkpoints',
        'curriculum': {
            'enabled': True,
            'initial_difficulty': 0.0,
            'difficulty_threshold': 0.8,
            'difficulty_increment': 0.01
        },
        'tracked_metrics': ['forward_velocity', 'base_height', 'action_smoothness']
    }

    # Create episode manager
    episode_manager = EpisodeManager(config)

    print("Episode Manager initialized successfully")
    print(f"Log directory: {episode_manager.log_dir}")
    print(f"Checkpoint directory: {episode_manager.checkpoint_dir}")

    # Example: Simulate a few episodes
    for i in range(5):
        # Start episode
        episode_id = episode_manager.start_episode(initial_position=(0.0, 0.0, 0.5))

        # Simulate episode (in real training, this would be the actual environment interaction)
        import random
        episode_length = random.randint(200, 500)
        total_reward = random.uniform(100, 1000)
        success = random.choice([True, False])
        termination_reason = random.choice(['goal_reached', 'timeout', 'fallen'])
        final_position = (random.uniform(1.0, 5.0), random.uniform(-1.0, 1.0), 0.5)
        initial_position = (0.0, 0.0, 0.5)

        # Add some metrics
        metrics = {
            'forward_velocity': random.uniform(0.5, 1.5),
            'base_height': random.uniform(0.4, 0.6),
            'action_smoothness': random.uniform(0.7, 0.9)
        }

        # End episode
        episode_info = episode_manager.end_episode(
            total_reward=total_reward,
            episode_length=episode_length,
            success=success,
            termination_reason=termination_reason,
            final_position=final_position,
            initial_position=initial_position,
            metrics=metrics
        )

        print(f"Episode {episode_id} completed with reward: {total_reward:.2f}")

    # Print final statistics
    stats = episode_manager.get_statistics()
    print("\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save episode history
    episode_manager.save_episode_history('./logs/episode_history.json')
    print("\nEpisode history saved to ./logs/episode_history.json")


if __name__ == "__main__":
    main()