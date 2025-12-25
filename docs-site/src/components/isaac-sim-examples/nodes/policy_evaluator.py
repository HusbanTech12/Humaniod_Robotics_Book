#!/usr/bin/env python3
# policy_evaluator.py

"""
Isaac Lab Policy Evaluation Framework for Humanoid Robot Control

This module provides tools for evaluating trained RL policies in simulation,
including performance metrics, comparison tools, and visualization capabilities.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import yaml
import json
import pickle
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pandas as pd
from pathlib import Path
import seaborn as sns
from datetime import datetime


@dataclass
class PolicyEvaluationResult:
    """
    Data class to store policy evaluation results
    """
    policy_name: str
    evaluation_id: str
    timestamp: float
    mean_reward: float
    std_reward: float
    success_rate: float
    mean_episode_length: float
    total_evaluations: int
    metrics: Dict[str, float]
    performance_breakdown: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    environment_conditions: Dict[str, Any]


class PolicyEvaluator:
    """
    Framework for evaluating trained RL policies for humanoid robot control
    """
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize the policy evaluator

        Args:
            config_path: Path to configuration file
            config: Configuration dictionary (if config_path is None)
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            # Default configuration
            self.config = {
                'evaluation_episodes': 50,
                'render': False,
                'save_videos': False,
                'metrics': {
                    'forward_velocity': True,
                    'base_height': True,
                    'action_smoothness': True,
                    'energy_efficiency': True,
                    'stability': True
                },
                'success_criteria': {
                    'min_distance': 2.0,  # Minimum distance to consider success
                    'max_time_penalty': 0.1,  # Maximum time penalty
                    'upright_tolerance': 0.7  # Minimum upright orientation
                },
                'evaluation_scenarios': [
                    'flat_ground',
                    'rough_terrain',
                    'stairs',
                    'obstacles'
                ],
                'logging': {
                    'log_dir': './logs/policy_evaluation',
                    'save_plots': True,
                    'save_results': True
                }
            }

        # Initialize logging directory
        self.log_dir = self.config['logging']['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize metrics tracking
        self.evaluation_results: List[PolicyEvaluationResult] = []
        self.metric_history = defaultdict(list)

        # Set up device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_policy(self, policy_path: str) -> nn.Module:
        """
        Load a trained policy from file

        Args:
            policy_path: Path to the trained policy file

        Returns:
            Loaded policy network
        """
        # Load the checkpoint
        checkpoint = torch.load(policy_path, map_location=self.device)

        if 'actor_critic_state_dict' in checkpoint:
            # This is a checkpoint with full training state
            policy_state_dict = checkpoint['actor_critic_state_dict']
        else:
            # This is a direct policy state dict
            policy_state_dict = checkpoint

        # For now, we'll return the state dict - in a real implementation,
        # you would need to recreate the network architecture
        # This is a placeholder implementation
        return policy_state_dict

    def evaluate_policy(self, policy_path: str, environment, policy_name: str = "unknown") -> PolicyEvaluationResult:
        """
        Evaluate a trained policy in the given environment

        Args:
            policy_path: Path to the trained policy file
            environment: Environment to evaluate the policy in
            policy_name: Name of the policy for identification

        Returns:
            PolicyEvaluationResult containing evaluation metrics
        """
        print(f"Evaluating policy: {policy_name}")
        print(f"Loading policy from: {policy_path}")

        # Load the policy
        policy = self.load_policy(policy_path)

        # Initialize evaluation metrics
        episode_rewards = []
        episode_lengths = []
        successes = []
        all_metrics = defaultdict(list)

        # Run multiple evaluation episodes
        num_episodes = self.config['evaluation_episodes']
        for episode in range(num_episodes):
            print(f"Evaluation episode {episode + 1}/{num_episodes}")

            # Reset environment
            obs = environment.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_metrics = defaultdict(float)

            # Run episode
            while not done:
                # Get action from policy (placeholder implementation)
                with torch.no_grad():
                    # In a real implementation, you would use the loaded policy to get actions
                    # For now, we'll generate random actions as a placeholder
                    action = torch.randn(environment.action_space.shape[0]).to(self.device)
                    action = torch.clamp(action, -1.0, 1.0)  # Clamp to valid range

                # Take step in environment
                next_obs, reward, done, info = environment.step(action.cpu().numpy())

                # Accumulate metrics
                episode_reward += reward
                episode_length += 1

                # Collect episode-specific metrics if available
                if 'metrics' in info:
                    for key, value in info['metrics'].items():
                        episode_metrics[key] += value

                obs = next_obs

            # Store episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Determine success based on criteria
            success = self._check_success_criteria(episode_metrics, episode_length)
            successes.append(success)

            # Store metrics for this episode
            for key, value in episode_metrics.items():
                all_metrics[key].append(value)

            print(f"  Episode reward: {episode_reward:.2f}, Length: {episode_length}, Success: {success}")

        # Calculate overall metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        success_rate = np.mean(successes) if successes else 0.0
        mean_episode_length = np.mean(episode_lengths)

        # Calculate metric averages
        metric_averages = {}
        for key, values in all_metrics.items():
            metric_averages[key] = np.mean(values)

        # Create performance breakdown
        performance_breakdown = {
            'reward_distribution': {
                'min': float(np.min(episode_rewards)),
                'max': float(np.max(episode_rewards)),
                'percentile_25': float(np.percentile(episode_rewards, 25)),
                'percentile_75': float(np.percentile(episode_rewards, 75))
            },
            'episode_length_distribution': {
                'min': int(np.min(episode_lengths)),
                'max': int(np.max(episode_lengths)),
                'percentile_25': float(np.percentile(episode_lengths, 25)),
                'percentile_75': float(np.percentile(episode_lengths, 75))
            }
        }

        # Create evaluation result
        result = PolicyEvaluationResult(
            policy_name=policy_name,
            evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=time.time(),
            mean_reward=float(mean_reward),
            std_reward=float(std_reward),
            success_rate=float(success_rate),
            mean_episode_length=float(mean_episode_length),
            total_evaluations=num_episodes,
            metrics=metric_averages,
            performance_breakdown=performance_breakdown,
            evaluation_config=self.config.copy(),
            environment_conditions=getattr(environment, 'get_conditions', lambda: {})()
        )

        # Store result
        self.evaluation_results.append(result)

        # Log metrics history
        self.metric_history['mean_reward'].append(mean_reward)
        self.metric_history['success_rate'].append(success_rate)

        return result

    def _check_success_criteria(self, metrics: Dict[str, float], episode_length: int) -> bool:
        """
        Check if an episode meets success criteria

        Args:
            metrics: Dictionary of episode metrics
            episode_length: Length of the episode

        Returns:
            True if episode is considered successful
        """
        # Placeholder success criteria
        # In a real implementation, this would check actual success conditions
        success = True

        # Example criteria (these would be customized based on the task)
        if 'distance_traveled' in metrics:
            success = success and (metrics['distance_traveled'] >= self.config['success_criteria']['min_distance'])

        return success

    def compare_policies(self, policy_paths: List[str], environment, policy_names: List[str] = None) -> List[PolicyEvaluationResult]:
        """
        Compare multiple policies

        Args:
            policy_paths: List of paths to policy files
            environment: Environment to evaluate policies in
            policy_names: Optional list of policy names

        Returns:
            List of evaluation results for each policy
        """
        if policy_names is None:
            policy_names = [f"Policy_{i}" for i in range(len(policy_paths))]

        results = []
        for path, name in zip(policy_paths, policy_names):
            result = self.evaluate_policy(path, environment, name)
            results.append(result)

        return results

    def generate_evaluation_report(self, results: List[PolicyEvaluationResult], output_path: str = None) -> str:
        """
        Generate a comprehensive evaluation report

        Args:
            results: List of evaluation results to include in report
            output_path: Optional path to save the report

        Returns:
            Path to the generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"evaluation_report_{timestamp}.json")

        # Prepare report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': len(results),
            'config': self.config,
            'policies': []
        }

        for result in results:
            policy_data = {
                'policy_name': result.policy_name,
                'evaluation_id': result.evaluation_id,
                'mean_reward': result.mean_reward,
                'std_reward': result.std_reward,
                'success_rate': result.success_rate,
                'mean_episode_length': result.mean_episode_length,
                'total_evaluations': result.total_evaluations,
                'metrics': result.metrics,
                'performance_breakdown': result.performance_breakdown
            }
            report_data['policies'].append(policy_data)

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"Evaluation report saved to: {output_path}")
        return output_path

    def plot_evaluation_results(self, results: List[PolicyEvaluationResult], output_path: str = None):
        """
        Create visualizations of evaluation results

        Args:
            results: List of evaluation results to visualize
            output_path: Optional path to save the plot
        """
        if not results:
            print("No results to plot")
            return

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"evaluation_plot_{timestamp}.png")

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Policy Evaluation Results', fontsize=16)

        # Extract data
        policy_names = [r.policy_name for r in results]
        mean_rewards = [r.mean_reward for r in results]
        success_rates = [r.success_rate for r in results]
        episode_lengths = [r.mean_episode_length for r in results]

        # Plot 1: Mean rewards
        axes[0, 0].bar(policy_names, mean_rewards, color='skyblue')
        axes[0, 0].set_title('Mean Reward per Policy')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Success rates
        axes[0, 1].bar(policy_names, success_rates, color='lightgreen')
        axes[0, 1].set_title('Success Rate per Policy')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Episode lengths
        axes[1, 0].bar(policy_names, episode_lengths, color='salmon')
        axes[1, 0].set_title('Mean Episode Length per Policy')
        axes[1, 0].set_ylabel('Mean Episode Length')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 4: Success rate vs Mean reward scatter
        scatter = axes[1, 1].scatter(mean_rewards, success_rates, s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Mean Reward')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Success Rate vs Mean Reward')

        # Add policy names to scatter plot
        for i, name in enumerate(policy_names):
            axes[1, 1].annotate(name, (mean_rewards[i], success_rates[i]), xytext=(5, 5),
                              textcoords='offset points', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Evaluation plot saved to: {output_path}")

    def plot_metric_history(self, output_path: str = None):
        """
        Plot the history of key metrics over time

        Args:
            output_path: Optional path to save the plot
        """
        if not self.metric_history['mean_reward']:
            print("No metric history to plot")
            return

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"metric_history_{timestamp}.png")

        plt.figure(figsize=(12, 8))

        # Plot mean reward over time
        plt.subplot(2, 1, 1)
        plt.plot(self.metric_history['mean_reward'], marker='o', linestyle='-', label='Mean Reward')
        plt.title('Mean Reward Over Time')
        plt.xlabel('Evaluation Run')
        plt.ylabel('Mean Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot success rate over time
        plt.subplot(2, 1, 2)
        plt.plot(self.metric_history['success_rate'], marker='s', linestyle='-', label='Success Rate', color='orange')
        plt.title('Success Rate Over Time')
        plt.xlabel('Evaluation Run')
        plt.ylabel('Success Rate')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Metric history plot saved to: {output_path}")

    def save_evaluation_results(self, results: List[PolicyEvaluationResult], output_path: str = None):
        """
        Save evaluation results to a file

        Args:
            results: List of evaluation results to save
            output_path: Optional path to save the results
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"evaluation_results_{timestamp}.pkl")

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = asdict(result)
            # Convert numpy types to native Python types for JSON serialization
            for key, value in serializable_result.items():
                if isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
            serializable_results.append(serializable_result)

        with open(output_path, 'wb') as f:
            pickle.dump(serializable_results, f)

        print(f"Evaluation results saved to: {output_path}")

    def load_evaluation_results(self, input_path: str) -> List[PolicyEvaluationResult]:
        """
        Load evaluation results from a file

        Args:
            input_path: Path to the saved results file

        Returns:
            List of loaded evaluation results
        """
        with open(input_path, 'rb') as f:
            serializable_results = pickle.load(f)

        results = []
        for serializable_result in serializable_results:
            # Convert back to PolicyEvaluationResult
            result = PolicyEvaluationResult(
                policy_name=serializable_result['policy_name'],
                evaluation_id=serializable_result['evaluation_id'],
                timestamp=serializable_result['timestamp'],
                mean_reward=serializable_result['mean_reward'],
                std_reward=serializable_result['std_reward'],
                success_rate=serializable_result['success_rate'],
                mean_episode_length=serializable_result['mean_episode_length'],
                total_evaluations=serializable_result['total_evaluations'],
                metrics=serializable_result['metrics'],
                performance_breakdown=serializable_result['performance_breakdown'],
                evaluation_config=serializable_result['evaluation_config'],
                environment_conditions=serializable_result['environment_conditions']
            )
            results.append(result)

        # Store in instance variable
        self.evaluation_results = results

        print(f"Evaluation results loaded from: {input_path}")
        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of overall performance across all evaluations

        Returns:
            Dictionary with performance summary
        """
        if not self.evaluation_results:
            return {}

        # Extract metrics from all evaluations
        all_mean_rewards = [r.mean_reward for r in self.evaluation_results]
        all_success_rates = [r.success_rate for r in self.evaluation_results]
        all_episode_lengths = [r.mean_episode_length for r in self.evaluation_results]

        summary = {
            'total_evaluations': len(self.evaluation_results),
            'policies_evaluated': list(set(r.policy_name for r in self.evaluation_results)),
            'overall_mean_reward': float(np.mean(all_mean_rewards)),
            'overall_std_reward': float(np.std(all_mean_rewards)),
            'overall_success_rate': float(np.mean(all_success_rates)),
            'overall_mean_episode_length': float(np.mean(all_episode_lengths)),
            'best_policy': self.evaluation_results[np.argmax(all_mean_rewards)].policy_name if all_mean_rewards else None,
            'best_reward': float(np.max(all_mean_rewards)) if all_mean_rewards else None,
            'improvement_trend': self._calculate_improvement_trend()
        }

        return summary

    def _calculate_improvement_trend(self) -> float:
        """
        Calculate the improvement trend based on metric history

        Returns:
            Improvement trend (slope of linear regression)
        """
        if len(self.metric_history['mean_reward']) < 2:
            return 0.0

        x = np.arange(len(self.metric_history['mean_reward']))
        y = np.array(self.metric_history['mean_reward'])

        # Calculate slope of linear regression
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export evaluation results to a pandas DataFrame

        Returns:
            DataFrame containing evaluation results
        """
        if not self.evaluation_results:
            return pd.DataFrame()

        data = []
        for result in self.evaluation_results:
            row = {
                'policy_name': result.policy_name,
                'evaluation_id': result.evaluation_id,
                'timestamp': datetime.fromtimestamp(result.timestamp),
                'mean_reward': result.mean_reward,
                'std_reward': result.std_reward,
                'success_rate': result.success_rate,
                'mean_episode_length': result.mean_episode_length,
                'total_evaluations': result.total_evaluations
            }
            # Add metric columns
            for key, value in result.metrics.items():
                row[f'metric_{key}'] = value

            data.append(row)

        return pd.DataFrame(data)

    def export_results_csv(self, output_path: str = None):
        """
        Export evaluation results to CSV

        Args:
            output_path: Optional path to save the CSV file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"evaluation_results_{timestamp}.csv")

        df = self.export_to_dataframe()
        df.to_csv(output_path, index=False)

        print(f"Evaluation results exported to CSV: {output_path}")


class PolicyComparisonTool:
    """
    Tool for comparing multiple policies side-by-side
    """
    def __init__(self):
        self.evaluator = PolicyEvaluator()

    def compare_policies_interactive(self, policy_paths: List[str], environment, policy_names: List[str] = None):
        """
        Interactive comparison of policies with detailed metrics

        Args:
            policy_paths: List of paths to policy files
            environment: Environment to evaluate policies in
            policy_names: Optional list of policy names
        """
        if policy_names is None:
            policy_names = [f"Policy_{i}" for i in range(len(policy_paths))]

        print("Starting interactive policy comparison...")
        print("="*60)

        # Evaluate all policies
        results = []
        for path, name in zip(policy_paths, policy_names):
            print(f"\nEvaluating {name}...")
            result = self.evaluator.evaluate_policy(path, environment, name)
            results.append(result)

        # Generate comparison report
        self._generate_comparison_report(results)

        # Create visualizations
        self.evaluator.plot_evaluation_results(results)

        return results

    def _generate_comparison_report(self, results: List[PolicyEvaluationResult]):
        """
        Generate a detailed comparison report

        Args:
            results: List of evaluation results to compare
        """
        print("\n" + "="*60)
        print("POLICY COMPARISON REPORT")
        print("="*60)

        # Print header
        print(f"{'Policy Name':<20} {'Mean Reward':<12} {'Std Reward':<12} {'Success Rate':<12} {'Avg Length':<12}")
        print("-" * 70)

        # Print results for each policy
        for result in results:
            print(f"{result.policy_name:<20} {result.mean_reward:<12.2f} {result.std_reward:<12.2f} "
                  f"{result.success_rate:<12.2%} {result.mean_episode_length:<12.2f}")

        print("\n" + "="*60)
        print("RANKINGS")
        print("="*60)

        # Rank by mean reward
        sorted_by_reward = sorted(results, key=lambda x: x.mean_reward, reverse=True)
        print("By Mean Reward:")
        for i, result in enumerate(sorted_by_reward, 1):
            print(f"  {i}. {result.policy_name}: {result.mean_reward:.2f}")

        # Rank by success rate
        sorted_by_success = sorted(results, key=lambda x: x.success_rate, reverse=True)
        print(f"\nBy Success Rate:")
        for i, result in enumerate(sorted_by_success, 1):
            print(f"  {i}. {result.policy_name}: {result.success_rate:.2%}")

        # Performance summary
        summary = self.evaluator.get_performance_summary()
        print(f"\nPerformance Summary:")
        print(f"  Best Policy: {summary.get('best_policy', 'N/A')}")
        print(f"  Best Reward: {summary.get('best_reward', 0.0):.2f}")
        print(f"  Improvement Trend: {summary.get('improvement_trend', 0.0):.4f}")


def main():
    """
    Example usage of the PolicyEvaluator
    """
    print("Policy Evaluation Framework for Humanoid Robot Control")
    print("=" * 60)

    # Example configuration for evaluation
    eval_config = {
        'evaluation_episodes': 20,  # Reduced for example
        'render': False,
        'save_videos': False,
        'metrics': {
            'forward_velocity': True,
            'base_height': True,
            'action_smoothness': True,
            'energy_efficiency': True,
            'stability': True
        },
        'success_criteria': {
            'min_distance': 2.0,
            'max_time_penalty': 0.1,
            'upright_tolerance': 0.7
        },
        'logging': {
            'log_dir': './logs/policy_evaluation',
            'save_plots': True,
            'save_results': True
        }
    }

    # Create evaluator
    evaluator = PolicyEvaluator(config=eval_config)

    print("Policy evaluator initialized with configuration:")
    print(f"  - Evaluation episodes: {eval_config['evaluation_episodes']}")
    print(f"  - Metrics tracked: {list(eval_config['metrics'].keys())}")
    print(f"  - Log directory: {eval_config['logging']['log_dir']}")

    # Example: Create some mock evaluation results (in real usage, you would evaluate actual policies)
    from dataclasses import asdict

    # Mock results for demonstration
    mock_results = []
    for i in range(3):
        mock_result = PolicyEvaluationResult(
            policy_name=f"Policy_{i+1}",
            evaluation_id=f"eval_{i+1}",
            timestamp=time.time() - i*3600,  # Different timestamps
            mean_reward=np.random.uniform(100, 500),
            std_reward=np.random.uniform(10, 50),
            success_rate=np.random.uniform(0.5, 1.0),
            mean_episode_length=np.random.uniform(200, 500),
            total_evaluations=eval_config['evaluation_episodes'],
            metrics={
                'forward_velocity': np.random.uniform(0.5, 1.5),
                'base_height': np.random.uniform(0.4, 0.6),
                'action_smoothness': np.random.uniform(0.7, 0.9)
            },
            performance_breakdown={
                'reward_distribution': {
                    'min': 50.0,
                    'max': 600.0,
                    'percentile_25': 150.0,
                    'percentile_75': 400.0
                }
            },
            evaluation_config=eval_config,
            environment_conditions={'terrain': 'flat', 'obstacles': 0}
        )
        mock_results.append(mock_result)
        evaluator.evaluation_results.append(mock_result)

    print(f"\nGenerated {len(mock_results)} mock evaluation results")

    # Generate report
    report_path = evaluator.generate_evaluation_report(mock_results)
    print(f"Report generated: {report_path}")

    # Create visualizations
    evaluator.plot_evaluation_results(mock_results)
    evaluator.plot_metric_history()

    # Export to CSV
    evaluator.export_results_csv()

    # Print performance summary
    summary = evaluator.get_performance_summary()
    print(f"\nPerformance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\nPolicy evaluation framework setup complete!")
    print(f"Log directory: {evaluator.log_dir}")


if __name__ == "__main__":
    main()