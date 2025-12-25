#!/usr/bin/env python3
# performance_validator.py

"""
Isaac Lab Performance Validation Framework for Humanoid Robot Control

This module provides tools for validating the performance of trained policies
and ensuring successful sim-to-real transfer for humanoid robot control.
"""

import os
import numpy as np
import torch
import yaml
import json
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path

# Isaac Sim imports (these would be used in a real implementation)
try:
    import omni
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.articulations import ArticulationView
except ImportError:
    print("Isaac Sim modules not available - using mock implementations for validation")


@dataclass
class PerformanceMetrics:
    """
    Data class to store performance metrics
    """
    episode_reward: float
    success_rate: float
    episode_length: int
    distance_traveled: float
    energy_consumption: float
    stability_score: float
    tracking_accuracy: float
    computation_time: float
    sim_real_gap: float
    timestamp: float


class PerformanceValidator:
    """
    Framework for validating performance of humanoid robot control policies
    """
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize the performance validator

        Args:
            config_path: Path to configuration file
            config: Configuration dictionary (if config_path is None)
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            # Default configuration
            self.config = {
                'validation_episodes': 50,
                'render': False,
                'metrics': {
                    'reward': True,
                    'success_rate': True,
                    'distance': True,
                    'energy': True,
                    'stability': True,
                    'tracking': True
                },
                'validation_scenarios': [
                    'flat_ground',
                    'rough_terrain',
                    'stairs',
                    'obstacles'
                ],
                'success_criteria': {
                    'min_success_rate': 0.8,
                    'min_distance': 2.0,
                    'max_energy_per_meter': 100.0,
                    'min_stability_score': 0.7
                },
                'logging': {
                    'log_dir': './logs/performance_validation',
                    'save_plots': True,
                    'save_results': True
                }
            }

        # Initialize logging directory
        self.log_dir = self.config['logging']['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize metrics tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.metric_history = defaultdict(list)

        # Set up logging
        self.setup_logging()

        # Performance thresholds
        self.success_thresholds = self.config.get('success_criteria', {})

    def setup_logging(self):
        """
        Set up logging for performance validation
        """
        # Create logger
        self.logger = logging.getLogger('PerformanceValidator')
        self.logger.setLevel(logging.INFO)

        # Create file handler
        log_file = os.path.join(self.log_dir, 'performance_validation.log')
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def validate_policy_performance(self,
                                 policy,
                                 environment,
                                 policy_name: str = "unknown") -> Dict[str, Any]:
        """
        Validate the performance of a trained policy

        Args:
            policy: Trained policy to validate
            environment: Environment to test the policy in
            policy_name: Name of the policy for identification

        Returns:
            Dictionary containing performance metrics
        """
        self.logger.info(f"Starting performance validation for policy: {policy_name}")

        # Initialize validation metrics
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        total_episodes = self.config['validation_episodes']

        distance_traveled_list = []
        energy_consumption_list = []
        stability_scores = []
        tracking_accuracy_list = []

        # Run validation episodes
        for episode in range(total_episodes):
            self.logger.info(f"Validation episode {episode + 1}/{total_episodes}")

            # Reset environment
            obs = environment.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_distance = 0.0
            episode_energy = 0.0

            # Track state for metrics calculation
            initial_position = None
            final_position = None
            position_history = []

            # Run episode
            while not done:
                # Get action from policy
                with torch.no_grad():
                    # In a real implementation, this would use the actual policy
                    # For now, we'll generate a random action as a placeholder
                    if isinstance(obs, dict):
                        obs_tensor = torch.tensor(obs['obs'], dtype=torch.float32)
                    else:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32)

                    # Placeholder action - in real implementation, use actual policy
                    action = torch.randn(environment.action_space.shape[0])
                    action = torch.clamp(action, -1.0, 1.0)

                # Take step in environment
                next_obs, reward, done, info = environment.step(action.numpy())

                # Track metrics
                episode_reward += reward
                episode_length += 1

                # Store position for distance calculation (if available)
                if 'position' in info:
                    position_history.append(info['position'])
                    if initial_position is None:
                        initial_position = info['position']

                # Calculate energy consumption (placeholder)
                if 'action' in info:
                    episode_energy += np.sum(np.square(info['action']))

                obs = next_obs

            # Calculate final position and distance
            if position_history:
                final_position = position_history[-1]
                if initial_position is not None:
                    episode_distance = np.linalg.norm(
                        np.array(final_position) - np.array(initial_position)
                    )
                else:
                    episode_distance = np.linalg.norm(np.array(final_position))

            # Calculate stability score (placeholder)
            stability_score = self._calculate_stability_score(info)

            # Calculate tracking accuracy (placeholder)
            tracking_accuracy = self._calculate_tracking_accuracy(info)

            # Store episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            distance_traveled_list.append(episode_distance)
            energy_consumption_list.append(episode_energy)
            stability_scores.append(stability_score)
            tracking_accuracy_list.append(tracking_accuracy)

            # Determine success (placeholder - would be based on actual criteria)
            success = self._check_success_criteria(
                episode_distance, stability_score, tracking_accuracy
            )
            if success:
                success_count += 1

            # Log episode metrics
            episode_metrics = PerformanceMetrics(
                episode_reward=episode_reward,
                success_rate=1.0 if success else 0.0,
                episode_length=episode_length,
                distance_traveled=episode_distance,
                energy_consumption=episode_energy,
                stability_score=stability_score,
                tracking_accuracy=tracking_accuracy,
                computation_time=0.0,  # Placeholder
                sim_real_gap=0.0,      # Placeholder
                timestamp=time.time()
            )
            self.performance_history.append(episode_metrics)

            self.logger.info(
                f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                f"Distance={episode_distance:.2f}, Success={success}"
            )

        # Calculate overall metrics
        success_rate = success_count / total_episodes
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        avg_distance = np.mean(distance_traveled_list)
        avg_energy = np.mean(energy_consumption_list)
        avg_stability = np.mean(stability_scores)
        avg_tracking = np.mean(tracking_accuracy_list)

        # Calculate energy efficiency
        energy_efficiency = avg_energy / avg_distance if avg_distance > 0 else float('inf')

        # Create validation result
        validation_result = {
            'policy_name': policy_name,
            'total_episodes': total_episodes,
            'success_rate': success_rate,
            'average_reward': avg_reward,
            'average_episode_length': avg_length,
            'average_distance_traveled': avg_distance,
            'average_energy_consumption': avg_energy,
            'average_stability_score': avg_stability,
            'average_tracking_accuracy': avg_tracking,
            'energy_efficiency': energy_efficiency,
            'metrics_breakdown': {
                'rewards': episode_rewards,
                'lengths': episode_lengths,
                'distances': distance_traveled_list,
                'energies': energy_consumption_list,
                'stabilities': stability_scores,
                'tracking_accuracies': tracking_accuracy_list
            },
            'validation_passed': self._check_validation_criteria({
                'success_rate': success_rate,
                'average_distance': avg_distance,
                'energy_efficiency': energy_efficiency,
                'average_stability': avg_stability
            })
        }

        self.logger.info(f"Validation completed for {policy_name}")
        self.logger.info(f"Success rate: {success_rate:.2%}")
        self.logger.info(f"Average reward: {avg_reward:.2f}")
        self.logger.info(f"Average distance: {avg_distance:.2f}")
        self.logger.info(f"Validation passed: {validation_result['validation_passed']}")

        return validation_result

    def _calculate_stability_score(self, info: Dict[str, Any]) -> float:
        """
        Calculate stability score based on robot state

        Args:
            info: Information dictionary from environment step

        Returns:
            Stability score (0.0 to 1.0)
        """
        # Placeholder implementation - in real scenario, this would use actual stability metrics
        # such as deviation from upright position, joint oscillations, etc.
        return np.random.uniform(0.6, 0.9)  # Random value for demonstration

    def _calculate_tracking_accuracy(self, info: Dict[str, Any]) -> float:
        """
        Calculate tracking accuracy based on robot state

        Args:
            info: Information dictionary from environment step

        Returns:
            Tracking accuracy score (0.0 to 1.0)
        """
        # Placeholder implementation - in real scenario, this would use actual tracking metrics
        # such as how well the robot follows desired trajectories
        return np.random.uniform(0.7, 0.95)  # Random value for demonstration

    def _check_success_criteria(self, distance: float, stability: float, tracking: float) -> bool:
        """
        Check if an episode meets success criteria

        Args:
            distance: Distance traveled in the episode
            stability: Stability score for the episode
            tracking: Tracking accuracy for the episode

        Returns:
            True if episode is considered successful
        """
        # Placeholder success criteria
        min_distance = self.config.get('success_criteria', {}).get('min_distance', 1.0)
        min_stability = self.config.get('success_criteria', {}).get('min_stability_score', 0.5)

        return distance >= min_distance and stability >= min_stability

    def _check_validation_criteria(self, metrics: Dict[str, float]) -> bool:
        """
        Check if overall validation meets criteria

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            True if validation passes
        """
        thresholds = self.config.get('success_criteria', {})

        # Check each criterion
        success_rate_ok = metrics.get('success_rate', 0) >= thresholds.get('min_success_rate', 0.8)
        distance_ok = metrics.get('average_distance', 0) >= thresholds.get('min_distance', 2.0)
        energy_ok = (metrics.get('energy_efficiency', float('inf')) <=
                    thresholds.get('max_energy_per_meter', 100.0))
        stability_ok = metrics.get('average_stability', 0) >= thresholds.get('min_stability_score', 0.7)

        return success_rate_ok and distance_ok and energy_ok and stability_ok

    def compare_policies(self, policies: List[Tuple[Any, str]], environment) -> List[Dict[str, Any]]:
        """
        Compare multiple policies

        Args:
            policies: List of tuples (policy, name)
            environment: Environment to test policies in

        Returns:
            List of validation results for each policy
        """
        results = []
        for policy, name in policies:
            result = self.validate_policy_performance(policy, environment, name)
            results.append(result)

        return results

    def generate_validation_report(self, results: List[Dict[str, Any]], output_path: str = None) -> str:
        """
        Generate a comprehensive validation report

        Args:
            results: List of validation results to include in report
            output_path: Optional path to save the report

        Returns:
            Path to the generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"validation_report_{timestamp}.json")

        # Prepare report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_validations': len(results),
            'config': self.config,
            'results': []
        }

        for result in results:
            report_data['results'].append(result)

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Validation report saved to: {output_path}")
        return output_path

    def plot_validation_results(self, results: List[Dict[str, Any]], output_path: str = None):
        """
        Create visualizations of validation results

        Args:
            results: List of validation results to visualize
            output_path: Optional path to save the plot
        """
        if not results:
            self.logger.warning("No results to plot")
            return

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"validation_plot_{timestamp}.png")

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Policy Performance Validation Results', fontsize=16)

        # Extract data
        policy_names = [r['policy_name'] for r in results]
        success_rates = [r['success_rate'] for r in results]
        avg_rewards = [r['average_reward'] for r in results]
        avg_distances = [r['average_distance_traveled'] for r in results]
        avg_stabilities = [r['average_stability_score'] for r in results]
        energy_efficiencies = [r['energy_efficiency'] for r in results]

        # Plot 1: Success rates
        axes[0, 0].bar(policy_names, success_rates, color='lightblue')
        axes[0, 0].set_title('Success Rate per Policy')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Average rewards
        axes[0, 1].bar(policy_names, avg_rewards, color='lightgreen')
        axes[0, 1].set_title('Average Reward per Policy')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Average distances
        axes[0, 2].bar(policy_names, avg_distances, color='salmon')
        axes[0, 2].set_title('Average Distance per Policy')
        axes[0, 2].set_ylabel('Average Distance (m)')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Plot 4: Stability scores
        axes[1, 0].bar(policy_names, avg_stabilities, color='gold')
        axes[1, 0].set_title('Average Stability Score per Policy')
        axes[1, 0].set_ylabel('Stability Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 5: Energy efficiency
        axes[1, 1].bar(policy_names, energy_efficiencies, color='orange')
        axes[1, 1].set_title('Energy Efficiency per Policy')
        axes[1, 1].set_ylabel('Energy per Distance Unit')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Plot 6: Validation pass/fail
        validation_results = [r['validation_passed'] for r in results]
        colors = ['green' if v else 'red' for v in validation_results]
        axes[1, 2].bar(policy_names, [1 if v else 0 for v in validation_results], color=colors)
        axes[1, 2].set_title('Validation Pass/Fail')
        axes[1, 2].set_ylabel('Pass (1) / Fail (0)')
        axes[1, 2].set_ylim(0, 1.1)
        axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Validation plot saved to: {output_path}")

    def save_validation_results(self, results: List[Dict[str, Any]], output_path: str = None):
        """
        Save validation results to a file

        Args:
            results: List of validation results to save
            output_path: Optional path to save the results
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"validation_results_{timestamp}.pkl")

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

        self.logger.info(f"Validation results saved to: {output_path}")

    def load_validation_results(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Load validation results from a file

        Args:
            input_path: Path to the saved results file

        Returns:
            List of loaded validation results
        """
        with open(input_path, 'rb') as f:
            results = pickle.load(f)

        self.logger.info(f"Validation results loaded from: {input_path}")
        return results

    def get_performance_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get a summary of performance across all validation results

        Args:
            results: List of validation results

        Returns:
            Dictionary with performance summary
        """
        if not results:
            return {}

        # Extract metrics from all results
        all_success_rates = [r['success_rate'] for r in results]
        all_avg_rewards = [r['average_reward'] for r in results]
        all_avg_distances = [r['average_distance_traveled'] for r in results]
        all_avg_stabilities = [r['average_stability_score'] for r in results]

        summary = {
            'total_validations': len(results),
            'policies_validated': [r['policy_name'] for r in results],
            'overall_average_success_rate': float(np.mean(all_success_rates)),
            'overall_average_reward': float(np.mean(all_avg_rewards)),
            'overall_average_distance': float(np.mean(all_avg_distances)),
            'overall_average_stability': float(np.mean(all_avg_stabilities)),
            'success_rate_std': float(np.std(all_success_rates)),
            'reward_std': float(np.std(all_avg_rewards)),
            'best_policy': results[np.argmax(all_avg_rewards)]['policy_name'] if all_avg_rewards else None,
            'best_success_rate': float(np.max(all_success_rates)) if all_success_rates else None,
            'validation_pass_rate': float(np.mean([r['validation_passed'] for r in results]))
        }

        return summary

    def export_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Export validation results to a pandas DataFrame

        Args:
            results: List of validation results

        Returns:
            DataFrame containing validation results
        """
        if not results:
            return pd.DataFrame()

        data = []
        for result in results:
            row = {
                'policy_name': result['policy_name'],
                'total_episodes': result['total_episodes'],
                'success_rate': result['success_rate'],
                'average_reward': result['average_reward'],
                'average_episode_length': result['average_episode_length'],
                'average_distance_traveled': result['average_distance_traveled'],
                'average_energy_consumption': result['average_energy_consumption'],
                'average_stability_score': result['average_stability_score'],
                'average_tracking_accuracy': result['average_tracking_accuracy'],
                'energy_efficiency': result['energy_efficiency'],
                'validation_passed': result['validation_passed']
            }
            data.append(row)

        return pd.DataFrame(data)

    def export_results_csv(self, results: List[Dict[str, Any]], output_path: str = None):
        """
        Export validation results to CSV

        Args:
            results: List of validation results
            output_path: Optional path to save the CSV file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"validation_results_{timestamp}.csv")

        df = self.export_to_dataframe(results)
        df.to_csv(output_path, index=False)

        self.logger.info(f"Validation results exported to CSV: {output_path}")

    def calculate_sim_to_real_gap(self, sim_results: Dict[str, Any], real_results: Dict[str, Any]) -> float:
        """
        Calculate the sim-to-real performance gap

        Args:
            sim_results: Performance results from simulation
            real_results: Performance results from real world

        Returns:
            Sim-to-real gap (absolute difference in performance)
        """
        # Calculate gap as absolute difference in key metrics
        reward_gap = abs(sim_results.get('average_reward', 0) - real_results.get('average_reward', 0))
        success_gap = abs(sim_results.get('success_rate', 0) - real_results.get('success_rate', 0))
        distance_gap = abs(sim_results.get('average_distance_traveled', 0) -
                          real_results.get('average_distance_traveled', 0))

        # Weighted combination of gaps (weights can be adjusted)
        gap = 0.4 * reward_gap + 0.4 * success_gap + 0.2 * distance_gap

        self.logger.info(f"Sim-to-real gap calculated: {gap:.4f}")
        return gap

    def validate_sim_to_real_transfer(self, sim_results: Dict[str, Any],
                                   real_results: Dict[str, Any],
                                   max_gap: float = 0.1) -> bool:
        """
        Validate if sim-to-real transfer is successful

        Args:
            sim_results: Performance results from simulation
            real_results: Performance results from real world
            max_gap: Maximum acceptable sim-to-real gap

        Returns:
            True if sim-to-real transfer is considered successful
        """
        gap = self.calculate_sim_to_real_gap(sim_results, real_results)
        success = gap <= max_gap

        self.logger.info(f"Sim-to-real transfer validation: Gap={gap:.4f}, Max allowed={max_gap}, Result={'PASS' if success else 'FAIL'}")
        return success


class SimRealTransferValidator:
    """
    Specialized validator for sim-to-real transfer validation
    """
    def __init__(self, config_path: str = None):
        """
        Initialize the sim-to-real transfer validator

        Args:
            config_path: Path to configuration file
        """
        self.performance_validator = PerformanceValidator(config_path)
        self.logger = self.performance_validator.logger

    def validate_transfer(self, sim_policy, real_policy, sim_env, real_env) -> Dict[str, Any]:
        """
        Validate sim-to-real transfer performance

        Args:
            sim_policy: Policy trained in simulation
            real_policy: Policy deployed on real robot (or simulation with real-world parameters)
            sim_env: Simulation environment
            real_env: Real-world environment (or simulation with real-world parameters)

        Returns:
            Dictionary with transfer validation results
        """
        self.logger.info("Starting sim-to-real transfer validation...")

        # Validate simulation policy in simulation
        self.logger.info("Validating simulation policy in simulation environment...")
        sim_results = self.performance_validator.validate_policy_performance(
            sim_policy, sim_env, "Simulation Policy"
        )

        # Validate real policy in real environment
        self.logger.info("Validating real policy in real environment...")
        real_results = self.performance_validator.validate_policy_performance(
            real_policy, real_env, "Real Policy"
        )

        # Calculate sim-to-real gap
        gap = self.performance_validator.calculate_sim_to_real_gap(sim_results, real_results)

        # Validate transfer success
        transfer_success = self.performance_validator.validate_sim_to_real_transfer(
            sim_results, real_results
        )

        # Generate comprehensive report
        transfer_report = {
            'simulation_results': sim_results,
            'real_world_results': real_results,
            'sim_to_real_gap': gap,
            'transfer_success': transfer_success,
            'recommendations': self._generate_recommendations(sim_results, real_results, gap)
        }

        self.logger.info(f"Sim-to-real transfer validation completed. Success: {transfer_success}")
        return transfer_report

    def _generate_recommendations(self, sim_results: Dict[str, Any],
                               real_results: Dict[str, Any],
                               gap: float) -> List[str]:
        """
        Generate recommendations based on transfer validation results

        Args:
            sim_results: Simulation validation results
            real_results: Real-world validation results
            gap: Sim-to-real performance gap

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check if gap is too large
        if gap > 0.15:  # 15% gap threshold
            recommendations.append("Sim-to-real gap is large (>15%). Consider increasing domain randomization.")

        # Check if real performance is poor
        if real_results['success_rate'] < 0.5:
            recommendations.append("Real-world success rate is low (<50%). Consider additional real-world training.")

        # Check if simulation performance is much better than real
        if sim_results['success_rate'] - real_results['success_rate'] > 0.3:
            recommendations.append("Large performance difference between sim and real. Consider adjusting simulation parameters.")

        # Check energy efficiency differences
        if abs(sim_results['energy_efficiency'] - real_results['energy_efficiency']) > 0.5:
            recommendations.append("Energy efficiency differs significantly between sim and real. Consider refining dynamics models.")

        if not recommendations:
            recommendations.append("Sim-to-real transfer appears successful. No major improvements needed.")

        return recommendations


def main():
    """
    Example usage of the PerformanceValidator
    """
    print("Performance Validation Framework for Humanoid Robot Control")
    print("=" * 60)

    # Example configuration for validation
    validation_config = {
        'validation_episodes': 20,  # Reduced for example
        'render': False,
        'metrics': {
            'reward': True,
            'success_rate': True,
            'distance': True,
            'energy': True,
            'stability': True,
            'tracking': True
        },
        'success_criteria': {
            'min_success_rate': 0.7,
            'min_distance': 1.0,
            'max_energy_per_meter': 150.0,
            'min_stability_score': 0.6
        },
        'logging': {
            'log_dir': './logs/performance_validation',
            'save_plots': True,
            'save_results': True
        }
    }

    # Create validator
    validator = PerformanceValidator(config=validation_config)

    print("Performance validator initialized with configuration:")
    print(f"  - Validation episodes: {validation_config['validation_episodes']}")
    print(f"  - Success criteria: {validation_config['success_criteria']}")
    print(f"  - Log directory: {validation_config['logging']['log_dir']}")

    # Example: Create mock validation results (in real usage, you would validate actual policies)
    mock_results = [
        {
            'policy_name': 'Policy_1',
            'total_episodes': validation_config['validation_episodes'],
            'success_rate': 0.85,
            'average_reward': 250.0,
            'average_episode_length': 350,
            'average_distance_traveled': 3.2,
            'average_energy_consumption': 85.5,
            'average_stability_score': 0.82,
            'average_tracking_accuracy': 0.88,
            'energy_efficiency': 26.7,
            'validation_passed': True
        },
        {
            'policy_name': 'Policy_2',
            'total_episodes': validation_config['validation_episodes'],
            'success_rate': 0.72,
            'average_reward': 180.0,
            'average_episode_length': 320,
            'average_distance_traveled': 2.8,
            'average_energy_consumption': 95.2,
            'average_stability_score': 0.75,
            'average_tracking_accuracy': 0.82,
            'energy_efficiency': 34.0,
            'validation_passed': True
        },
        {
            'policy_name': 'Policy_3',
            'total_episodes': validation_config['validation_episodes'],
            'success_rate': 0.45,
            'average_reward': 120.0,
            'average_episode_length': 280,
            'average_distance_traveled': 1.5,
            'average_energy_consumption': 120.0,
            'average_stability_score': 0.52,
            'average_tracking_accuracy': 0.65,
            'energy_efficiency': 80.0,
            'validation_passed': False
        }
    ]

    print(f"\nProcessing {len(mock_results)} mock validation results")

    # Generate report
    report_path = validator.generate_validation_report(mock_results)
    print(f"Report generated: {report_path}")

    # Create visualizations
    validator.plot_validation_results(mock_results)

    # Export to CSV
    validator.export_results_csv(mock_results)

    # Print performance summary
    summary = validator.get_performance_summary(mock_results)
    print(f"\nPerformance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Test sim-to-real transfer validation
    print(f"\nTesting sim-to-real transfer validation...")
    sim_real_validator = SimRealTransferValidator()

    # Mock sim and real results
    mock_sim_results = mock_results[0]  # Assume this is sim performance
    mock_real_results = {
        'policy_name': 'Real_Policy_1',
        'total_episodes': validation_config['validation_episodes'],
        'success_rate': 0.78,  # Slightly lower than sim
        'average_reward': 230.0,  # Slightly lower than sim
        'average_episode_length': 340,
        'average_distance_traveled': 2.9,
        'average_energy_consumption': 90.0,
        'average_stability_score': 0.78,
        'average_tracking_accuracy': 0.85,
        'energy_efficiency': 31.0,
        'validation_passed': True
    }

    transfer_report = sim_real_validator.validate_transfer(
        sim_policy=None, real_policy=None,
        sim_env=None, real_env=None  # Mock environments
    )

    print(f"Transfer validation report:")
    print(f"  - Gap: {transfer_report.get('sim_to_real_gap', 'N/A')}")
    print(f"  - Success: {transfer_report.get('transfer_success', 'N/A')}")

    print(f"\nPerformance validation framework setup complete!")
    print(f"Log directory: {validator.log_dir}")


if __name__ == "__main__":
    main()