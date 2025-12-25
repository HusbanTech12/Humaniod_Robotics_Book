#!/usr/bin/env python3
# sim_real_comparator.py

"""
Isaac Lab Sim-to-Real Comparison Tools for Humanoid Robot Control

This module provides tools for comparing simulation and real-world performance
of humanoid robot control policies, enabling validation of sim-to-real transfer.
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
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class ComparisonMetrics:
    """
    Data class to store sim-to-real comparison metrics
    """
    metric_name: str
    sim_values: List[float]
    real_values: List[float]
    difference: List[float]
    correlation: float
    mse: float
    mae: float
    kl_divergence: float
    timestamp: float


class SimRealComparator:
    """
    Framework for comparing simulation and real-world performance of humanoid robot control policies
    """
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize the sim-to-real comparator

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
                'comparison_metrics': [
                    'reward',
                    'success_rate',
                    'distance_traveled',
                    'energy_consumption',
                    'stability_score',
                    'tracking_accuracy'
                ],
                'comparison_episodes': 50,
                'statistical_tests': {
                    'enable_ks_test': True,
                    'enable_t_test': True,
                    'enable_correlation': True,
                    'significance_level': 0.05
                },
                'visualization': {
                    'enable_plots': True,
                    'save_plots': True,
                    'plot_format': 'png'
                },
                'logging': {
                    'log_dir': './logs/sim_real_comparison',
                    'save_results': True
                }
            }

        # Initialize logging directory
        self.log_dir = self.config['logging']['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize comparison metrics tracking
        self.comparison_history: List[ComparisonMetrics] = []
        self.sim_data_history = defaultdict(list)
        self.real_data_history = defaultdict(list)

        # Set up logging
        self.setup_logging()

        # Initialize visualization
        self.init_visualization()

    def setup_logging(self):
        """
        Set up logging for sim-to-real comparison
        """
        # Create logger
        self.logger = logging.getLogger('SimRealComparator')
        self.logger.setLevel(logging.INFO)

        # Create file handler
        log_file = os.path.join(self.log_dir, 'sim_real_comparison.log')
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

    def init_visualization(self):
        """
        Initialize visualization settings
        """
        plt.style.use('seaborn-v0_8')
        self.fig_size = (12, 8)

    def compare_performance(self,
                          sim_policy,
                          real_policy,
                          sim_environment,
                          real_environment,
                          comparison_name: str = "comparison") -> Dict[str, Any]:
        """
        Compare performance between simulation and real-world policies

        Args:
            sim_policy: Policy trained/validated in simulation
            real_policy: Policy deployed in real world
            sim_environment: Simulation environment
            real_environment: Real-world environment (or simulation with real-world parameters)
            comparison_name: Name for this comparison run

        Returns:
            Dictionary containing comparison results
        """
        self.logger.info(f"Starting sim-to-real comparison: {comparison_name}")

        # Collect data from simulation environment
        self.logger.info("Collecting simulation data...")
        sim_data = self._collect_performance_data(sim_policy, sim_environment, 'simulation')

        # Collect data from real environment
        self.logger.info("Collecting real-world data...")
        real_data = self._collect_performance_data(real_policy, real_environment, 'real')

        # Perform statistical comparisons
        comparison_results = self._perform_statistical_comparison(sim_data, real_data)

        # Calculate similarity metrics
        similarity_metrics = self._calculate_similarity_metrics(sim_data, real_data)

        # Generate comprehensive comparison report
        comparison_report = {
            'comparison_name': comparison_name,
            'timestamp': time.time(),
            'sim_data': sim_data,
            'real_data': real_data,
            'statistical_comparison': comparison_results,
            'similarity_metrics': similarity_metrics,
            'sim_to_real_gap': self._calculate_sim_to_real_gap(sim_data, real_data),
            'transfer_success': self._assess_transfer_success(comparison_results, similarity_metrics)
        }

        self.logger.info(f"Sim-to-real comparison completed for {comparison_name}")
        self.logger.info(f"Transfer success: {comparison_report['transfer_success']}")

        return comparison_report

    def _collect_performance_data(self, policy, environment, data_type: str) -> Dict[str, List[float]]:
        """
        Collect performance data from a given policy and environment

        Args:
            policy: Policy to evaluate
            environment: Environment to test in
            data_type: Type of data ('simulation' or 'real')

        Returns:
            Dictionary containing collected performance data
        """
        episodes = self.config.get('comparison_episodes', 50)
        metrics = self.config.get('comparison_metrics', [
            'reward', 'success_rate', 'distance_traveled'
        ])

        data = {metric: [] for metric in metrics}

        for episode in range(episodes):
            self.logger.info(f"Collecting {data_type} data: Episode {episode + 1}/{episodes}")

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

            # Store collected metrics
            data['reward'].append(episode_reward)
            data['episode_length'].append(episode_length)
            data['distance_traveled'].append(episode_distance)
            data['energy_consumption'].append(episode_energy)
            data['stability_score'].append(stability_score)
            data['tracking_accuracy'].append(tracking_accuracy)

            # Calculate success rate (placeholder)
            success = self._check_success_criteria(
                episode_distance, stability_score, tracking_accuracy
            )
            data['success_rate'].append(1.0 if success else 0.0)

        return data

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
        return distance >= 1.0 and stability >= 0.5

    def _perform_statistical_comparison(self, sim_data: Dict[str, List[float]],
                                     real_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform statistical comparison between simulation and real-world data

        Args:
            sim_data: Performance data from simulation
            real_data: Performance data from real world

        Returns:
            Dictionary containing statistical comparison results
        """
        comparison_results = {}

        # Get metrics to compare
        metrics = self.config.get('comparison_metrics', sim_data.keys())

        for metric in metrics:
            if metric in sim_data and metric in real_data:
                sim_values = np.array(sim_data[metric])
                real_values = np.array(real_data[metric])

                # Calculate basic statistics
                comparison_results[metric] = {
                    'sim_mean': float(np.mean(sim_values)),
                    'real_mean': float(np.mean(real_values)),
                    'sim_std': float(np.std(sim_values)),
                    'real_std': float(np.std(real_values)),
                    'mean_difference': float(np.mean(sim_values) - np.mean(real_values)),
                    'relative_difference': float((np.mean(sim_values) - np.mean(real_values)) / np.mean(sim_values)) if np.mean(sim_values) != 0 else 0.0
                }

                # Perform statistical tests if enabled
                stats_config = self.config.get('statistical_tests', {})

                if stats_config.get('enable_ks_test', True):
                    # Kolmogorov-Smirnov test for distribution similarity
                    ks_stat, ks_p_value = stats.ks_2samp(sim_values, real_values)
                    comparison_results[metric]['ks_test'] = {
                        'statistic': float(ks_stat),
                        'p_value': float(ks_p_value),
                        'significant_difference': ks_p_value < stats_config.get('significance_level', 0.05)
                    }

                if stats_config.get('enable_t_test', True):
                    # T-test for mean difference significance
                    t_stat, t_p_value = stats.ttest_ind(sim_values, real_values)
                    comparison_results[metric]['t_test'] = {
                        'statistic': float(t_stat),
                        'p_value': float(t_p_value),
                        'significant_difference': t_p_value < stats_config.get('significance_level', 0.05)
                    }

                if stats_config.get('enable_correlation', True):
                    # Correlation between sim and real (if same number of samples)
                    if len(sim_values) == len(real_values):
                        correlation, corr_p_value = stats.pearsonr(sim_values, real_values)
                        comparison_results[metric]['correlation'] = {
                            'coefficient': float(correlation),
                            'p_value': float(corr_p_value),
                            'significant': corr_p_value < stats_config.get('significance_level', 0.05)
                        }

        return comparison_results

    def _calculate_similarity_metrics(self, sim_data: Dict[str, List[float]],
                                   real_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate similarity metrics between simulation and real-world data

        Args:
            sim_data: Performance data from simulation
            real_data: Performance data from real world

        Returns:
            Dictionary containing similarity metrics
        """
        similarity_metrics = {}

        # Get metrics to compare
        metrics = self.config.get('comparison_metrics', sim_data.keys())

        for metric in metrics:
            if metric in sim_data and metric in real_data:
                sim_values = np.array(sim_data[metric])
                real_values = np.array(real_data[metric])

                # Ensure same length for comparison
                min_len = min(len(sim_values), len(real_values))
                sim_values = sim_values[:min_len]
                real_values = real_values[:min_len]

                # Calculate similarity metrics
                similarity_metrics[metric] = {
                    'mse': float(mean_squared_error(sim_values, real_values)),
                    'mae': float(mean_absolute_error(sim_values, real_values)),
                    'rmse': float(np.sqrt(mean_squared_error(sim_values, real_values))),
                    'max_error': float(np.max(np.abs(sim_values - real_values))),
                    'mean_absolute_difference': float(np.mean(np.abs(sim_values - real_values))),
                    'cosine_similarity': float(np.dot(sim_values, real_values) /
                                             (np.linalg.norm(sim_values) * np.linalg.norm(real_values))),
                    'overlap_coefficient': self._calculate_overlap_coefficient(sim_values, real_values)
                }

        return similarity_metrics

    def _calculate_overlap_coefficient(self, sim_values: np.ndarray, real_values: np.ndarray) -> float:
        """
        Calculate overlap coefficient between two distributions

        Args:
            sim_values: Simulation values
            real_values: Real-world values

        Returns:
            Overlap coefficient (0.0 to 1.0, where 1.0 is perfect overlap)
        """
        # Estimate PDFs using histograms
        min_val = min(np.min(sim_values), np.min(real_values))
        max_val = max(np.max(sim_values), np.max(real_values))

        bins = np.linspace(min_val, max_val, 50)
        sim_hist, _ = np.histogram(sim_values, bins=bins, density=True)
        real_hist, _ = np.histogram(real_values, bins=bins, density=True)

        # Calculate overlap
        overlap = np.sum(np.minimum(sim_hist, real_hist)) * (bins[1] - bins[0])
        return float(overlap)

    def _calculate_sim_to_real_gap(self, sim_data: Dict[str, List[float]],
                                 real_data: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate the sim-to-real gap for each metric

        Args:
            sim_data: Performance data from simulation
            real_data: Performance data from real world

        Returns:
            Dictionary containing sim-to-real gap for each metric
        """
        gap_metrics = {}

        # Get metrics to compare
        metrics = self.config.get('comparison_metrics', sim_data.keys())

        for metric in metrics:
            if metric in sim_data and metric in real_data:
                sim_mean = np.mean(sim_data[metric])
                real_mean = np.mean(real_data[metric])

                # Calculate gap as relative difference
                if sim_mean != 0:
                    gap = abs(sim_mean - real_mean) / abs(sim_mean)
                else:
                    gap = abs(real_mean) if real_mean != 0 else 0.0

                gap_metrics[metric] = float(gap)

        return gap_metrics

    def _assess_transfer_success(self, comparison_results: Dict[str, Any],
                               similarity_metrics: Dict[str, Any],
                               max_gap_threshold: float = 0.15) -> bool:
        """
        Assess whether sim-to-real transfer was successful

        Args:
            comparison_results: Statistical comparison results
            similarity_metrics: Similarity metrics
            max_gap_threshold: Maximum acceptable gap threshold

        Returns:
            True if transfer is considered successful
        """
        # Calculate overall success based on multiple criteria
        gaps = []

        for metric, results in comparison_results.items():
            if 'relative_difference' in results:
                gaps.append(abs(results['relative_difference']))

        # Calculate mean gap across all metrics
        mean_gap = np.mean(gaps) if gaps else 1.0  # Default to failure if no gaps calculated

        # Check if mean gap is within threshold
        gap_success = mean_gap <= max_gap_threshold

        # Additional criteria could be added here
        # For example, check correlation, statistical significance, etc.

        return gap_success

    def compare_multiple_policies(self, policy_pairs: List[Tuple[Any, Any, str]],
                                sim_env, real_env) -> List[Dict[str, Any]]:
        """
        Compare multiple policy pairs (sim and real versions)

        Args:
            policy_pairs: List of tuples (sim_policy, real_policy, name)
            sim_env: Simulation environment
            real_env: Real-world environment

        Returns:
            List of comparison results for each policy pair
        """
        results = []
        for sim_policy, real_policy, name in policy_pairs:
            result = self.compare_performance(sim_policy, real_policy, sim_env, real_env, name)
            results.append(result)

        return results

    def generate_comparison_report(self, results: List[Dict[str, Any]],
                                 output_path: str = None) -> str:
        """
        Generate a comprehensive comparison report

        Args:
            results: List of comparison results to include in report
            output_path: Optional path to save the report

        Returns:
            Path to the generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"comparison_report_{timestamp}.json")

        # Prepare report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_comparisons': len(results),
            'config': self.config,
            'results': []
        }

        for result in results:
            report_data['results'].append(result)

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Comparison report saved to: {output_path}")
        return output_path

    def plot_comparison_results(self, results: List[Dict[str, Any]],
                              output_path: str = None):
        """
        Create visualizations of comparison results

        Args:
            results: List of comparison results to visualize
            output_path: Optional path to save the plot
        """
        if not results:
            self.logger.warning("No results to plot")
            return

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"comparison_plot_{timestamp}.png")

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sim-to-Real Performance Comparison', fontsize=16)

        # Extract data for plotting
        if results:
            result = results[0]  # Use first result for plotting example
            sim_data = result['sim_data']
            real_data = result['real_data']
            comparison = result['statistical_comparison']

            # Get first available metric for plotting
            metric = next(iter(sim_data.keys()), 'reward')

            if metric in sim_data and metric in real_data:
                sim_values = sim_data[metric]
                real_values = real_data[metric]

                # Plot 1: Distribution comparison
                axes[0, 0].hist(sim_values, alpha=0.7, label='Simulation', bins=20, density=True)
                axes[0, 0].hist(real_values, alpha=0.7, label='Real World', bins=20, density=True)
                axes[0, 0].set_title(f'Distribution Comparison: {metric}')
                axes[0, 0].set_xlabel(metric)
                axes[0, 0].set_ylabel('Density')
                axes[0, 0].legend()

                # Plot 2: Scatter plot (if same length)
                if len(sim_values) == len(real_values):
                    axes[0, 1].scatter(sim_values, real_values, alpha=0.6)
                    axes[0, 1].plot([min(sim_values), max(sim_values)],
                                   [min(sim_values), max(sim_values)], 'r--', lw=2)
                    axes[0, 1].set_xlabel('Simulation Values')
                    axes[0, 1].set_ylabel('Real World Values')
                    axes[0, 1].set_title(f'Scatter Plot: {metric}')

                # Plot 3: Time series comparison (first few episodes)
                min_len = min(len(sim_values), len(real_values), 20)  # Limit for readability
                x_vals = range(min_len)
                axes[1, 0].plot(x_vals, sim_values[:min_len], label='Simulation', marker='o')
                axes[1, 0].plot(x_vals, real_values[:min_len], label='Real World', marker='s')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel(metric)
                axes[1, 0].set_title(f'Time Series: {metric}')
                axes[1, 0].legend()

                # Plot 4: Difference metrics
                differences = [abs(s - r) for s, r in zip(sim_values[:min_len], real_values[:min_len])]
                axes[1, 1].bar(range(len(differences)), differences)
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel(f'Absolute Difference')
                axes[1, 1].set_title(f'Differences: {metric}')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Comparison plot saved to: {output_path}")

    def plot_similarity_heatmap(self, results: List[Dict[str, Any]],
                               output_path: str = None):
        """
        Create a heatmap showing similarity between simulation and real-world data

        Args:
            results: List of comparison results
            output_path: Optional path to save the plot
        """
        if not results:
            self.logger.warning("No results to plot")
            return

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"similarity_heatmap_{timestamp}.png")

        # Create similarity heatmap
        if results:
            result = results[0]  # Use first result for heatmap
            similarity_metrics = result['similarity_metrics']

            # Prepare data for heatmap
            metrics = list(similarity_metrics.keys())
            mse_values = [similarity_metrics[m]['mse'] for m in metrics]
            mae_values = [similarity_metrics[m]['mae'] for m in metrics]
            correlation_values = [similarity_metrics[m].get('cosine_similarity', 0) for m in metrics]

            # Create DataFrame for heatmap
            df = pd.DataFrame({
                'MSE': mse_values,
                'MAE': mae_values,
                'Cosine Similarity': correlation_values
            }, index=metrics)

            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.T, annot=True, cmap='RdYlGn_r', center=0.5,
                       fmt='.3f', cbar_kws={'label': 'Value'})
            plt.title('Sim-to-Real Similarity Metrics')
            plt.ylabel('Metrics')
            plt.xlabel('Performance Aspects')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Similarity heatmap saved to: {output_path}")

    def save_comparison_results(self, results: List[Dict[str, Any]],
                              output_path: str = None):
        """
        Save comparison results to a file

        Args:
            results: List of comparison results to save
            output_path: Optional path to save the results
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"comparison_results_{timestamp}.pkl")

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

        self.logger.info(f"Comparison results saved to: {output_path}")

    def load_comparison_results(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Load comparison results from a file

        Args:
            input_path: Path to the saved results file

        Returns:
            List of loaded comparison results
        """
        with open(input_path, 'rb') as f:
            results = pickle.load(f)

        self.logger.info(f"Comparison results loaded from: {input_path}")
        return results

    def get_comparison_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get a summary of comparison across all results

        Args:
            results: List of comparison results

        Returns:
            Dictionary with comparison summary
        """
        if not results:
            return {}

        # Extract metrics from all results
        all_gaps = []
        all_success_rates = []
        all_correlations = []

        for result in results:
            gaps = result.get('sim_to_real_gap', {})
            if gaps:
                all_gaps.append(list(gaps.values()))

            comparison = result.get('statistical_comparison', {})
            for metric, data in comparison.items():
                if 'correlation' in data and 'coefficient' in data['correlation']:
                    all_correlations.append(data['correlation']['coefficient'])

            all_success_rates.append(result.get('transfer_success', False))

        summary = {
            'total_comparisons': len(results),
            'policies_compared': [r['comparison_name'] for r in results],
            'average_gap': float(np.mean([np.mean(gaps) for gaps in all_gaps])) if all_gaps else 0.0,
            'gap_std': float(np.std([np.mean(gaps) for gaps in all_gaps])) if all_gaps else 0.0,
            'success_rate': float(np.mean(all_success_rates)),
            'average_correlation': float(np.mean(all_correlations)) if all_correlations else 0.0,
            'best_comparison': results[np.argmin([np.mean(list(r['sim_to_real_gap'].values()))
                                               for r in results])]['comparison_name'] if all_gaps else None,
            'worst_comparison': results[np.argmax([np.mean(list(r['sim_to_real_gap'].values()))
                                                for r in results])]['comparison_name'] if all_gaps else None
        }

        return summary

    def export_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Export comparison results to a pandas DataFrame

        Args:
            results: List of comparison results

        Returns:
            DataFrame containing comparison results
        """
        if not results:
            return pd.DataFrame()

        data = []
        for result in results:
            row = {
                'comparison_name': result['comparison_name'],
                'timestamp': datetime.fromtimestamp(result['timestamp']),
                'transfer_success': result['transfer_success'],
                'sim_to_real_gap_avg': np.mean(list(result['sim_to_real_gap'].values()))
            }

            # Add gap metrics for each performance metric
            for metric, gap_value in result['sim_to_real_gap'].items():
                row[f'gap_{metric}'] = gap_value

            # Add success rate
            row['sim_success_rate'] = np.mean(result['sim_data']['success_rate'])
            row['real_success_rate'] = np.mean(result['real_data']['success_rate'])

            data.append(row)

        return pd.DataFrame(data)

    def export_results_csv(self, results: List[Dict[str, Any]], output_path: str = None):
        """
        Export comparison results to CSV

        Args:
            results: List of comparison results
            output_path: Optional path to save the CSV file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f"comparison_results_{timestamp}.csv")

        df = self.export_to_dataframe(results)
        df.to_csv(output_path, index=False)

        self.logger.info(f"Comparison results exported to CSV: {output_path}")

    def calculate_fidelity_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate an overall fidelity score for sim-to-real transfer

        Args:
            results: List of comparison results

        Returns:
            Fidelity score (0.0 to 1.0, where 1.0 is perfect fidelity)
        """
        if not results:
            return 0.0

        # Calculate fidelity based on multiple factors
        gaps = []
        correlations = []

        for result in results:
            # Average gap across all metrics
            avg_gap = np.mean(list(result['sim_to_real_gap'].values()))
            gaps.append(avg_gap)

            # Average correlation across all metrics
            comparison = result['statistical_comparison']
            metric_correlations = []
            for metric, data in comparison.items():
                if 'correlation' in data and 'coefficient' in data['correlation']:
                    metric_correlations.append(data['correlation']['coefficient'])

            if metric_correlations:
                correlations.append(np.mean(metric_correlations))

        # Calculate fidelity score
        # Lower gaps = higher fidelity
        # Higher correlations = higher fidelity
        avg_gap = np.mean(gaps) if gaps else 1.0
        avg_corr = np.mean(correlations) if correlations else 0.0

        # Normalize gap (invert and scale to 0-1, where 0 gap = 1.0 fidelity)
        gap_score = max(0, 1 - avg_gap)  # Cap at 1.0

        # Combine scores (equal weight)
        fidelity_score = 0.5 * gap_score + 0.5 * avg_corr

        return min(1.0, fidelity_score)  # Cap at 1.0


def main():
    """
    Example usage of the SimRealComparator
    """
    print("Sim-to-Real Comparison Framework for Humanoid Robot Control")
    print("=" * 60)

    # Example configuration for comparison
    comparison_config = {
        'comparison_metrics': [
            'reward', 'success_rate', 'distance_traveled',
            'energy_consumption', 'stability_score'
        ],
        'comparison_episodes': 20,  # Reduced for example
        'statistical_tests': {
            'enable_ks_test': True,
            'enable_t_test': True,
            'enable_correlation': True,
            'significance_level': 0.05
        },
        'logging': {
            'log_dir': './logs/sim_real_comparison',
            'save_results': True
        }
    }

    # Create comparator
    comparator = SimRealComparator(config=comparison_config)

    print("Sim-to-real comparator initialized with configuration:")
    print(f"  - Comparison episodes: {comparison_config['comparison_episodes']}")
    print(f"  - Metrics: {comparison_config['comparison_metrics']}")
    print(f"  - Log directory: {comparison_config['logging']['log_dir']}")

    # Example: Create mock comparison results (in real usage, you would compare actual policies)
    mock_results = [
        {
            'comparison_name': 'Policy_1_Comparison',
            'timestamp': time.time(),
            'sim_data': {
                'reward': [250.0, 265.0, 240.0, 280.0, 270.0] * 4,  # 20 episodes
                'success_rate': [1.0, 1.0, 0.8, 1.0, 1.0] * 4,
                'distance_traveled': [3.2, 3.5, 2.8, 3.8, 3.4] * 4,
                'energy_consumption': [85.5, 90.2, 82.1, 95.8, 88.7] * 4,
                'stability_score': [0.82, 0.85, 0.78, 0.88, 0.84] * 4,
                'tracking_accuracy': [0.88, 0.90, 0.85, 0.92, 0.89] * 4
            },
            'real_data': {
                'reward': [230.0, 245.0, 225.0, 260.0, 250.0] * 4,  # 20 episodes
                'success_rate': [0.9, 0.9, 0.7, 0.9, 0.8] * 4,
                'distance_traveled': [2.9, 3.2, 2.5, 3.4, 3.0] * 4,
                'energy_consumption': [90.5, 95.2, 87.1, 100.8, 93.7] * 4,
                'stability_score': [0.78, 0.82, 0.74, 0.84, 0.80] * 4,
                'tracking_accuracy': [0.84, 0.86, 0.81, 0.88, 0.85] * 4
            },
            'statistical_comparison': {
                'reward': {
                    'sim_mean': 261.0,
                    'real_mean': 242.0,
                    'mean_difference': 19.0,
                    'relative_difference': 0.0728,
                    'ks_test': {'statistic': 0.15, 'p_value': 0.85, 'significant_difference': False},
                    't_test': {'statistic': 2.34, 'p_value': 0.025, 'significant_difference': True}
                }
            },
            'similarity_metrics': {
                'reward': {
                    'mse': 144.0,
                    'mae': 12.0,
                    'rmse': 12.0,
                    'cosine_similarity': 0.94
                }
            },
            'sim_to_real_gap': {
                'reward': 0.0728,
                'success_rate': 0.05,
                'distance_traveled': 0.095,
                'energy_consumption': 0.078,
                'stability_score': 0.049
            },
            'transfer_success': True
        }
    ]

    print(f"\nProcessing {len(mock_results)} mock comparison results")

    # Generate report
    report_path = comparator.generate_comparison_report(mock_results)
    print(f"Report generated: {report_path}")

    # Create visualizations
    comparator.plot_comparison_results(mock_results)
    comparator.plot_similarity_heatmap(mock_results)

    # Export to CSV
    comparator.export_results_csv(mock_results)

    # Print comparison summary
    summary = comparator.get_comparison_summary(mock_results)
    print(f"\nComparison Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Calculate fidelity score
    fidelity_score = comparator.calculate_fidelity_score(mock_results)
    print(f"\nOverall fidelity score: {fidelity_score:.3f}")

    print(f"\nSim-to-real comparison framework setup complete!")
    print(f"Log directory: {comparator.log_dir}")


if __name__ == "__main__":
    main()