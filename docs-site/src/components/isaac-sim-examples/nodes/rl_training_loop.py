#!/usr/bin/env python3
# rl_training_loop.py

"""
Isaac Lab Reinforcement Learning Training Loop for Humanoid Robot Control

This module implements the main training loop for reinforcement learning
using Isaac Lab and Isaac Sim. It orchestrates the training process for
humanoid robot control policies.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import yaml
import argparse
import datetime

# Isaac Sim imports
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp

# Isaac Gym imports
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.gym.tasks.utils.usd_utils import set_drive
from omni.isaac.gym.tasks.utils.differential_drive import WheelBaseEnv
from omni.isaac.gym.tasks.utils.terrain_analysis import TerrainAnalysis

# Import the RL environment from our training environment
from rl_training_env import HumanoidRLManager


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for humanoid control
    """
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims, critic_hidden_dims, activation="elu"):
        super(ActorCritic, self).__init__()

        # Activation function
        if activation == "elu":
            activation_func = nn.ELU
        elif activation == "relu":
            activation_func = nn.ReLU
        elif activation == "tanh":
            activation_func = nn.Tanh
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        # Actor network
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_func())

        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i+1]))
            actor_layers.append(activation_func())

        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic network
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_func())

        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i+1]))
            critic_layers.append(activation_func())

        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self):
        """Forward pass is not implemented for this class"""
        raise NotImplementedError

    def act(self, observations):
        """Get actions from the actor network"""
        mean_actions = self.actor(observations)
        return mean_actions

    def get_value(self, observations):
        """Get value from the critic network"""
        values = self.critic(observations)
        return values


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for humanoid control
    """
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims, critic_hidden_dims,
                 activation, learning_rate, clip_param, value_loss_coef, entropy_coef, max_grad_norm):
        # Create the actor-critic network
        self.actor_critic = ActorCritic(
            num_actor_obs, num_critic_obs, num_actions,
            actor_hidden_dims, critic_hidden_dims, activation
        ).to("cuda:0")

        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps=1e-5)

        # PPO parameters
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # For action noise
        self.action_std = 0.5  # Initial action standard deviation

    def update(self, batch):
        """
        Update the policy using PPO
        """
        obs_batch = batch["observations"]
        actions_batch = batch["actions"]
        old_values_batch = batch["values"]
        old_actions_log_prob_batch = batch["actions_log_prob"]
        advantages_batch = batch["advantages"]
        returns_batch = batch["returns"]

        # Get current policy values
        values = self.actor_critic.get_value(obs_batch).squeeze()

        # Get action distribution
        mean_actions = self.actor_critic.act(obs_batch)
        actions_dists = torch.distributions.Normal(mean_actions, self.action_std)

        # Calculate action log probabilities
        actions_log_prob = actions_dists.log_prob(actions_batch).sum(dim=-1)

        # Calculate entropy
        entropy = actions_dists.entropy().sum(dim=-1)

        # Calculate ratio
        ratio = torch.exp(actions_log_prob - old_actions_log_prob_batch)

        # Calculate surrogate objectives
        advantages = advantages_batch
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        action_loss = -torch.min(surr1, surr2).mean()

        # Calculate value loss
        value_pred_clipped = old_values_batch + (values - old_values_batch).clamp(-self.clip_param, self.clip_param)
        value_losses = (values - returns_batch).pow(2)
        value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        # Calculate entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'value_loss': value_loss.item(),
            'action_loss': action_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item()
        }


class RolloutBuffer:
    """
    Rollout buffer for storing trajectories
    """
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, action_shape, device="cuda:0"):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # Core storage
        self.observations = torch.zeros((num_transitions_per_env, num_envs, *obs_shape), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((num_transitions_per_env, num_envs, *action_shape), device=self.device, dtype=torch.float)
        self.rewards = torch.zeros((num_transitions_per_env, num_envs), device=self.device, dtype=torch.float)
        self.values = torch.zeros((num_transitions_per_env, num_envs), device=self.device, dtype=torch.float)
        self.actions_log_prob = torch.zeros((num_transitions_per_env, num_envs), device=self.device, dtype=torch.float)
        self.returns = torch.zeros((num_transitions_per_env, num_envs), device=self.device, dtype=torch.float)
        self.advantages = torch.zeros((num_transitions_per_env, num_envs), device=self.device, dtype=torch.float)

        self.step = 0

    def add_transitions(self, observations, actions, rewards, values, actions_log_prob):
        """
        Add transitions to the buffer
        """
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob)

        self.step += 1

    def clear(self):
        """
        Clear the buffer
        """
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """
        Compute returns using Generalized Advantage Estimation (GAE)
        """
        # Compute returns and advantages
        with torch.no_grad():
            last_gae_lam = 0
            for step in reversed(range(self.num_transitions_per_env)):
                if step == self.num_transitions_per_env - 1:
                    next_values = last_values
                else:
                    next_values = self.values[step + 1]

                delta = self.rewards[step] + gamma * next_values - self.values[step]
                last_gae_lam = delta + gamma * lam * last_gae_lam
                self.advantages[step] = last_gae_lam

            self.returns = self.advantages + self.values

    def get_statistics(self):
        """
        Get buffer statistics
        """
        done = self.step == self.num_transitions_per_env
        if not done:
            return -1

        # Mean, std and min of the returns
        returns = self.returns.cpu().numpy()
        return {
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'return_min': np.min(returns),
            'return_max': np.max(returns)
        }


class RLTrainingLoop:
    """
    Main RL Training Loop for Humanoid Robot Control
    """
    def __init__(self, config_path):
        """
        Initialize the training loop with configuration
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract parameters from config
        self.num_envs = self.config['general']['num_envs']
        self.max_iterations = self.config['training']['max_iterations']
        self.save_interval = self.config['training']['save_interval']
        self.print_stats = self.config['training']['print_stats']
        self.episode_length = self.config['general']['episode_length']
        self.num_learning_epochs = self.config['ppo']['num_learning_epochs']
        self.num_mini_batches = self.config['ppo']['num_mini_batches']
        self.gamma = self.config['ppo']['gamma']
        self.lam = self.config['ppo']['lam']
        self.learning_rate = self.config['ppo']['learning_rate']

        # Create environment
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize the environment
        self.env = HumanoidRLManager(
            task_cfg=self.config,
            sim_device=self.device,
            graphics_device=self.device,
            headless=False  # Set to True for headless training
        )

        # Get observation and action dimensions
        obs_dim = self.env.num_obs
        action_dim = self.env.num_actions

        # Create PPO agent
        self.agent = PPOAgent(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=self.config['policy']['network']['actor_hidden_dims'],
            critic_hidden_dims=self.config['policy']['network']['critic_hidden_dims'],
            activation=self.config['policy']['network']['activation'],
            learning_rate=self.learning_rate,
            clip_param=self.config['ppo']['clip_param'],
            value_loss_coef=self.config['ppo']['value_loss_coef'],
            entropy_coef=self.config['ppo']['entropy_coef'],
            max_grad_norm=self.config['ppo']['max_grad_norm']
        )

        # Create rollout buffer
        self.rollout_buffer = RolloutBuffer(
            num_envs=self.num_envs,
            num_transitions_per_env=self.episode_length,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
            device=self.device
        )

        # Initialize logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f"logs/humanoid_ppo_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.iteration = 0

    def run_training(self):
        """
        Main training loop
        """
        print("Starting RL Training for Humanoid Robot Control")
        print("=" * 50)

        # Get initial observations
        obs_dict = self.env.get_observations()
        obs = obs_dict[self.env.task._robot.name]

        # Training loop
        for iteration in range(self.max_iterations):
            self.iteration = iteration

            # Collect rollouts
            for step in range(self.episode_length):
                # Get actions from the policy
                with torch.no_grad():
                    mean_actions = self.agent.actor_critic.act(obs)

                    # Add noise for exploration
                    noise = torch.randn_like(mean_actions) * self.agent.action_std
                    actions = torch.tanh(mean_actions + noise)

                    # Get values from critic
                    values = self.agent.actor_critic.get_value(obs).squeeze()

                    # Calculate action log probabilities
                    actions_dist = torch.distributions.Normal(mean_actions, self.agent.action_std)
                    actions_log_prob = actions_dist.log_prob(actions).sum(dim=-1)

                # Store the current transition
                self.rollout_buffer.add_transitions(
                    observations=obs,
                    actions=actions,
                    rewards=torch.zeros(self.num_envs, device=self.device),
                    values=values,
                    actions_log_prob=actions_log_prob
                )

                # Apply actions to the environment
                self.env.pre_physics_step(actions)

                # Step the simulation
                self.env.post_physics_step()

                # Get next observations
                next_obs_dict = self.env.get_observations()
                next_obs = next_obs_dict[self.env.task._robot.name]

                # Calculate rewards
                rewards = self.env.task.calculate_metrics(next_obs_dict)

                # Update rewards in the buffer
                self.rollout_buffer.rewards[step] = rewards

                # Check for done environments
                dones = self.env.task.is_done()

                # Store episode statistics
                if dones.any():
                    # Calculate episode rewards for done environments
                    episode_rewards = rewards[dones].cpu().numpy()
                    self.episode_rewards.extend(episode_rewards)

                    # Update episode lengths for done environments
                    self.episode_lengths.extend([self.episode_length] * len(episode_rewards))

                # Update observations
                obs = next_obs

            # Compute returns and advantages
            with torch.no_grad():
                last_values = self.agent.actor_critic.get_value(obs).squeeze()
            self.rollout_buffer.compute_returns(last_values, self.gamma, self.lam)

            # Update the policy
            mean_value_loss = 0
            mean_action_loss = 0
            mean_entropy_loss = 0

            for epoch in range(self.num_learning_epochs):
                # Shuffle the data
                batch_size = self.num_envs * self.episode_length
                mini_batch_size = batch_size // self.num_mini_batches

                # Create random indices
                indices = torch.randperm(batch_size, device=self.device)

                for i in range(self.num_mini_batches):
                    start_idx = i * mini_batch_size
                    end_idx = (i + 1) * mini_batch_size
                    batch_idx = indices[start_idx:end_idx]

                    # Reshape indices to (sequence_length, num_envs)
                    sequence_length = self.episode_length
                    num_envs_per_batch = mini_batch_size // sequence_length
                    batch_idx = batch_idx.view(sequence_length, num_envs_per_batch)

                    # Get batch
                    batch = {
                        'observations': self.rollout_buffer.observations.view(-1, *self.rollout_buffer.observations.shape[2:])[batch_idx],
                        'actions': self.rollout_buffer.actions.view(-1, *self.rollout_buffer.actions.shape[2:])[batch_idx],
                        'values': self.rollout_buffer.values.view(-1)[batch_idx],
                        'actions_log_prob': self.rollout_buffer.actions_log_prob.view(-1)[batch_idx],
                        'advantages': self.rollout_buffer.advantages.view(-1)[batch_idx],
                        'returns': self.rollout_buffer.returns.view(-1)[batch_idx]
                    }

                    # Update the agent
                    loss_dict = self.agent.update(batch)
                    mean_value_loss += loss_dict['value_loss']
                    mean_action_loss += loss_dict['action_loss']
                    mean_entropy_loss += loss_dict['entropy_loss']

            # Calculate mean losses
            mean_value_loss /= (self.num_learning_epochs * self.num_mini_batches)
            mean_action_loss /= (self.num_learning_epochs * self.num_mini_batches)
            mean_entropy_loss /= (self.num_learning_epochs * self.num_mini_batches)

            # Log training statistics
            if self.print_stats and iteration % 10 == 0:
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0

                print(f"Iteration: {iteration}, Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.2f}")
                print(f"Value Loss: {mean_value_loss:.4f}, Action Loss: {mean_action_loss:.4f}, Entropy Loss: {mean_entropy_loss:.4f}")

                # Log to tensorboard
                self.writer.add_scalar('Train/mean_reward', mean_reward, iteration)
                self.writer.add_scalar('Train/mean_length', mean_length, iteration)
                self.writer.add_scalar('Loss/value_loss', mean_value_loss, iteration)
                self.writer.add_scalar('Loss/action_loss', mean_action_loss, iteration)
                self.writer.add_scalar('Loss/entropy_loss', mean_entropy_loss, iteration)

            # Save model periodically
            if iteration % self.save_interval == 0:
                self.save_model(f"checkpoints/humanoid_ppo_{iteration}.pth")

            # Decay action noise over time
            if iteration % 100 == 0:
                self.agent.action_std = max(0.1, self.agent.action_std * 0.995)

            # Clear the buffer for next iteration
            self.rollout_buffer.clear()

        # Save final model
        self.save_model("checkpoints/humanoid_ppo_final.pth")
        print("Training completed!")

    def save_model(self, path):
        """
        Save the trained model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'iteration': self.iteration,
            'actor_critic_state_dict': self.agent.actor_critic.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load a trained model
        """
        checkpoint = torch.load(path)
        self.agent.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        print(f"Model loaded from {path}")


def main():
    """
    Main function to run the RL training loop
    """
    parser = argparse.ArgumentParser(description='RL Training for Humanoid Robot Control')
    parser.add_argument('--config', type=str, default='config/rl_policy_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint to resume training from')

    args = parser.parse_args()

    # Create and run the training loop
    training_loop = RLTrainingLoop(args.config)

    if args.resume:
        training_loop.load_model(args.resume)

    training_loop.run_training()


if __name__ == "__main__":
    main()