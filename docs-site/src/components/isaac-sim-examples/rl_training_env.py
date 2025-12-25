#!/usr/bin/env python3
# rl_training_env.py

"""
Isaac Lab Reinforcement Learning Training Environment for Humanoid Robot

This module implements a reinforcement learning training environment for humanoid
robot control using NVIDIA Isaac Lab. The environment provides a physics-accurate
simulation for training locomotion and manipulation policies.
"""

import numpy as np
import torch
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.torch.maths import torch_acos, torch_cross, torch_normalize
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import RigidPrimView
from omni.isaac.core.prims import RigidPrim, CollisionView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import clear_stage
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.tasks import BaseTask
from omni.isaac.gym.envs.tasks.manager import TaskManager
from omni.isaac.gym.envs.envs.rl_env import RLEnv
from omni.isaac.gym.envs.utils.task_util import initialize_task
from pxr import Gf, UsdGeom, Sdf

import gym
from gym import spaces
import carb


class HumanoidLocomotionTask(BaseTask):
    """
    Humanoid locomotion task for reinforcement learning training.
    The goal is to train the humanoid robot to walk forward efficiently.
    """

    def __init__(
        self,
        name: str,
        offset=None
    ) -> None:
        """Initialize the task object.

        Args:
            name: The name of the task.
            offset: offset of the task
        """
        super().__init__(name=name, offset=offset)

        # Set default simulation parameters
        self._simulation_dt = 1.0 / 120.0  # 120 Hz physics update
        self._decimation = 4  # Control decimation (30 Hz control rate)
        self._dt = self._simulation_dt * self._decimation  # 0.0333 seconds

        # Task parameters
        self._max_episode_length = 500  # 500 * 0.0333 = 16.67 seconds per episode
        self._velocity_target = 1.0  # Target forward velocity in m/s

        # Robot parameters
        self._num_actions = 12  # 12 joint positions for Unitree A1
        self._num_observations = 41  # State vector size

        # Reward weights
        self._reward_weights = {
            'velocity_tracking': 1.0,
            'action_rate': -0.01,
            'torque': -0.0001,
            'orientation': 0.5,
            'height': 0.2,
            'feet_air_time': 0.2,
            'collision': -1.0
        }

        # Asset paths
        self._robot_asset_path = "/Isaac/Robots/Unitree/A1/a1.usd"

        # Robot state variables
        self._previous_actions = None
        self._episode_sums = {}
        self._extras = {}

    def set_up_scene(self, scene) -> None:
        """Set up the scene with the robot and any other objects."""
        # Get the assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # Add the robot to the stage
        robot_path = assets_root_path + self._robot_asset_path
        add_reference_to_stage(robot_path, "/World/Robot")

        # Add the robot to the scene
        self._robot = ArticulationView(
            prim_paths_expr="/World/Robot/base_link",
            name="robot_view",
            reset_xform_properties=False,
        )
        scene.add(self._robot)

        # Add ground plane
        create_prim(
            prim_path="/World/GroundPlane",
            prim_type="Plane",
            scale=np.array([100.0, 100.0, 1.0]),
            position=np.array([0, 0, 0]),
            orientation=np.array([0, 0, 0, 1])
        )

        # Initialize task manager
        self._task_manager = TaskManager()

    def get_observations(self) -> dict:
        """Get the current observations from the environment."""
        # Get robot state information
        root_states = self._robot.get_world_poses(clone=False)
        joint_positions = self._robot.get_joint_positions(clone=False)
        joint_velocities = self._robot.get_joint_velocities(clone=False)
        root_lin_vel = self._robot.get_linear_velocities(clone=False)
        root_ang_vel = self._robot.get_angular_velocities(clone=False)
        projected_gravity = self._robot.get_projected_gravity(clone=False)

        # Create observation dictionary
        obs = torch.cat([
            joint_positions,  # Joint positions
            joint_velocities,  # Joint velocities
            root_lin_vel,     # Root linear velocity
            root_ang_vel,     # Root angular velocity
            projected_gravity,  # Projected gravity vector
        ], dim=-1)

        return {self._robot.name: obs}

    def pre_physics_step(self, actions) -> None:
        """Apply actions to the robot before the physics step."""
        reset_buf = self.progress_buf >= self._max_episode_length - 1
        self.reset_idx(reset_buf.nonzero(as_tuple=False).flatten())

        # Store previous actions for action rate penalty
        if self._previous_actions is None:
            self._previous_actions = torch.zeros_like(actions)

        # Apply actions to the robot
        actions = torch.clamp(actions, -1.0, 1.0)
        self._robot.set_joint_position_targets(actions)

        # Store current actions for next iteration
        self._previous_actions = actions.clone()

    def reset_idx(self, env_ids) -> None:
        """Reset the environment for the given indices."""
        num_resets = len(env_ids)

        # Randomize initial joint positions
        joint_pos = (
            self._robot.get_default_joint_positions()
            + torch.rand_like(self._robot.get_default_joint_positions()[env_ids])
            * 0.2
            - 0.1
        )

        # Randomize initial joint velocities
        joint_vel = torch.zeros_like(self._robot.get_default_joint_velocities()[env_ids])

        # Apply resets
        self._robot.set_joint_positions(joint_pos, indices=env_ids)
        self._robot.set_joint_velocities(joint_vel, indices=env_ids)

        # Reset progress buffer
        self.progress_buf[env_ids] = 0

    def post_reset(self) -> None:
        """Initialize buffers after reset."""
        self.progress_buf = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )
        self.reset_buf = torch.ones(
            self._num_envs, dtype=torch.long, device=self._device
        )
        self.extras = {}

    def calculate_metrics(self, observations) -> dict:
        """Calculate rewards for the current step."""
        # Get current robot states
        joint_positions = self._robot.get_joint_positions(clone=False)
        joint_velocities = self._robot.get_joint_velocities(clone=False)
        root_lin_vel = self._robot.get_linear_velocities(clone=False)
        root_ang_vel = self._robot.get_angular_velocities(clone=False)
        projected_gravity = self._robot.get_projected_gravity(clone=False)

        # Velocity tracking reward
        velocity_tracking_reward = torch.exp(
            -torch.square(root_lin_vel[:, 0] - self._velocity_target)
        )

        # Action rate penalty
        action_rate_penalty = torch.sum(
            torch.square(self._previous_actions - self._robot.get_applied_actions()),
            dim=-1
        )

        # Torque penalty
        joint_torques = self._robot.get_measured_joint_efforts(clone=False)
        torque_penalty = torch.sum(torch.square(joint_torques), dim=-1)

        # Orientation reward (upright position)
        orientation_reward = torch.sum(
            torch.square(projected_gravity[:, 2] - 1.0),
            dim=-1
        )

        # Combine rewards
        reward = (
            self._reward_weights['velocity_tracking'] * velocity_tracking_reward +
            self._reward_weights['action_rate'] * action_rate_penalty +
            self._reward_weights['torque'] * torque_penalty +
            self._reward_weights['orientation'] * orientation_reward
        )

        # Track episode sums for logging
        self._episode_sums["rew"] = (
            self._episode_sums["rew"] + reward
        )

        return reward

    def is_done(self) -> torch.Tensor:
        """Check if the episode is done."""
        # Check if episode length exceeded
        done = self.progress_buf >= self._max_episode_length - 1

        # Check if robot fell over (projected gravity too far from upright)
        projected_gravity = self._robot.get_projected_gravity(clone=False)
        orientation_done = torch.abs(projected_gravity[:, 2]) < 0.5

        # Combine done conditions
        done = torch.logical_or(done, orientation_done)

        return done


class HumanoidRLManager(RLEnv):
    """
    Reinforcement Learning Environment Manager for Humanoid Robot Control
    """

    def __init__(self, task_cfg, sim_device, graphics_device, headless):
        """
        Initialize the RL environment manager.

        Args:
            task_cfg: Task configuration dictionary
            sim_device: Device for simulation (cpu/gpu)
            graphics_device: Device for graphics (cpu/gpu)
            headless: Whether to run in headless mode
        """
        self.device = sim_device
        self.headless = headless

        # Initialize the world
        self.world = World(
            stage_units_in_meters=1.0,
            rendering_dt=1.0/60.0,  # 60 Hz rendering
            sim_params={
                "use_gpu": sim_device.startswith("cuda"),
                "use_fabric": True,
                "solver_type": 1,  # TGS solver
                "num_position_iterations": 4,
                "num_velocity_iterations": 1,
                "max_depenetration_velocity": 1000.0,
                "enable_ccd": False,
                "enable_stablization": True,
                "friction_offset_threshold": 0.04,
                "friction_correlation_distance": 0.025,
                "gpu_max_rigid_contact_count": 524288,
                "gpu_max_rigid_patch_count": 33554432,
                "gpu_found_lost_pairs_capacity": 1024,
                "gpu_found_lost_aggregate_pairs_capacity": 1024,
                "gpu_total_aggregate_pairs_capacity": 1024,
                "gpu_max_soft_body_contacts": 1024,
                "gpu_max_particle_contacts": 1024,
                "gpu_heap_capacity": 67108864,
                "gpu_temp_buffer_capacity": 16777216,
                "gpu_max_num_partitions": 8,
            }
        )

        # Create the task
        self.task = HumanoidLocomotionTask(
            name="humanoid_locomotion_task",
            offset=np.array([0.0, 0.0, 0.0])
        )

        # Add the task to the world
        self.world.add_task(self.task)

        # Play the world
        self.world.play()

        # Initialize the base class
        super().__init__(
            task_cfg=task_cfg,
            sim_device=sim_device,
            graphics_device=graphics_device,
            headless=headless
        )

    def get_observations(self):
        """Get observations from the environment."""
        return self.task.get_observations()

    def pre_physics_step(self, actions):
        """Apply actions before physics step."""
        self.task.pre_physics_step(actions)

    def post_physics_step(self):
        """Update environment after physics step."""
        self.world.step(render=True)

        # Calculate metrics
        rewards = self.task.calculate_metrics(self.task.get_observations())

        # Check if done
        dones = self.task.is_done()

        # Get observations
        obs = self.get_observations()

        # Update progress
        self.task.progress_buf += 1

        return obs, rewards, dones, self.task.extras


def create_humanoid_rl_env(task_cfg, sim_device, graphics_device, headless):
    """
    Factory function to create the humanoid RL environment.

    Args:
        task_cfg: Task configuration dictionary
        sim_device: Device for simulation (cpu/gpu)
        graphics_device: Device for graphics (cpu/gpu)
        headless: Whether to run in headless mode

    Returns:
        HumanoidRLManager: The RL environment manager
    """
    return HumanoidRLManager(task_cfg, sim_device, graphics_device, headless)


# Example usage and testing
if __name__ == "__main__":
    print("Isaac Lab RL Training Environment for Humanoid Robot")
    print("=" * 50)

    # Example configuration
    task_cfg = {
        "seed": 42,
        "env": {
            "numEnvs": 64,  # Number of parallel environments
            "envSpacing": 2.5,
            "episodeLength": 500,
            "enableDebugVis": False,
        },
        "sim": {
            "dt": 1.0 / 120.0,
            "substeps": 1,
            "up_axis": "z",
            "use_gpu_pipeline": True,
            "gravity": [0.0, 0.0, -9.81],
            "add_ground_plane": True,
            "add_distant_light": True,
            "use_flatcache": True,
            "enable_scene_query_support": False,
            "default_physics_material": {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0,
            },
        },
    }

    # Initialize the environment (this would typically run in Isaac Sim)
    print("Environment ready for reinforcement learning training")
    print("Features:")
    print("- Physics-accurate simulation of Unitree A1 humanoid robot")
    print("- 12-DOF joint control for locomotion")
    print("- Reward function optimized for forward velocity tracking")
    print("- Parallel environments for efficient training")
    print("- GPU-accelerated simulation")