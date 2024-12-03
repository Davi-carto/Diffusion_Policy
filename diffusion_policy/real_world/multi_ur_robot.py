import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from diffusion_policy.real_world.rtde_interpolation_controller import (
    RTDEInterpolationController, Command)

class MultiURRobot:
    """
    Wrapper for controlling multiple UR robots simultaneously
    """
    def __init__(self,
            shm_manager: SharedMemoryManager,
            robot_ips,
            n_robots=2,
            frequency=125,
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=0.25,
            max_rot_speed=0.16,
            launch_timeout=3,
            tcp_offset_poses=None,
            payload_masses=None,
            payload_cogs=None,
            joints_inits=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128
            ):
        """
        Parameters are similar to RTDEInterpolationController but accept lists for multiple robots
        
        Args:
            robot_ips: list of robot IP addresses
            n_robots: number of robots to control (default: 2)
            tcp_offset_poses: list of 6d poses or None
            payload_masses: list of masses or None
            payload_cogs: list of 3d positions or None
            joints_inits: list of 6d joint positions or None
        """
        self.n_robots = n_robots
        if not isinstance(robot_ips, (list, tuple)):
            robot_ips = [robot_ips]
        
        if len(robot_ips) != self.n_robots:
            raise ValueError(f"Expected {self.n_robots} robot IPs, but got {len(robot_ips)}")
        
        self.controllers = []
        
        def validate_param(param, param_name):
            if param is not None:
                if not isinstance(param, (list, tuple)):
                    param = [param] * self.n_robots
                if len(param) != self.n_robots:
                    raise ValueError(
                        f"Expected {self.n_robots} values for {param_name}, but got {len(param)}")
            else:
                param = [None] * self.n_robots
            return param
        
        tcp_offset_poses = validate_param(tcp_offset_poses, 'tcp_offset_poses')
        payload_masses = validate_param(payload_masses, 'payload_masses')
        payload_cogs = validate_param(payload_cogs, 'payload_cogs')
        joints_inits = validate_param(joints_inits, 'joints_inits')
        
        for i in range(self.n_robots):
            controller = RTDEInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ips[i],
                frequency=frequency,
                lookahead_time=lookahead_time,
                gain=gain,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                launch_timeout=launch_timeout,
                tcp_offset_pose=tcp_offset_poses[i],
                payload_mass=payload_masses[i],
                payload_cog=payload_cogs[i],
                joints_init=joints_inits[i],
                joints_init_speed=joints_init_speed,
                soft_real_time=soft_real_time,
                verbose=verbose,
                receive_keys=receive_keys,
                get_max_k=get_max_k
            )
            self.controllers.append(controller)

    def start(self, wait=True):
        """Start all robot controllers"""
        for controller in self.controllers:
            controller.start(wait=wait)

    def stop(self, wait=True):
        """Stop all robot controllers"""
        for controller in self.controllers:
            controller.stop(wait=wait)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def is_ready(self):
        """Check if all controllers are ready"""
        return all(controller.is_ready for controller in self.controllers)

    def servoL(self, poses, durations=None):
        """
        Send servoL commands to all robots
        
        Args:
            poses: list of 6d poses for each robot
            durations: list of durations or single duration for all robots
        """
        if durations is None:
            durations = [0.1] * self.n_robots
        elif isinstance(durations, (int, float)):
            durations = [durations] * self.n_robots

        assert len(poses) == self.n_robots
        assert len(durations) == self.n_robots

        for controller, pose, duration in zip(self.controllers, poses, durations):
            controller.servoL(pose, duration)

    def schedule_waypoint(self, poses, target_times):
        """
        Schedule waypoints for all robots
        
        Args:
            poses: list of 6d poses for each robot
            target_times: list of target times or single target time for all robots
        """
        if isinstance(target_times, (int, float)):
            target_times = [target_times] * self.n_robots

        assert len(poses) == self.n_robots
        assert len(target_times) == self.n_robots

        for controller, pose, target_time in zip(self.controllers, poses, target_times):
            controller.schedule_waypoint(pose, target_time)

    def get_state(self, k=None, out=None):
        """
        Get state from all robots
        
        Returns:
            list of states from each robot
        """
        states = []
        for controller in self.controllers:
            state = controller.get_state(k=k, out=out)
            states.append(state)
        return states

    def get_all_state(self):
        """
        Get all states from all robots
        
        Returns:
            list of all states from each robot
        """
        all_states = []
        for controller in self.controllers:
            states = controller.get_all_state()
            all_states.append(states)
        return all_states
