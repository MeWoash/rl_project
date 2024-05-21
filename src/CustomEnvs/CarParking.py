# autopep8: off
from pathlib import Path
from tabnanny import check
from traceback import print_tb
import numpy as np
import gymnasium
import mujoco
from gymnasium.spaces import Box
from typing import Dict, Text, Tuple, Union
import sys
import os
import math
from enum import IntEnum
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__,'..','..').resolve()))
from CustomEnvs.Utils import *
from CustomEnvs.Indexes import *
from Rendering.RendererClass import Renderer
from Rendering.Utils import TextOverlay
from MJCFGenerator.Config import *
from PathsConfig import *

# autopep8: on


def calculate_reward(observation, init_car_distance, params, **kwargs):
    reward_info = {}
    
    
    exp_scale = np.exp(-observation[OBS_INDEX.DISTANCE_BEGIN]/params['exp_scale'])
    
    reward_info['angle_diff_weight_scaled'] = params['angle_diff_weight'] * exp_scale
    reward_info['dist_weight_scaled'] = params['dist_weight'] + params['angle_diff_weight']* (1-exp_scale)
    
    angle_diff_sum = np.sum(observation[OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1])
    angle_diff_punish = normalize_data(angle_diff_sum,
                                            0, reward_info['angle_diff_weight_scaled'],
                                            0, len(observation[OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1])*math.pi
                                            )
    
    
    # dist_clipped = np.clip(observation[OBS_INDEX.DISTANCE_BEGIN], 0, init_car_distance) 
    dist_diff_punish = normalize_data(observation[OBS_INDEX.DISTANCE_BEGIN],
        0, reward_info['dist_weight_scaled'],
        0, init_car_distance
        )
    
    reward_info['dist_reward'] = reward_info['dist_weight_scaled'] - dist_diff_punish
    reward_info['angle_diff_reward'] =reward_info['angle_diff_weight_scaled'] - angle_diff_punish

    reward = reward_info['dist_reward'] + reward_info['angle_diff_reward']
    
    return reward, reward_info

class CarParkingEnv(gymnasium.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "none"
        ],
    }
    
    # TRUNCATE
    
    
    def __init__(self,
                render_mode: str = "none",
                simulation_frame_skip: int = 4,
                time_limit = 30,
                enable_random_spawn = True,
                enable_spawn_noise = True,
                spawn_dist_noise = 0.5,
                spawn_angle_degrees_noise = 10,
                **kwargs):

        
        self.enable_random_spawn= enable_random_spawn
        self.enable_spawn_noise=enable_spawn_noise
        self.spawn_dist_noise = spawn_dist_noise
        self.spawn_angle_degrees_noise = np.radians(spawn_angle_degrees_noise)
        
        self.time_limit = time_limit
        self.simulation_frame_skip = simulation_frame_skip
        
        # RENDER VARIABLES
        self.render_mode = render_mode
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

        self.camera_name = None
        self.camera_id = 0
        self.episode_number = 0

        self._initialize_simulation()
        self._set_default_action_space()
        self._set_default_observation_space()
        self._calculate_spawn_points()

        # REWARD VARIABLES
        self.reward_range = (-1, 1)
        self.episode_mujoco_max_step = self.time_limit / self.model.opt.timestep
        self.episode_env_max_step = self.episode_mujoco_max_step/self.simulation_frame_skip
        
        self.episode_cumulative_reward = 0
        
        self.time_velocity_not_low = 0
        
        self.reward_params = {
            "dist_weight": 0.25,
            "angle_diff_weight": 0.75,
            "exp_scale":2,
            "max_step_reward": 1
        }
        
        assert\
              self.reward_params["dist_weight"] \
            + self.reward_params["angle_diff_weight"]\
                == self.reward_params["max_step_reward"], f'Weights have to sum to {self.reward_params["max_step_reward"]}'

    def _set_default_action_space(self):

        engine_bounds = np.array([-1,1]).reshape(2,1)
        angle_bounds = np.array([-1,1]).reshape(2,1)

        boundMatrix = np.zeros((2,ACTION_INDEX.ACTION_SIZE))
        boundMatrix[:, ACTION_INDEX.ENGINE:ACTION_INDEX.ENGINE+1] = engine_bounds
        boundMatrix[:, ACTION_INDEX.ANGLE:ACTION_INDEX.ANGLE+1] = angle_bounds

        self.action_space = Box(low=boundMatrix[0, :], high=boundMatrix[1, :], dtype=np.float32)
        return self.action_space

    def _set_default_observation_space(self):
        max_xy_dist = math.sqrt(MAP_LENGTH[0]**2 + MAP_LENGTH[1]**2)
        
        
        velocity_bounds = np.array([-10, 10]).reshape(2, 1)
        dist_bounds = np.array([0, max_xy_dist]).reshape(2, 1)
        angle_diff_bounds = np.tile(np.array([0, np.pi]).reshape(2, 1), (1, 2))
        range_sensors_bounds = np.tile(
            np.array([0, SENSORS_MAX_RANGE]).reshape(2, 1), (1, CAR_N_RANGE_SENSORS + TRAILER_N_RANGE_SENSORS))
        car_rel_pos_bounds = np.array(
            [[-MAP_LENGTH[0], -MAP_LENGTH[1]],
             [MAP_LENGTH[0], MAP_LENGTH[1]]])
        angle_bounds = np.tile(np.array([-np.pi, np.pi]).reshape(2, 1), (1, 2))

        # KEEP ORDER AS IN OBSINDEX
        boundMatrix = np.zeros((2,OBS_INDEX.OBS_SIZE))
        
        boundMatrix[:, OBS_INDEX.VELOCITY_BEGIN:OBS_INDEX.VELOCITY_END+1] = velocity_bounds
        boundMatrix[:, OBS_INDEX.DISTANCE_BEGIN:OBS_INDEX.DISTANCE_END+1] = dist_bounds
        boundMatrix[:, OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1] = angle_diff_bounds
        boundMatrix[:, OBS_INDEX.REL_POS_BEGIN:OBS_INDEX.REL_POS_END+1] = car_rel_pos_bounds
        boundMatrix[:, OBS_INDEX.YAW_BEGIN:OBS_INDEX.YAW_END+1] = angle_bounds
        boundMatrix[:, OBS_INDEX.RANGE_BEGIN:OBS_INDEX.RANGE_END+1] = range_sensors_bounds

        self.observation_space = Box(low=boundMatrix[0, :], high=boundMatrix[1, :], dtype=np.float32)
        return self.observation_space

    def _initialize_simulation(self):
        
        model_path = os.path.join(str(MJCF_OUT_DIR), MJCF_MODEL_NAME)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.mujoco_renderer: Renderer = Renderer(self)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) #  WTHOUT THIS LINE FIRST STEP CONSIST 0s

    def _calculate_spawn_points(self):
        self.spawn_points = []
        for spawn_params in CAR_SPAWN_KWARGS:
            pos = spawn_params['pos']
            quat = euler_to_quat(*np.radians(spawn_params['euler']))
            self.spawn_points.append([*pos, *quat])
            
            
    def render(self):
        o = TextOverlay()
        if self.render_mode == "human":
            self._prep_overlay(o)

        if self.render_mode != "none":
            return self.mujoco_renderer.render(self.render_mode, self.camera_id, o)


    def _prep_overlay(self, overlay: TextOverlay):
        overlay.add("Env Step",f"{self.episode_env_step:.0f} / {self.episode_env_max_step:.0f}", "bottom left")
        overlay.add("MuJoCo Step",f"{self.episode_mujoco_step:.0f} / {self.episode_mujoco_max_step:.0f}", "bottom left")
        overlay.add("episode", f"{self.episode_number}", "bottom left")
        overlay.add("time", f"{self.data.time:.2f} / {self.time_limit:.2f}", "bottom left")
        
        overlay.add("Env stats", "values", "top left")
        overlay.add("speed",f"{self.observation[OBS_INDEX.VELOCITY_BEGIN:OBS_INDEX.VELOCITY_END+1]}", "top left")
        overlay.add("dist",f"{self.observation[OBS_INDEX.DISTANCE_BEGIN:OBS_INDEX.DISTANCE_END+1]}", "top left")
        overlay.add("adiff",f"{self.observation[OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1]}", "top left")
        overlay.add("rel pos",f"{self.observation[OBS_INDEX.REL_POS_BEGIN:OBS_INDEX.REL_POS_END+1]}", "top left")
        overlay.add("eul",f"{self.observation[OBS_INDEX.YAW_BEGIN:OBS_INDEX.YAW_END+1]}", "top left")
        overlay.add("range car",f"{self.observation[OBS_INDEX.RANGE_BEGIN:OBS_INDEX.RANGE_BEGIN+CAR_N_RANGE_SENSORS]}", "top left")
        overlay.add("range trl",f"{self.observation[OBS_INDEX.RANGE_BEGIN+CAR_N_RANGE_SENSORS:OBS_INDEX.RANGE_END+1]}", "top left")
        
        overlay.add("extra_obs",f"{self.extra_observation}", "top left")
        
        
        overlay.add("Model Action", "values", "top right")
        overlay.add("engine", f"{self.action[ACTION_INDEX.ENGINE]:.2f}", "top right")
        overlay.add("wheel", f"{self.action[ACTION_INDEX.ANGLE]:.2f}", "top right")
        
        overlay.add("Reward metrics", "values", "bottom right")
        overlay.add("reward", f"{self.reward:.2f}", "bottom right")
        overlay.add("mean reward", f"{self.episode_mean_reward:.2f}", "bottom right")
        
        overlay.add("dist reward", f"{self.reward_info['dist_reward']:.2f} / {self.reward_info['dist_weight_scaled']:.2f}", "bottom right")
        overlay.add("angle reward", f"{self.reward_info['angle_diff_reward']:.2f}/{self.reward_info['angle_diff_weight_scaled']:.2f}", "bottom right")
        
    def close(self):
        """Close all processes like rendering contexts"""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()


    def _apply_forces(self, action):
        enginePowerCtrl = action[ACTION_INDEX.ENGINE]
        wheelsAngleCtrl = normalize_data(action[ACTION_INDEX.ANGLE], *WHEEL_ANGLE_LIMIT_RAD, -1, 1)

        self.data.actuator(f"{CAR_NAME}/engine").ctrl = enginePowerCtrl
        self.data.actuator(f"{CAR_NAME}/wheel1_angle").ctrl = wheelsAngleCtrl
        self.data.actuator(f"{CAR_NAME}/wheel2_angle").ctrl = wheelsAngleCtrl


    def _do_simulation(self, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """

        mujoco.mj_step(self.model, self.data, nstep=n_frames)
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _calculate_additional_variables(self):
        self.episode_mujoco_step =  int(round(self.data.time / self.model.opt.timestep))
        self.episode_mujoco_time = self.data.time
        self.episode_env_step = int(round(self.episode_mujoco_step / self.simulation_frame_skip))
    
    def step(self, action):
        self._calculate_additional_variables()
        
        self.action = action
        self._apply_forces(action)
        self._do_simulation(self.simulation_frame_skip)

        self.observation, self.extra_observation = self._get_obs()

        self.reward, self.reward_info = calculate_reward(self.observation, self.init_car_distance, self.reward_params)

        self.terminated = self._check_terminate_condition()
        self.truncated = self._check_truncated_condition()

        self.episode_cumulative_reward += self.reward
        self.episode_mean_reward = self.episode_cumulative_reward/(self.episode_env_step+1)
        
        renderRetVal = self.render()
        return self.observation, self.reward, self.terminated, self.truncated, self._get_info()

    def _get_info(self):
        info = {
            "extra_obs": self.extra_observation,
            
            "episode_mujoco_time": self.episode_mujoco_time,
            "episode_mujoco_step": self.episode_mujoco_step,
            "episode_env_step": self.episode_env_step,
            "episode_number":self.episode_number,
            "episode_cumulative_reward":self.episode_cumulative_reward,
            "episode_mean_reward": self.episode_mean_reward
            
                     }
        return info

    def _check_terminate_condition(self):
        terminated = False
        if self.observation[OBS_INDEX.DISTANCE_BEGIN] <= 0.5 \
                and abs(self.observation[OBS_INDEX.VELOCITY_BEGIN]) <= 0.1 \
                and self.observation[OBS_INDEX.ANGLE_DIFF_BEGIN] <= math.radians(15)\
                and self.observation[OBS_INDEX.ANGLE_DIFF_END] <= math.radians(15):  
            terminated = True
            self.reward += 10
        return terminated

    def _check_truncated_condition(self):
        truncated = False

        if self.observation[OBS_INDEX.DISTANCE_BEGIN] < 1 and abs(self.observation[OBS_INDEX.VELOCITY_BEGIN]) > 0.3\
                or self.observation[OBS_INDEX.DISTANCE_BEGIN] >= 1 and abs(self.observation[OBS_INDEX.VELOCITY_BEGIN]) > 0.6:
            self.time_velocity_not_low = self.data.time

        if self.data.time - self.time_velocity_not_low >= 3:
            truncated = True
        elif self.data.time > self.time_limit:
            truncated = True
        elif any(contact_val > 0 for contact_val in self.extra_observation[EXTRA_OBS_INDEX.CONTACT_BEGIN:EXTRA_OBS_INDEX.CONTACT_END+1]):
            truncated = True
            
        if truncated:
            self.reward -= 1
        return truncated

    def _get_obs(self):
        observation = np.zeros(OBS_INDEX.OBS_SIZE, dtype=np.float32)
        extra_observation = np.zeros(EXTRA_OBS_INDEX.OBS_SIZE, dtype=np.float32)
        
        car_global_pos = self.data.sensor(
            f'{CAR_NAME}/pos_global_sensor').data

        car_parking_rel_pos = self.data.sensor(
            f"{CAR_NAME}_to_{PARKING_NAME}_pos").data

        carSpeed = self.data.sensor(f'{CAR_NAME}/speed_sensor').data[0]

        range_sensors = []
        for i in range(CAR_N_RANGE_SENSORS):
            range_sensors.append(self.data.sensor(
                f'{CAR_NAME}/range_sensor_{i}').data[0])
        for i in range(TRAILER_N_RANGE_SENSORS):
            range_sensors.append(self.data.sensor(
                f'{CAR_NAME}/{TRAILER_NAME}/range_sensor_{i}').data[0])

        car_dist_to_parking = np.linalg.norm(car_parking_rel_pos[:2])

        contact_data = [
            self.data.sensor(f"{CAR_NAME}/touch_sensor").data[0],
            self.data.sensor(f"{CAR_NAME}/{TRAILER_NAME}/touch_sensor").data[0]
        ]
        
        
        # GET ANGLES
        carQuat = self.data.body(f"{CAR_NAME}/").xquat
        car_euler = quat_to_euler(carQuat)
        
        trailerQuat = self.data.body(f"{CAR_NAME}/{TRAILER_NAME}/").xquat
        trailer_euler = quat_to_euler(trailerQuat)
        
        targetQuat = self.data.body(f"{PARKING_NAME}/").xquat
        target_euler = quat_to_euler(targetQuat)

        trailer_joint_angle = self.data.joint(f"{CAR_NAME}/{TRAILER_NAME}/").qpos
        
        # ANGLE DIFFS
        carAngleDiff = car_euler[2] - target_euler[2]
        normalizedCarAngleDiff = normalize_angle_diff(carAngleDiff)
        
        trailerAngleDiff = trailer_euler[2] - target_euler[2]
        normalizedTrailerAngleDiff = normalize_angle_diff(trailerAngleDiff)
        
        # GATHER OBS
        observation[OBS_INDEX.VELOCITY_BEGIN:OBS_INDEX.VELOCITY_END+1] = carSpeed
        observation[OBS_INDEX.DISTANCE_BEGIN:OBS_INDEX.DISTANCE_END+1] = car_dist_to_parking
        observation[OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1] = normalizedCarAngleDiff, normalizedTrailerAngleDiff
        observation[OBS_INDEX.REL_POS_BEGIN:OBS_INDEX.REL_POS_END+1] = car_parking_rel_pos[:2]
        observation[OBS_INDEX.YAW_BEGIN:OBS_INDEX.YAW_END+1] = car_euler[2], trailer_euler[2]
        observation[OBS_INDEX.RANGE_BEGIN:OBS_INDEX.RANGE_END+1] = range_sensors
        
        # GATHER EXTRA OBS
        extra_observation[EXTRA_OBS_INDEX.GLOBAL_POS_BEGIN:EXTRA_OBS_INDEX.GLOBAL_POS_END+1] = car_global_pos[:2]
        extra_observation[EXTRA_OBS_INDEX.CONTACT_BEGIN:EXTRA_OBS_INDEX.CONTACT_END+1] = contact_data
        
        return observation, extra_observation

    def random_spawn(self):
        spawn_index = np.random.randint(0, len(self.spawn_points))
        self.data.joint(f"{CAR_NAME}/").qpos = self.spawn_points[spawn_index]
    
    def add_spawn_noise(self):
     
        qpos = self.data.joint(f"{CAR_NAME}/").qpos
        pos = qpos[:3]
        quat = qpos[3:]
        
        eul = quat_to_euler(quat)
        eul[2] = eul[2] + np.random.normal(0, self.spawn_angle_degrees_noise/3, size=1)
        quat = euler_to_quat(*eul)
        
        
        pos[0:2] = pos[0:2] + np.random.normal(0, self.spawn_dist_noise/3, size=2)
        
        maplength = MAP_LENGTH
        carSizeOffset = max(CAR_DIMS)
        pos[0] = np.clip(pos[0],  -maplength[0]/2 + carSizeOffset, maplength[0]/2 - carSizeOffset)
        pos[1] = np.clip(pos[1], -maplength[1]/2 + carSizeOffset , maplength[1]/2 - carSizeOffset)
        
        self.data.joint(f"{CAR_NAME}/").qpos = [*pos, *quat]
        
    def reset(self, seed = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        if seed is not None:
            np.random.seed(seed)
        
        self._reset_simulation()
        if self.enable_random_spawn:
            self.random_spawn()
        if self.enable_spawn_noise:
            self.add_spawn_noise()
        
        # RESET VARAIBLES
        self.time_velocity_not_low = 0
        self.norm_angle_cost = 0
        self.velocity_cost = 0
        self.norm_velocity_cost = 0
        self.episode_cumulative_reward = 0
        self.episode_mean_reward = 0
        self.init_car_distance = np.linalg.norm(self.data.joint(f"{CAR_NAME}/").qpos[:2] - self.data.site(f"{PARKING_NAME}/site_center_{CAR_NAME}").xpos[:2])
        self.init_trailer_distance = np.linalg.norm(self.data.joint(f"{CAR_NAME}/{TRAILER_NAME}/").qpos[:2] - self.data.site(f"{PARKING_NAME}/site_center_{TRAILER_NAME}").xpos[:2])
        self.episode_number += 1

        obs, extra_obs = self._get_obs()
        return obs, {"extra_obs":extra_obs}


if __name__ == "__main__":
    env = CarParkingEnv(render_mode="human")
    from stable_baselines3.common.env_checker import check_env
    check_env(env, skip_render_check=True)
