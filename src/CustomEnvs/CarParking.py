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


SELF_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SELF_DIR.parent))

import MJCFGenerator
from Rendering.RendererClass import Renderer
from Rendering.Utils import TextOverlay

MODEL_NAME = "out.xml"
MJCF_OUT_DIR = MJCFGenerator.MJCF_OUT_DIR
MODEL_PATH = os.path.join(str(MJCF_OUT_DIR), MODEL_NAME)

CAR_NAME = MJCFGenerator.GeneratorClass._carName
TRAILER_NAME = MJCFGenerator.GeneratorClass._trailerName
PARKING_NAME = MJCFGenerator.GeneratorClass._spotName
CAR_SPAWN_HEIGHT = MJCFGenerator.GeneratorClass._carSpawnHeight


# TODO PARAMETERS SHOULD BE SCRAPED FROM MJDATA
MAP_SIZE = MJCFGenerator.GeneratorClass._map_length
MAX_X_Y_DIST = math.sqrt(MAP_SIZE[0]**2 + MAP_SIZE[1]**2)
MAX_X_Y_Z_DIST = math.sqrt(MAP_SIZE[2]**2 +
                           math.sqrt(MAP_SIZE[0]**2 + MAP_SIZE[1]**2))
MAX_SENSOR_VAL = MJCFGenerator.Car._maxSensorVal

WHEEL_ANGLE_RANGE = [math.radians(MJCFGenerator.Wheel._wheel_angle_limit[0]), math.radians(MJCFGenerator.Wheel._wheel_angle_limit[1])]

CAR_N_RANGE_SENSORS = 5
TRAILER_N_RANGE_SENSORS = 5
N_RANGE_SENSORS = CAR_N_RANGE_SENSORS + TRAILER_N_RANGE_SENSORS


# autopep8: on

# INCLUSIVE INDEXES
class ObsIndex(IntEnum):
    # 1 val
    VELOCITY_BEGIN = 0
    VELOCITY_END = 0

    # 1 val
    DISTANCE_BEGIN = 1
    DISTANCE_END = 1

    # 2 val
    ANGLE_DIFF_BEGIN = 2
    ANGLE_DIFF_END = 3

    # 2 val
    CONTACT_BEGIN = 4
    CONTACT_END = 5

    # 2 val 
    POS_BEGIN = 6
    POS_END = 7

    # 2 val
    YAW_BEGIN = 8
    YAW_END = 9
    
    # 8 val
    RANGE_BEGIN = 10
    RANGE_END = 19
    
    OBS_SIZE = 20



def normalize_data(x, dst_a, dst_b, min_x=-1, max_x=1):
    normalized = dst_a + ((x - min_x)*(dst_b-dst_a))/(max_x-min_x)
    return normalized

def normalize_angle_diff(angle_diff):
    return abs(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))

def quat_to_euler(quat):
    w, x, y, z = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return [roll_x, pitch_y, yaw_z]

def euler_to_quat(roll, pitch, yaw):

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = np.array([w, x, y, z])

    norm = np.linalg.norm(quat)
    quat = quat / norm

    return quat.flatten()


class CarParkingEnv(gymnasium.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "none"
        ],
    }

    def __init__(self,
                 xml_file: str = MODEL_PATH,
                 render_mode: str = "none",
                 simulation_frame_skip: int = 4,
                 time_limit = 30,
                 capture_frames = False,
                 capture_fps = 24,
                 frame_size = (1920, 1080), # Width, Height
                 enable_random_spawn = True,
                 enable_spawn_noise = True,
                 **kwargs):

        
        self.enable_random_spawn= enable_random_spawn
        self.enable_spawn_noise=enable_spawn_noise
        
        self.time_limit = time_limit
        self.fullpath = xml_file
        self.simulation_frame_skip = simulation_frame_skip
        
        # RENDER VARIABLES
        self.render_mode = render_mode
        self.capture_frames = capture_frames
        self.capture_fps = capture_fps
        self.frame_size = frame_size
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

        # TODO CAMERA SETTINGS
        self.camera_name = None
        self.camera_id = 0
        self.episode_number = 0

        self._initialize_simulation()
        self._set_default_action_space()
        self._set_default_observation_space()
        self._calculate_spawn_points()
        
        # TRUNCATE VARIABLES
        self.time_velocity_not_low = 0

        # REWARD VARIABLES
        self.reward_range = (-1, 1)
        self.episode_mujoco_max_step = self.time_limit / self.model.opt.timestep
        self.episode_env_max_step = self.episode_mujoco_max_step/self.simulation_frame_skip
        self.angle_cost = 0
        self.velocity_cost = 0
        
        self.velocity_max_cost = max([abs(self.action_space.low[0]), abs(self.action_space.high[0])]) * self.episode_env_max_step
        self.angle_max_cost = max([abs(self.action_space.low[1]), abs(self.action_space.high[1])]) * self.episode_env_max_step
        
        
        self.dist_punish_weight = 0.75 
        self.angle_diff_punish_weight = 0.25
        self.velocity_cost_punish_weight = 0
        self.angle_cost_punish_weight = 0
        self.max_step_reward = 1
        
        self.episode_cumulative_reward = 0
        
        
        assert\
              self.dist_punish_weight \
            + self.velocity_cost_punish_weight\
            + self.angle_cost_punish_weight \
            + self.angle_diff_punish_weight\
                == self.max_step_reward, f"Weights have to sum to {self.max_step_reward}"

    def _set_default_action_space(self):

        engineCtrlRange = np.array(
            [-1,
             1])
        wheelAngleCtrlRange = np.array(
            [-1,
             1])

        low = np.array(
            [engineCtrlRange[0], wheelAngleCtrlRange[0]])
        high = np.array(
            [engineCtrlRange[1], wheelAngleCtrlRange[1]])

        self.action_space = Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_default_observation_space(self):

        carSpeedRange = np.array([-10, 10]).reshape(2, 1)
        distRange = np.array([0, MAX_X_Y_DIST]).reshape(2, 1)
        angleDiff = np.tile(np.array([-np.pi, np.pi]).reshape(2, 1), (1, 2))
        contactRange = np.tile(np.array([0, MAX_SENSOR_VAL]).reshape(2, 1), (1, 2))
        range_sensorsRange = np.tile(
            np.array([0, MAX_SENSOR_VAL]).reshape(2, 1), (1, N_RANGE_SENSORS))
        carPositionGlobalRange = np.array(
            [[-MAP_SIZE[0]/2, -MAP_SIZE[1]/2],
             [MAP_SIZE[0]/2, MAP_SIZE[1]/2]])
        car_eulerRange = np.tile(np.array([0, np.pi]).reshape(2, 1), (1, 2))

        # KEEP ORDER AS IN OBSINDEX
        boundMatrix = np.hstack(
            [carSpeedRange,
             distRange,
             angleDiff,
             contactRange,
             carPositionGlobalRange,
             car_eulerRange,
             range_sensorsRange])

        self.observation_space = Box(
            low=boundMatrix[0, :], high=boundMatrix[1, :], dtype=np.float32)
        return self.observation_space

    def _initialize_simulation(self):
        # source MujocoEnv
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)

        self.mujoco_renderer: Renderer = Renderer(self.model,
                                                self.data,
                                                self.simulation_frame_skip,
                                                self.capture_frames,
                                                self.capture_fps,
                                                self.frame_size)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) #  WTHOUT THIS LINE FIRST STEP CONSIST 0s

    def _calculate_spawn_points(self):
        self.spawn_points = []
        for spawn_params in MJCFGenerator.SPAWN_POINTS:
            pos = spawn_params['pos']
            pos[2] = MJCFGenerator.GeneratorClass._carSpawnHeight
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
        overlay.add("speed",f"{self.observation[ObsIndex.VELOCITY_BEGIN:ObsIndex.VELOCITY_END+1]}", "top left")
        overlay.add("dist",f"{self.observation[ObsIndex.DISTANCE_BEGIN:ObsIndex.DISTANCE_END+1]}", "top left")
        overlay.add("adiff",f"{self.observation[ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1]}", "top left")
        overlay.add("contact",f"{self.observation[ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1]}", "top left")
        overlay.add("pos",f"{self.observation[ObsIndex.POS_BEGIN:ObsIndex.POS_END+1]}", "top left")
        overlay.add("eul",f"{self.observation[ObsIndex.YAW_BEGIN:ObsIndex.YAW_END+1]}", "top left")
        overlay.add("range car",f"{self.observation[ObsIndex.RANGE_BEGIN:ObsIndex.RANGE_BEGIN+CAR_N_RANGE_SENSORS]}", "top left")
        overlay.add("range trl",f"{self.observation[ObsIndex.RANGE_BEGIN+CAR_N_RANGE_SENSORS:ObsIndex.RANGE_END+1]}", "top left")
        overlay.add("Model Action", "values", "top right")
        overlay.add("engine", f"{self.action[0]:.2f}", "top right")
        overlay.add("wheel", f"{self.action[1]:.2f}", "top right")
        
        overlay.add("Reward metrics", "values", "bottom right")
        overlay.add("reward", f"{self.reward:.2f}", "bottom right")
        overlay.add("cum_reward", f"{self.episode_cumulative_reward:.2f} ~ {self.episode_mean_reward:.2f}", "bottom right")
        
        overlay.add("dist", f"{self.observation[ObsIndex.DISTANCE_BEGIN]:.2f} ~ {self.norm_dist:.2f}", "bottom right")
        overlay.add("v_sum", f"{self.velocity_cost:.2f} ~ {self.norm_velocity_cost:.2f}", "bottom right")
        overlay.add("a_sum", f"{self.angle_cost:.2f} ~ {self.norm_angle_cost:.2f}", "bottom right")
        overlay.add("a_diff", f"{self.angle_diff:.2f} ~ {self.norm_angle_diff:.2f}", "bottom right")
        
    def close(self):
        """Close all processes like rendering contexts"""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()


    def _apply_forces(self, action=None):
        enginePowerCtrl = action[0]
        wheelsAngleCtrl = normalize_data(action[1], *WHEEL_ANGLE_RANGE)

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

        self.observation = self._get_obs()

        self.reward = self._calculate_reward()

        self.terminated = self._check_terminate_condition()
        self.truncated = self._check_truncated_condition()

        self.episode_cumulative_reward += self.reward
        self.episode_mean_reward = self.episode_cumulative_reward/(self.episode_env_step+1) #  normalize_data(self.cumulative_reward, 0, 1, 0, ()*self.max_step_reward)
        
        renderRetVal = self.render()
        return self.observation, self.reward, self.terminated, self.truncated, self._get_info()

    def _get_info(self):
        info = {
            "episode_mujoco_time": self.episode_mujoco_time,
            "episode_mujoco_step": self.episode_mujoco_step,
            "episode_env_step": self.episode_env_step,
            "episode_number":self.episode_number,
            "episode_cumulative_reward":self.episode_cumulative_reward,
            "episode_mean_reward": self.episode_mean_reward
                     }
        return info
    
    def _calculate_reward(self):
        
        self.velocity_cost += abs(self.action[0])
        self.angle_cost += abs(self.action[1])
        
        self.angle_diff = np.sum(self.observation[ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1])
        DIST_SCALE = 2
        exp_scale = np.exp(-self.observation[ObsIndex.DISTANCE_BEGIN]/DIST_SCALE)
        
        self.norm_angle_diff = normalize_data(self.angle_diff,
                                              0, self.angle_diff_punish_weight*exp_scale,
                                              0, len(self.observation[ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1])*math.pi)
        self.norm_angle_diff = np.clip(self.norm_angle_diff, 0, self.angle_diff_punish_weight)
        
        self.norm_dist = normalize_data(np.clip(self.observation[ObsIndex.DISTANCE_BEGIN], 0 ,self.init_distance), 0, self.dist_punish_weight, 0, self.init_distance)
        
        self.norm_velocity_cost = normalize_data(self.velocity_cost, 0, self.velocity_cost_punish_weight, 0, self.velocity_max_cost)
        self.norm_angle_cost = normalize_data(self.angle_cost, 0, self.angle_cost_punish_weight, 0, self.angle_max_cost)
        
        punish = self.norm_dist + self.norm_velocity_cost + self.norm_angle_cost + self.norm_angle_diff

        reward = self.max_step_reward - punish
        return reward

    def _check_terminate_condition(self):
        terminated = False
        if self.observation[ObsIndex.DISTANCE_BEGIN] <= 0.25 \
                and abs(self.observation[ObsIndex.VELOCITY_BEGIN]) <= 0.1 \
                and self.observation[ObsIndex.ANGLE_DIFF_BEGIN] <= math.radians(10)\
                and self.observation[ObsIndex.ANGLE_DIFF_END] <= math.radians(10):  
            terminated = True
        return terminated

    def _check_truncated_condition(self):
        truncated = False

        if self.observation[ObsIndex.DISTANCE_BEGIN] < 1 and abs(self.observation[ObsIndex.VELOCITY_BEGIN]) > 0.05\
                or self.observation[ObsIndex.DISTANCE_BEGIN] >= 1 and abs(self.observation[ObsIndex.VELOCITY_BEGIN]) > 0.3:
            self.time_velocity_not_low = self.data.time

        if self.data.time - self.time_velocity_not_low >= 3:
            truncated = True
        elif self.data.time > self.time_limit:
            truncated = True
        elif any(contact_val > 0 for contact_val in self.observation[ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1]):
            truncated = True
            
        if truncated:
            self.reward -= 1
        return truncated

    def _get_obs(self):
        carPositionGlobal = self.data.sensor(
            f'{CAR_NAME}/pos_global_sensor').data

        carPositionParking = self.data.sensor(
            f"{CAR_NAME}_to_{PARKING_NAME}_pos").data

        carSpeed = self.data.sensor(f'{CAR_NAME}/speed_sensor').data[0]

        range_sensors = []
        for i in range(CAR_N_RANGE_SENSORS):
            range_sensors.append(self.data.sensor(
                f'{CAR_NAME}/range_sensor_{i}').data[0])
        for i in range(TRAILER_N_RANGE_SENSORS):
            range_sensors.append(self.data.sensor(
                f'{CAR_NAME}/{TRAILER_NAME}/range_sensor_{i}').data[0])

        distToTarget = np.linalg.norm(carPositionParking[:2])

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

        # ANGLE DIFFS
        carAngleDiff = car_euler[2] - target_euler[2]
        normalizedCarAngleDiff = normalize_angle_diff(carAngleDiff)
        
        trailerAngleDiff = trailer_euler[2] - target_euler[2]
        normalizedTrailerAngleDiff = normalize_angle_diff(trailerAngleDiff)
        
        # GATHER OBS
        observation = np.zeros(ObsIndex.OBS_SIZE, dtype=np.float32)
        
        observation[ObsIndex.VELOCITY_BEGIN:ObsIndex.VELOCITY_END+1] = carSpeed
        observation[ObsIndex.DISTANCE_BEGIN:ObsIndex.DISTANCE_END+1] = distToTarget
        observation[ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1] = normalizedCarAngleDiff, normalizedTrailerAngleDiff
        observation[ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1] = contact_data
        observation[ObsIndex.POS_BEGIN:ObsIndex.POS_END+1] = carPositionGlobal[:2]
        observation[ObsIndex.YAW_BEGIN:ObsIndex.YAW_END+1] = car_euler[2], trailer_euler[2]
        observation[ObsIndex.RANGE_BEGIN:ObsIndex.RANGE_END+1] = range_sensors
        
        return observation

    def random_spawn(self):
        spawn_index = np.random.randint(0, len(self.spawn_points))
        self.data.joint(f"{CAR_NAME}/").qpos = self.spawn_points[spawn_index]
    
    def add_spawn_noise(self):
        # 3 sigmas rule
        angle_diff = np.radians(10)
        pos_diff = 1
        
        qpos = self.data.joint(f"{CAR_NAME}/").qpos
        pos = qpos[:3]
        quat = qpos[3:]
        
        eul = quat_to_euler(quat)
        eul[2] = eul[2] + np.random.normal(0, angle_diff/3, size=1)
        quat = euler_to_quat(*eul)
        
        
        pos[0:2] = pos[0:2] + np.random.normal(0, pos_diff/3, size=2)
        
        maplength = MJCFGenerator.GeneratorClass._map_length
        carSizeOffset = max(MJCFGenerator.GeneratorClass._car_dims)
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
        self.angle_cost = 0
        self.norm_angle_cost = 0
        self.velocity_cost = 0
        self.norm_velocity_cost = 0
        self.episode_cumulative_reward = 0
        self.episode_mean_reward = 0
        self.init_distance = np.linalg.norm(self.data.joint(f"{CAR_NAME}/").qpos[:2] - self.data.site(f"{MJCFGenerator.GeneratorClass._spotName}/site_center").xpos[:2])
        self.episode_number += 1

        obs = self._get_obs()
        return obs, {}


if __name__ == "__main__":
    env = CarParkingEnv(render_mode="human")
    from stable_baselines3.common.env_checker import check_env
    check_env(env, skip_render_check=True)
