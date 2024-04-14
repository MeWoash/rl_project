# autopep8: off
from pathlib import Path
from tabnanny import check
from traceback import print_tb
import numpy as np
import gymnasium
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from typing import Dict, Text, Tuple, Union
import sys
import os
import math
from enum import IntEnum

SELF_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SELF_DIR.parent))

import MJCFGenerator
from Rendering.RendererClass import Renderer
from Rendering.Utils import TextOverlay

MODEL_NAME = "out.xml"
MJCF_OUT_DIR = MJCFGenerator.MJCF_OUT_DIR
MODEL_PATH = os.path.join(str(MJCF_OUT_DIR), MODEL_NAME)

CAR_NAME = MJCFGenerator.Generator._carName # NOT GUARANTEED TO MATCH
PARKING_NAME = MJCFGenerator.Generator._spotName # NOT GUARANTEED TO MATCH


# TODO PARAMETERS SHOULD BE SCRAPED FROM MJDATA
MAP_SIZE = [20, 20, 20, 5]
MAX_X_Y_DIST = math.sqrt(MAP_SIZE[0]**2 + MAP_SIZE[1]**2)
MAX_X_Y_Z_DIST = math.sqrt(MAP_SIZE[2]**2 +
                           math.sqrt(MAP_SIZE[0]**2 + MAP_SIZE[1]**2))
MAX_SENSOR_VAL = MJCFGenerator.Car._maxSensorVal

WHEEL_ANGLE_RANGE = [math.radians(MJCFGenerator.Wheel._wheel_angle_limit[0]), math.radians(MJCFGenerator.Wheel._wheel_angle_limit[1])]
N_RANGE_SENSORS = 8


# autopep8: on


class ObsIndex(IntEnum):
    VELOCITY_BEGIN = 0
    VELOCITY_END = 0

    DISTANCE_BEGIN = 1
    DISTANCE_END = 1

    ANGLE_DIFF_BEGIN = 2
    ANGLE_DIFF_END = 2

    CONTACT_BEGIN = 3
    CONTACT_END = 3

    RANGE_BEGIN = 4
    RANGE_END = 11

    POS_BEGIN = 12
    POS_END = 14

    EUL_BEGIN = 15
    EUL_END = 17


def normalize_data(x, dst_a, dst_b, min_x=-1, max_x=1):
    normalized = dst_a + ((x - min_x)*(dst_b-dst_a))/(max_x-min_x)
    return normalized


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

    return roll_x, pitch_y, yaw_z


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
                 render_mode: str = "rgb_array",
                 simulation_frame_skip: int = 4,
                 capture_frames = False,
                 capture_fps = 24,
                 frame_size = (480, 480), # Width, Height
                 **kwargs):

        self.fullpath = xml_file
        self.render_mode = render_mode
        self.simulation_frame_skip = simulation_frame_skip
        
        self.capture_frames = capture_frames
        self.capture_fps = capture_fps
        self.frame_size = frame_size
        
        
        self.time_velocity_not_low = None

        # TODO CAMERA SETTINGS
        self.camera_name = None
        self.camera_id = 0
        self.episode = 0

        self._initialize_simulation()
        self._set_default_action_space()
        self._set_default_observation_space()

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
        angleDiff = np.array([-np.pi, np.pi]).reshape(2, 1)
        contactRange = np.array([0, MAX_SENSOR_VAL]).reshape(2, 1)
        range_sensorsRange = np.tile(
            np.array([0, MAX_SENSOR_VAL]).reshape(2, 1), (1, N_RANGE_SENSORS))
        carPositionGlobalRange = np.array(
            [[-MAP_SIZE[0]/2, -MAP_SIZE[1]/2, 0],
             [MAP_SIZE[0]/2, MAP_SIZE[1]/2, MAP_SIZE[2]]])
        car_eulerRange = np.tile(
            np.array([-np.pi, np.pi]).reshape(2, 1), (1, 3))

        boundMatrix = np.hstack(
            [carSpeedRange, distRange, angleDiff, contactRange, range_sensorsRange, carPositionGlobalRange, car_eulerRange])

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

    def render(self):
        o = TextOverlay()
        if self.render_mode == "human":
            self._prep_overlay(o)

        if self.render_mode != "none":
            return self.mujoco_renderer.render(self.render_mode, self.camera_id, o)


    def _prep_overlay(self, overlay: TextOverlay):
        overlay.add("Step",f"{round(self.data.time / self.model.opt.timestep)}", "bottom left")
        overlay.add("episode", f"{self.episode}", "bottom left")
        overlay.add("time", "%.2f"%self.data.time, "bottom left")
        overlay.add("steps/1s", f"{1 / (self.model.opt.timestep*self.simulation_frame_skip)}", "bottom left")
        overlay.add("Env stats", "values", "top left")
        overlay.add("reward","%.2f"%self.reward, "top left")
        overlay.add("speed",f"{self.observation[ObsIndex.VELOCITY_BEGIN:ObsIndex.VELOCITY_END+1]}", "top left")
        overlay.add("dist",f"{self.observation[ObsIndex.DISTANCE_BEGIN:ObsIndex.DISTANCE_END+1]}", "top left")
        overlay.add("adiff",f"{self.observation[ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1]}", "top left")
        overlay.add("contact",f"{self.observation[ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1]}", "top left")
        overlay.add("range",f"{self.observation[ObsIndex.RANGE_BEGIN:ObsIndex.RANGE_END+1]}", "top left")
        overlay.add("pos",f"{self.observation[ObsIndex.POS_BEGIN:ObsIndex.POS_END+1]}", "top left")
        overlay.add("eul",f"{self.observation[ObsIndex.EUL_BEGIN:ObsIndex.EUL_END+1]}", "top left")
        overlay.add("Model Action", "values", "top right")
        overlay.add("engine", "%.2f" % self.action[0], "top right")
        overlay.add("wheel", "%.2f" % self.action[1], "top right")


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

    def step(self, action):
        self.action = action
        self._apply_forces(action)
        self._do_simulation(self.simulation_frame_skip)

        self.observation = self._get_obs()

        self.reward = self._calculate_reward()

        self.terminated = self._check_terminate_condition()
        self.truncated = self._check_truncated_condition()

        self.info = {
            "episode_time":self.data.time,
            "episode_number":self.episode,
                     }
        renderRetVal = self.render()

        # print(self.reward)
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def _calculate_reward(self):

        normdist = normalize_data(
            self.observation[ObsIndex.DISTANCE_BEGIN], 0, 1, 0, MAX_X_Y_DIST)
        total_dist_reward = 1 - normdist

        time_punish = normalize_data(self.data.time, 0, 0.5, 0, 30)

        reward = total_dist_reward * (1 - time_punish)
        return reward

    def _check_terminate_condition(self):
        terminated = False
        if self.observation[ObsIndex.DISTANCE_BEGIN] <= 0.25 \
                and abs(self.observation[ObsIndex.VELOCITY_BEGIN]) <= 0.1 \
                and math.radians(-5) <= self.observation[ObsIndex.ANGLE_DIFF_BEGIN] <= math.radians(10):
            terminated = True
        return terminated

    def _check_truncated_condition(self):
        truncated = False

        if self.observation[ObsIndex.DISTANCE_BEGIN] < 1 and abs(self.observation[ObsIndex.VELOCITY_BEGIN]) > 0.05\
                or self.observation[ObsIndex.DISTANCE_BEGIN] >= 1 and abs(self.observation[ObsIndex.VELOCITY_BEGIN]) > 0.3:
            self.time_velocity_not_low = self.data.time

        if self.time_velocity_not_low is not None:
            if self.data.time - self.time_velocity_not_low >= 3:
                truncated = True

        if self.data.time > 30:
            truncated = True
        elif self.observation[ObsIndex.CONTACT_BEGIN] > 0:
            truncated = True

        return truncated

    def _get_obs(self):
        carPositionGlobal = self.data.sensor(
            f'{CAR_NAME}/pos_global_sensor').data

        carPositionParking = self.data.sensor(
            f"{CAR_NAME}_to_{PARKING_NAME}_pos").data

        carSpeed = self.data.sensor(f'{CAR_NAME}/speed_sensor').data[0]

        range_sensors = []
        for i in range(N_RANGE_SENSORS):
            range_sensors.append(self.data.sensor(
                f'{CAR_NAME}/range_sensor_{i}').data[0])

        distToTarget = np.linalg.norm(carPositionParking[:2])

        contact_data = self.data.sensor(
            f"{CAR_NAME}/touch_sensor").data[0]

        carQuat = self.data.body(f"{CAR_NAME}/").xquat
        # roll_x, pitch_y, yaw_z
        car_euler = quat_to_euler(carQuat)

        targetQuat = self.data.body(f"{PARKING_NAME}/").xquat
        # targetroll_x, targetpitch_y, targetyaw_z
        target_euler = quat_to_euler(targetQuat)

        angleDiff = car_euler[2] - target_euler[2]

        observation = np.array(
            [carSpeed, distToTarget, angleDiff, contact_data, *range_sensors, *carPositionGlobal, *car_euler], dtype=np.float32)
        return observation

    def reset(self, **kwargs):
        self._reset_simulation()
        self.time_velocity_not_low = None
        self.episode += 1
        observation = self._get_obs()
        return observation, {}


if __name__ == "__main__":
    env = CarParkingEnv(render_mode="human")

    from stable_baselines3.common.env_checker import check_env
    check_env(env)
