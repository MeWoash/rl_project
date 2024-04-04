from pathlib import Path
import numpy as np

import gymnasium
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box

from typing import Dict, Tuple, Union

import sys
import os
import math

from enum import IntEnum

SELF_DIR = Path(__file__).parent.resolve()
XML_FOLDER = SELF_DIR.joinpath("../xmls")
MODEL_PATH = os.path.join(str(XML_FOLDER), "generated.xml")

MAP_SIZE = (20, 20, 20, 5)


WHEEL_ANGLE_RANGE = (-45, 45)
CAR_NAME = "mainCar"


class ObsIndex(IntEnum):
    DISTANCE = 0
    ANGLE_DIFF = 1


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
        ],
        "render_fps": 125
    }

    def __init__(self,
                 xml_file: str = MODEL_PATH,
                 render_mode: str = "human",
                 frame_skip: int = 4,
                 **kwargs):

        self.fullpath = xml_file
        self.render_mode = render_mode
        self.frame_skip = frame_skip

        # TODO CAMERA SETTINGS
        self.camera_name = None
        self.camera_id = None
        self.init_obs = None

        self._initialize_simulation()
        self._set_default_action_space()
        self._set_default_observation_space()

    def _set_default_action_space(self):

        engineCtrlRange = np.array(
            [-1,
             1])
        wheelAngleCtrlRange = np.array(
            [math.radians(WHEEL_ANGLE_RANGE[0]),
             math.radians(WHEEL_ANGLE_RANGE[1])])

        low = np.array(
            [engineCtrlRange[0], wheelAngleCtrlRange[0]])
        high = np.array(
            [engineCtrlRange[1], wheelAngleCtrlRange[1]])

        self.action_space = Box(low=low, high=high, dtype=np.float64)
        return self.action_space

    def _set_default_observation_space(self):

        maxDistanceToTarget = math.sqrt(MAP_SIZE[2]**2 +
                                        math.sqrt(MAP_SIZE[0]**2 + MAP_SIZE[1]**2))

        distRange = np.array([0, maxDistanceToTarget])
        angleDiff = np.array([-np.pi, np.pi])

        low = np.array([distRange[0], angleDiff[0]])
        high = np.array([distRange[1], angleDiff[1]])

        self.observation_space = Box(low=low, high=high, dtype=np.float64)
        return self.observation_space

    def _initialize_simulation(self):
        # source MujocoEnv
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)
        self.mujoco_renderer = MujocoRenderer(self.model, self.data)

    def _reset_simulation(self):
        # source MujocoEnv
        mujoco.mj_resetData(self.model, self.data)

    def render(self):
        # source MujocoEnv
        return self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )

    def close(self):
        """Close all processes like rendering contexts"""
        # source MujocoEnv
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def _apply_forces(self, action=None):
        enginePowerCtrl = action[0]
        wheelsAngleCtrl = action[1]

        self.data.actuator(f"{CAR_NAME}_engine_power").ctrl = enginePowerCtrl
        self.data.actuator(f"{CAR_NAME}_wheel1_angle").ctrl = wheelsAngleCtrl
        self.data.actuator(f"{CAR_NAME}_wheel2_angle").ctrl = wheelsAngleCtrl

    def _do_simulation(self, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """

        mujoco.mj_step(self.model, self.data, nstep=n_frames)
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def step(self, action):

        self._apply_forces(action)
        self._do_simulation(self.frame_skip)

        observation = self._get_obs()

        if self.init_obs is None:
            self.init_obs = observation

        reward = self._calculate_reward(observation)

        terminated = self._check_terminate_condition(observation)
        truncated = self._check_truncated_condition(observation)

        self.prev_obs = observation

        info = {}
        renderRetVal = self.render()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, observation):
        reward = 10/observation[0]
        return reward

    def _check_terminate_condition(self, observation):
        terminated = False
        if observation[0] <= 0.1 and \
                math.radians(-5) <= observation[1] <= math.radians(5):
            terminated = True

        return terminated

    def _check_truncated_condition(self, observation):
        truncated = False

        if observation[0] > self.init_obs[0]*1.1 or self.data.time > 10:
            truncated = True
        return truncated

    def _get_obs(self):
        # carPositionGlobal = self.data.sensor('mainCar_posGlobal_sensor').data
        carPositionParking = self.data.sensor('mainCar_posTarget_sensor').data

        distToTarget = np.linalg.norm(carPositionParking[:1])

        carQuat = self.data.body('mainCar').xquat
        roll_x, pitch_y, yaw_z = quat_to_euler(carQuat)

        targetQuat = self.data.body('target_space').xquat
        targetroll_x, targetpitch_y, targetyaw_z = quat_to_euler(targetQuat)

        angleDiff = yaw_z - targetyaw_z

        observation = np.array(
            (distToTarget, angleDiff))
        return observation

    def reset(self, **kwargs):
        self._reset_simulation()

        observation = self._get_obs()
        return observation, {}


if __name__ == "__main__":
    env = CarParkingEnv(render_mode="human")

    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        # print(action)
        env.render()
