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
import matplotlib.pyplot as plt


SELF_DIR = Path(__file__).parent.resolve()
XML_FOLDER = SELF_DIR.joinpath("xmls")
MODEL_PATH = os.path.join(str(XML_FOLDER), "generated.xml")


WHEEL_ANGLE_RANGE = (-45, 45)
CAR_NAME = "car1"


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

        self._initialize_simulation()
        self._set_default_action_space()

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
        reward = 0
        terminated = False
        info = ""

        return observation, reward, terminated, False, info

    def _get_obs(self):
        carPosition = self.data.body(CAR_NAME).xpos
        observation = np.array([carPosition])
        return observation

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self._reset_simulation()

        observation = self._get_obs()
        return observation


if __name__ == "__main__":
    env = CarParkingEnv(render_mode="human")
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        # print(action)
        env.render()
