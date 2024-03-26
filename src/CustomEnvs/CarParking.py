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
XML_FOLDER = SELF_DIR.joinpath("../xmls")
MODEL_PATH = os.path.join(str(XML_FOLDER), "generated.xml")


WHEEL_ANGLE_RANGE = (-45, 45)
CAR_NAME = "mainCar"


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
        reward = 1/observation[0]

        terminated = False
        if observation[0] <= 0.1 and observation[4] <= math.radians(5):
            terminated = True

        info = ""

        return observation, reward, terminated, False, info

    def _get_obs(self):
        # carPositionGlobal = self.data.sensor('mainCar_posGlobal_sensor').data
        carPositionParking = self.data.sensor('mainCar_posTarget_sensor').data

        distToTarget = np.linalg.norm(carPositionParking[:1])

        carQuat = self.data.body('mainCar').xquat
        roll_x, pitch_y, yaw_z = quat_to_euler(carQuat)

        targetQuat = self.data.body('target_space').xquat
        targetroll_x, targetpitch_y, targetyaw_z = quat_to_euler(targetQuat)

        angleDiff = abs(targetyaw_z-yaw_z)

        observation = np.array(
            (distToTarget, roll_x, pitch_y, yaw_z, angleDiff))
        return observation

    def reset_model(self):
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
