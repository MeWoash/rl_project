import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from typing import Dict, Tuple, Union

import sys, os

XML_FOLDER = os.path.normpath(os.path.join(__file__,"..","..","xmls"))
MODEL_PATH = os.path.join(XML_FOLDER, "car-v1.xml")

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class CarParkingEnv(MujocoEnv, utils.EzPickle):
   metadata = {
   "render_modes": [
      "human",
      "rgb_array",
      "depth_array",
      ],
   "render_fps": 125
   }
        
   def __init__(
      self,
      xml_file: str = MODEL_PATH,
      frame_skip: int = 4,
      default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
      **kwargs,
   ):
      utils.EzPickle.__init__(
         self,
         xml_file,
         frame_skip,
         default_camera_config,
         camera_name="cam1",
         **kwargs,
      )

      MujocoEnv.__init__(
         self,
         xml_file,
         frame_skip,
         observation_space=None,
         default_camera_config=default_camera_config,
         camera_name="cam1",
         **kwargs,
      )

      obs_size = (
         self.data.qpos.size
         + self.data.qvel.size
      )
      self.observation_space = Box(
         low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
      )


   def _get_obs(self):
      position = self.data.qpos.flatten()
      velocity = np.clip(self.data.qvel.flatten(), -10, 10)

      observation = np.concatenate((position, velocity)).ravel()
      return observation

   def step(self, action):
      self.do_simulation(action, self.frame_skip)


      observation = self._get_obs()
      reward = 1
      reward_info = {
         "reward info": 1
      }

      terminated = False
      info = {
         "testinfo1": 0,
         **reward_info,
      }

      if self.render_mode == "human":
         self.render()
         
      return observation, reward, terminated, False, info

   def reset_model(self):
      qpos = self.init_qpos
      qvel = self.init_qvel
      self.set_state(qpos, qvel)

      observation = self._get_obs()
      return observation

   def _get_reset_info(self):
      return {
         "x_position": self.data.qpos[0],
         "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
      }


if __name__ == "__main__":
   env=CarParkingEnv()
   env.render_mode = "human"
   while True:
      action = env.action_space.sample()
      observation, reward, terminated, truncated, info = env.step(action)
      print(observation)
      env.render()