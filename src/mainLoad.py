import gymnasium
import CustomEnvs  # Registering envs


from pathlib import Path
import matplotlib.pyplot as plt

from ModelTools.ModelWrapper import ModelWrapper
from ModelTools.ModelPresets import *
from CustomEnvs.CarParking import CarParkingEnv, ObsIndex
import numpy as np

MODEL_DIR = Path(__file__).parent.joinpath("../out/models").resolve()


def describe_obs(obs):
    d = f"""
speed:      {obs[ObsIndex.VELOCITY_BEGIN:ObsIndex.VELOCITY_END+1]}    
dist:       {obs[ObsIndex.DISTANCE_BEGIN:ObsIndex.DISTANCE_END+1]}
adiff:      {obs[ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1]}
contact:    {obs[ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1]}
range:      {obs[ObsIndex.RANGE_BEGIN:ObsIndex.RANGE_END+1]}
pos:        {obs[ObsIndex.POS_BEGIN:ObsIndex.POS_END+1]}
eul:        {obs[ObsIndex.EUL_BEGIN:ObsIndex.EUL_END+1]}"""

    return d


def test_env(preset):
    modelWrapper = ModelWrapper()
    modelWrapper.create_gym_env("human")
    modelWrapper.create_model(**preset['createModelArgs'])
    modelWrapper.load_model()

    truncated = False
    terminated = False
    observation, info = modelWrapper.env.reset()

    while True:
        action, states = modelWrapper.model.predict(
            observation, deterministic=True)
        observation, reward, terminated, truncated, info = modelWrapper.env.step(
            action)
        if truncated or terminated:
            modelWrapper.env.reset()


def test_vec_env(preset):
    modelWrapper = ModelWrapper()
    modelWrapper.create_vec_env(**preset['createEnvArgs']['envArgs'])
    modelWrapper.create_model(**preset['createModelArgs'])
    modelWrapper.load_model()

    observation = modelWrapper.env.reset()
    while True:
        action, states = modelWrapper.model.predict(
            observation, deterministic=True)
        observation, reward, done, info = modelWrapper.env.step(action)


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    # test_vec_env(A2C_PRESET)
    test_env(A2C_PRESET)
