import gymnasium
import CustomEnvs  # Registering envs


from pathlib import Path
import matplotlib.pyplot as plt

from ModelTools.ModelWrapper import ModelWrapper
from ModelTools.ModelPresets import *

MODEL_DIR = Path(__file__).parent.joinpath("../out/models").resolve()


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
            observation)
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
        action, states = modelWrapper.model.predict(observation)
        observation, reward, done, info = modelWrapper.env.step(action)


if __name__ == "__main__":
    test_vec_env(A2C_PRESET)
    # test_env(SAC_PRESET)
