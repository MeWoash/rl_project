import gymnasium
import CustomEnvs  # Registering envs
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv

from CustomEnvs.CarParking import CarParkingEnv

from pathlib import Path
import matplotlib.pyplot as plt

MODEL_DIR = Path(__file__).parent.joinpath("../out/models").resolve()

if __name__ == "__main__":
    N_ENVS = 4
    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                       n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, env_kwargs={"render_mode": "human"})

    # env = CarParkingEnv(render_mode="human")
    model = A2C("MlpPolicy", env, device="cuda")
    model.load(MODEL_DIR/"testModel")

    truncated = False
    terminated = False
    observation = env.reset()

    while True:
        action, states = model.predict(observation)
        # print(action)
        observation, reward, done, info = env.step(action)
        # print(observation)
        if truncated or terminated:
            env.reset()
