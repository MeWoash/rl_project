import gymnasium
import CustomEnvs  # Registering envs
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv

from CustomEnvs.CarParking import CarParkingEnv

from pathlib import Path

MODEL_DIR = Path(__file__).parent.joinpath("../out/models").resolve()
LOGS_DIR = Path(__file__).parent.joinpath("../out/logs").resolve()

if __name__ == "__main__":

    # while True:
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(observation)
    #     # env.render()

    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                       n_envs=4, vec_env_cls=SubprocVecEnv, env_kwargs={"render_mode": "human"})

    # env = CarParkingEnv(render_mode="human")
    model = A2C("MlpPolicy", env, device="cuda", tensorboard_log=str(LOGS_DIR))
    model.learn(total_timesteps=250_000, progress_bar=True)
    model.save(MODEL_DIR/"testModel")
