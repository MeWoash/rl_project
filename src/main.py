import gymnasium
import CustomEnvs  # Registering envs
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":

    # while True:
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(observation)
    #     # env.render()

    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                       n_envs=8, vec_env_cls=SubprocVecEnv)
    model = A2C("MlpPolicy", env, device="cpu")
    model.learn(total_timesteps=25_000)
