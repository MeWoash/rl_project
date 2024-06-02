from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.common.noise import VectorizedActionNoise, OrnsteinUhlenbeckActionNoise, NormalActionNoise
import numpy as np


TRAIN_VECTOR = [
    {
        "name": "1_episode_freq_10_gradients",
        "modelConstructor" : SAC,
        "total_timesteps" : 50_000_000,
        "seed": 1,
        "normalize": False,
        "model_kwargs" : {
            "device":"cuda",
            "buffer_size":5_000_000,
            "batch_size":1_000,
            "use_sde":True,
            "train_freq":(1, "episode"),
            "gradient_steps":10,
        },
        "make_env_kwargs" : {
            "vec_env_cls": SubprocVecEnv,
            "n_envs": 1,
            "env_kwargs":{
                "render_mode": "none",
                "enable_random_spawn": True,
                "enable_spawn_noise": True
            }
        }
    },
]