from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.common.noise import VectorizedActionNoise, OrnsteinUhlenbeckActionNoise, NormalActionNoise
import numpy as np


TRAIN_VECTOR = [
    {
        "name": "16env_20step_noise",
        "modelConstructor" : SAC,
        "total_timesteps" : 50_000_000,
        "seed": 1,
        "normalize": False,
        "model_kwargs" : {
            "device":"cuda",
            # "buffer_size":10_000_000,
            "use_sde":True,
            "train_freq":(20, "step")
        },
        "make_env_kwargs" : {
            "vec_env_cls": SubprocVecEnv,
            "n_envs": 16,
            "env_kwargs":{
                "render_mode": "none",
                "enable_random_spawn": True,
                "enable_spawn_noise": True
            }
        }
    }
]