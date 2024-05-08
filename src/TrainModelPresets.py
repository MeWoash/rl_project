from pathlib import Path
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC
from TrainModel import *



if __name__ == "__main__":

    TRAIN_VECTOR = \
    [
        # ===========CONFIG LEARNING===============
        {
            "modelConstructor" : SAC,
            "total_timesteps" : 300_000,
            "model_kwargs" : {
                "device":"cuda",
            },
            "make_env_kwargs" : {
                "vec_env_cls": DummyVecEnv,
                "n_envs":1,
                "env_kwargs":{
                    "render_mode": "none",
                    "enable_random_spawn": True,
                    "enable_spawn_noise": True
                }
            }
        },
        # ===========CONFIG LEARNING===============
        {
            "modelConstructor" : A2C,
            "total_timesteps" : 300_000,
            "model_kwargs" : {
                "device":"cuda",
            },
            "make_env_kwargs" : {
                "vec_env_cls": DummyVecEnv,
                "n_envs":1,
                "env_kwargs":{
                    "render_mode": "none",
                    "enable_random_spawn": True,
                    "enable_spawn_noise": True
                }
            }
        },
        # ===========CONFIG LEARNING===============
        {
            "modelConstructor" : SAC,
            "total_timesteps" : 300_000,
            "model_kwargs" : {
                "device":"cuda",
            },
            "make_env_kwargs" : {
                "vec_env_cls": DummyVecEnv,
                "n_envs":1,
                "env_kwargs":{
                    "render_mode": "none",
                    "enable_random_spawn": False,
                    "enable_spawn_noise": True
                }
            }
        },
        # ===========CONFIG LEARNING===============
        {
            "modelConstructor" : A2C,
            "total_timesteps" : 300_000,
            "model_kwargs" : {
                "device":"cuda",
            },
            "make_env_kwargs" : {
                "vec_env_cls": DummyVecEnv,
                "n_envs":1,
                "env_kwargs":{
                    "render_mode": "none",
                    "enable_random_spawn": False,
                    "enable_spawn_noise": True
                }
            }
        },
        
    ]
    
    for kwargs in TRAIN_VECTOR:
        logdir = train_model(**kwargs)
        try:
            generate_media(logdir)
        except Exception as e:
            print(e)