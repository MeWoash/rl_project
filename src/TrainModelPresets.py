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
        {
            "modelConstructor":SAC,
            "n_envs":1,
            "total_timesteps":1_000_000,
        },
        {
            "modelConstructor":A2C,
            "n_envs":16,
            "total_timesteps":5_000_000,
        },
        {
            "modelConstructor":SAC,
            "n_envs":16,
            "total_timesteps":5_000_000,
        }
    ]
    
    for kwargs in TRAIN_VECTOR:
        logdir = train_model(**kwargs)
        generate_media(logdir)
    