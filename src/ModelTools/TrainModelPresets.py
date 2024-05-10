# autopep8: off
import sys
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC

sys.path.append(str(Path(__file__,'..','..').resolve()))
from ModelTools.TrainModel import train_model
from ModelTools.Callbacks import *
from MJCFGenerator.Generator import generate_MJCF 
# autopep8: on

if __name__ == "__main__":

    TRAIN_VECTOR = \
    [
        # ===========CONFIG LEARNING===============
        {
            "modelConstructor" : A2C,
            "total_timesteps" : 10_000,
            "model_kwargs" : {
                "device":"cpu"
            },
            "make_env_kwargs" : {
                "vec_env_cls": DummyVecEnv,
                "n_envs": 1,
                "env_kwargs":{
                    "render_mode": "none",
                    "enable_random_spawn": True,
                    "enable_spawn_noise": True
                }
            }
        }
        
    ]
    generate_MJCF()
    for kwargs in TRAIN_VECTOR:
        logdir = train_model(**kwargs)
        try:
            generate_media(logdir)
        except Exception as e:
            print(e)