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
from PostProcessing.PostProcess import generate_media
# autopep8: on

if __name__ == "__main__":

    TRAIN_VECTOR = \
    [
        # ===========CONFIG LEARNING===============
        {
            "modelConstructor" : SAC,
            "total_timesteps" : 5_000_000,
            "model_kwargs" : {
                "device":"cuda"
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
        }
        
    ]
    generate_MJCF()
    for kwargs in TRAIN_VECTOR:
        logdir = train_model(**kwargs)
        try:
            generate_media(logdir)
        except Exception as e:
            print(e)