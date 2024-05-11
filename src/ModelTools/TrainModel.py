# autopep8: off
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC
import sys

sys.path.append(str(Path(__file__,'..','..').resolve()))
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *
from PathsConfig import *
from MJCFGenerator.Generator import generate_MJCF 
from PostProcessing.PostProcess import generate_media

# autopep8: on

DEFAULT_MAKE_ENV_KWARGS = {
                    "vec_env_cls": DummyVecEnv,
                    "n_envs":1,
                    "render_mode": "none",
                    "enable_random_spawn": False,
                    "enable_spawn_noise": True
                    }

DEFAULT_MODEL_KWARGS = {
    "device":"cuda",
}

def train_model(modelConstructor = SAC,
                total_timesteps = 500_000,
                model_kwargs = DEFAULT_MODEL_KWARGS,
                make_env_kwargs = DEFAULT_MAKE_ENV_KWARGS
                ):
    
    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                            **make_env_kwargs
                            )
    env.seed(0)

    logdir = str(Path(OUT_RL_DIR, modelConstructor.__name__))
    print(f"MODEL LOGDIR = {logdir}")
    
    model = modelConstructor(env=env,
                verbose = 0,
                tensorboard_log=logdir,
                policy="MlpPolicy",
                **model_kwargs
                )
    
    CALLBACKS = [CSVCallback(log_interval = 5)]
    
    model.learn(progress_bar=True,
                callback=CALLBACKS,
                total_timesteps = total_timesteps
                )
    
    env.close()
    return CALLBACKS[0].logdir

if __name__ == "__main__":


    # ===========CONFIG LEARNING===============
    modelConstructor = SAC
    total_timesteps = 300_000
    model_kwargs = {
    "device":"cuda",
    }
    make_env_kwargs = {
    "vec_env_cls": DummyVecEnv,
    "n_envs":1,
    "env_kwargs":{
        "render_mode": "none",
        "enable_random_spawn": True,
        "enable_spawn_noise": True
        }
    }
    # ==========================================
    
    generate_MJCF()
    logdir = train_model(modelConstructor,
                         total_timesteps,
                         model_kwargs,
                         make_env_kwargs)
    generate_media(logdir)
    
    # last_model = LoadModel.get_last_modified_file(Path(logdir,'models'))
    # LoadModel.load_model(modelConstructor, last_model)
    