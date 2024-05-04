from pathlib import Path
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC
import LoadModel

OUT_RL_DIR = Path(__file__).parent.joinpath("../out/learning").resolve()


def train_model(modelConstructor = A2C, n_envs = 16, total_timesteps = 200_000):
    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                            n_envs=n_envs,
                            vec_env_cls=SubprocVecEnv,
                            env_kwargs={"render_mode": "none"})
    

    logdir = f"{OUT_RL_DIR.joinpath(modelConstructor.__name__)}"
    print(f"MODEL LOGDIR = {logdir}")
    
    model = modelConstructor(env=env,
                policy="MlpPolicy",
                device= "auto",
                tensorboard_log=logdir,
                )
    
    CALLBACKS = [CSVCallback(log_interval = 20)]
    
    model.learn(total_timesteps=total_timesteps,
                progress_bar=True,
                callback=CALLBACKS)
    
    env.close()
    return CALLBACKS[0].logdir

if __name__ == "__main__":


    modelConstructor = A2C
    n_envs = 16
    total_timesteps = 200_000
    logdir = train_model(modelConstructor, n_envs, total_timesteps)
    generate_media(logdir)
    
    # last_model = LoadModel.get_last_modified_file(Path(logdir,'models'))
    # LoadModel.load_model(modelConstructor, last_model)
    