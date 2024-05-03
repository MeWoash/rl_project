from pathlib import Path
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC

OUT_RL_DIR = Path(__file__).parent.joinpath("../out/learning").resolve()


if __name__ == "__main__":

    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                            n_envs=16,
                            vec_env_cls=SubprocVecEnv,
                            env_kwargs={"render_mode": "none"})
    
    modelConstrucotr = A2C
    
    logdir = f"{OUT_RL_DIR.joinpath(modelConstrucotr.__name__)}"
    print(f"MODEL LOGDIR = {logdir}")
    
    model = modelConstrucotr(env=env,
                policy="MlpPolicy",
                device= "auto",
                tensorboard_log=logdir)
    
    CALLBACKS = [CSVCallback(log_interval = 20)]
    
    model.learn(total_timesteps=200_000,
                progress_bar=True,
                callback=CALLBACKS)
    
    # do_basic_analysis(CALLBACKS[0].logdir)