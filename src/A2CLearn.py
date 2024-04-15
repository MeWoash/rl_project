from pathlib import Path
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C

RL_MODEL_DIR = Path(__file__).parent.joinpath("../out/models").resolve()
RL_LOGS_DIR = Path(__file__).parent.joinpath("../out/logs").resolve()


if __name__ == "__main__":

    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                            n_envs=16,
                            vec_env_cls=SubprocVecEnv,
                            env_kwargs={"render_mode": "none"})
    
    logdir = f"{RL_LOGS_DIR.joinpath('A2C')}"
    print(f"MODEL LOGDIR = {logdir}")
    model = A2C(env=env,
                policy="MlpPolicy",
                device= "cuda",
                tensorboard_log=logdir)
    
    CALLBACKS = [CSVCallback(log_interval = 20)]
    
    model.learn(total_timesteps=5_000_000,
                progress_bar=True,
                callback=CALLBACKS)
    
    
    
    model.save(RL_MODEL_DIR.joinpath(f"A2C/simple_A2C_model"))