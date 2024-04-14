from pathlib import Path
from CustomEnvs import CarParkingEnv
from CustomEnvs.Callbacks import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C

RL_MODEL_DIR = Path(__file__).parent.joinpath("../out/models").resolve()
RL_LOGS_DIR = Path(__file__).parent.joinpath("../out/logs").resolve()


if __name__ == "__main__":

    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                            n_envs=1,
                            vec_env_cls=SubprocVecEnv,
                            env_kwargs={"render_mode": "none"})
    
    logdir = f"{RL_LOGS_DIR.joinpath('A2C')}"
    print(f"MODEL LOGDIR = {logdir}")
    model = A2C(env=env,
                policy="MlpPolicy",
                tensorboard_log=logdir)
    
    CALLBACKS = [CustomMetricsCallback()]
    
    model.learn(total_timesteps=10_000,
                progress_bar=True,
                callback=CALLBACKS,
                log_interval=1)
    
    
    
    model.save(RL_MODEL_DIR.joinpath(f"A2C/simple_A2C_model"))