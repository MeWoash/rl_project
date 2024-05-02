from pathlib import Path
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC

RL_MODEL_DIR = Path(__file__).parent.joinpath("../out/models").resolve()
RL_LOGS_DIR = Path(__file__).parent.joinpath("../out/logs").resolve()

modelContructor = A2C

def test_env():

    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                            n_envs=1,
                            vec_env_cls=DummyVecEnv,
                            env_kwargs={"render_mode": "human"})
    
    modelConstrucotr = A2C
    
    logdir = f"{RL_LOGS_DIR.joinpath(modelConstrucotr.__name__)}"
    print(f"MODEL LOGDIR = {logdir}")
    
    model = modelConstrucotr.load(rf"D:\kody\rl_project\out\logs\A2C\A2C_5\best_model_rew-82_step-9720.zip")
    
    obs = env.reset()
    while True:
        actions, states = model.predict(obs)
        obs, rewards, dones, infos = env.step(actions)
        
        if dones.any():
            obs = env.reset()


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    test_env()