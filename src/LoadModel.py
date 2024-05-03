import os
from pathlib import Path
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC

# OUT_RL_DIR = Path(__file__).parent.joinpath("../out/learning").resolve()

modelContructor = A2C

def get_last_modified_file(directory_path):
    latest_time = 0
    latest_file = None

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            file_mtime = os.path.getmtime(filepath)
            
            if file_mtime > latest_time:
                latest_time = file_mtime
                latest_file = filepath
    print(f"Last file {latest_file}")         
    return latest_file


def load_model(modelConstructor, model_path):
    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                            n_envs=1,
                            vec_env_cls=DummyVecEnv,
                            env_kwargs={"render_mode": "human"})
    
    model = modelConstructor.load(model_path)
    
    obs = env.reset()
    while True:
        actions, states = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(actions)


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    
    last_model = get_last_modified_file(rf"out\learning\A2C\A2C_3\models")
    load_model(A2C, last_model)