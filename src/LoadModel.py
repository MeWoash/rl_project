import os
from pathlib import Path
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC

OUT_RL_DIR = Path(__file__).parent.joinpath("../out/learning").resolve()

modelContructor = A2C

def get_last_modified_file(directory_path, suffix=".zip"):
    latest_time = 0
    latest_file = None

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(suffix):
                filepath = os.path.join(root, filename)
                if os.path.isfile(filepath):
                    file_mtime = os.path.getmtime(filepath)
                    
                    if file_mtime > latest_time:
                        latest_time = file_mtime
                        latest_file = filepath

    if latest_file:
        print(f"Last modified {suffix} file: {latest_file}")
    else:
        print(f"No {suffix} files found.")
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
    
    modelContructor = SAC
    last_model = get_last_modified_file(str(OUT_RL_DIR))
    load_model(modelContructor, last_model)