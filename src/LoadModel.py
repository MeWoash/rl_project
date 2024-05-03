from pathlib import Path
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC

# OUT_RL_DIR = Path(__file__).parent.joinpath("../out/learning").resolve()

modelContructor = A2C

def test_env():

    env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                            n_envs=1,
                            vec_env_cls=DummyVecEnv,
                            env_kwargs={"render_mode": "human"})
    
    modelConstrucotr = A2C
    
    model = modelConstrucotr.load(rf"out\learning\A2C\A2C_2\models\best_model_rew-823.489_step-69808")
    
    obs = env.reset()
    while True:
        actions, states = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(actions)


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    test_env()