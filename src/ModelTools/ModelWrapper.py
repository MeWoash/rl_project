from CustomEnvs.CarParking import CarParkingEnv
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import A2C

RL_MODEL_DIR = Path(__file__).parent.joinpath("../../out/models").resolve()
RL_LOGS_DIR = Path(__file__).parent.joinpath("../../out/logs").resolve()


class ModelWrapper:

    def __init__(self) -> None:
        self.model = None
        self.env = None

    def create_vec_env(self, n_envs, render_mode, **kwargs):
        self.env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                                n_envs=n_envs,
                                vec_env_cls=SubprocVecEnv,
                                env_kwargs={"render_mode": render_mode})

    def create_gym_env(self, render_mode, **kwargs):
        self.env = CarParkingEnv(render_mode=render_mode)

    def create_model(self,
                     modelContructor,
                     modelArgs,
                     **kwargs):

        self.modelName = modelContructor.__name__
        self.model = modelContructor(
            env=self.env,
            tensorboard_log=f"{RL_LOGS_DIR.joinpath(str(self.modelName))}",
            **modelArgs)

    def learn_model(self, n_steps=10_000):
        self.model.learn(total_timesteps=n_steps, progress_bar=True)

    def save_model(self):
        self.model.save(RL_MODEL_DIR.joinpath(
            f"{self.modelName}/simple_{self.modelName}_model"))

    def load_model(self):
        path = RL_MODEL_DIR.joinpath(
            f"{self.modelName}/simple_{self.modelName}_model")
        self.model.load(path)
        print(f"loaded: {path}")
