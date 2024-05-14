# autopep8: off
from  datetime import datetime
import json
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC
import sys
import pickle

sys.path.append(str(Path(__file__,'..','..').resolve()))
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *
from PathsConfig import *
from MJCFGenerator.Generator import generate_MJCF 
from PostProcessing.PostProcess import generate_media
from stable_baselines3.common.vec_env import VecNormalize
from TrainCfg import TRAIN_VECTOR

# autopep8: on

class LearningContainer:
    def __init__(self) -> None:
        self.out_logdir = None
    
    def init_from_preset(self, train_preset):
        self.train_preset = train_preset
        return self
    
        
    def train_model(self):
        
        self.out_logdir = Path(OUT_LEARNING_DIR, f"{self.train_preset['name']}_{self.train_preset['modelConstructor'].__name__}_{datetime.now().strftime(rf'%d-%m-%y-%H-%M-%S')}").resolve()
        self.out_logdir.mkdir(parents=True, exist_ok=False)
        self.out_logdir.joinpath('models').mkdir(parents=True, exist_ok=True)
        
        print(f"MODEL LOGDIR = {self.out_logdir}")
        
        # SAVE INFO ABOUT TRAINING
        with open(str(self.out_logdir.joinpath('train_preset.pkl').resolve()), 'wb') as file:
            pickle.dump(self.train_preset, file)
        with open(str(self.out_logdir.joinpath('train_preset.txt').resolve()), 'w') as file:
            file.write(str(self.train_preset))
         
        training_env =  make_vec_env("CustomEnvs/CarParkingEnv-v0", **self.train_preset['make_env_kwargs'])
        if self.train_preset['normalize']:
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        model = self.train_preset['modelConstructor'](env=training_env,
                    verbose = 0,
                    tensorboard_log=str(self.out_logdir),
                    policy="MlpPolicy",
                    **self.train_preset['model_kwargs']
                    )
        callbacks = [CSVCallback(out_logdir = str(self.out_logdir), log_interval = 20)]
        try:
            model.learn(progress_bar=True,
                        callback=callbacks,
                        total_timesteps = self.train_preset['total_timesteps'])
        except KeyboardInterrupt:
            print("Training interrupted. Saving last model...")
            name = self.out_logdir.joinpath('models','last_model')
            callbacks[0]._save_model()
        training_env.close()
        
        
    def generate_media(self):
        if self.out_logdir is None:
            print("Log directory does not exist!")
        try:
            generate_media(self.out_logdir)
        except Exception as e:
            print(f"Failed to generate!\n{e}")
    
    def run_model(self, model_path):
        
        env = make_vec_env("CustomEnvs/CarParkingEnv-v0",
                                n_envs=1,
                                vec_env_cls=DummyVecEnv,
                                env_kwargs={"render_mode": "human"},
                                )
        if self.train_preset["normalize"]:
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            
        
        model = self.train_preset["modelConstructor"].load(model_path)
        obs = env.reset()
        while True:
            actions, states = model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = env.step(actions)

def train_models(): 
    generate_MJCF()
    for train_preset in TRAIN_VECTOR:
        learningContainer = LearningContainer()
        learningContainer.init_from_preset(train_preset).train_model()
        learningContainer.generate_media()
        
def run_model(path):
    model_path = path
    if not str(path).endswith(".zip"):
        model_path = get_last_modified_file(model_path)
    train_preset_path = Path(model_path,"..","..","train_preset.pkl").resolve()
    
    with open(train_preset_path, 'rb') as file:
        train_preset = pickle.load(file)
    
    learningContainer = LearningContainer()
    learningContainer.init_from_preset(train_preset)
    learningContainer.run_model(model_path)
    

if __name__ == "__main__":
    train_models()
    # run_model(Path(OUT_LEARNING_DIR,''))