# autopep8: off

from datetime import datetime
from cv2 import log
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from CustomEnvs.CarParking import EXTRA_OBS_INDEX, OBS_INDEX
import numpy as np
import pandas as pd
import sys

sys.path.append(str(Path(__file__,'..','..').resolve()))
from PostProcessing.PostProcess import *
from ModelTools.Utils import load_generate_csvs

# autopep8: on

class EpisodeStatsBuffer:
    def __init__(self, callback:Type[BaseCallback], env_index, log_interval=20):
        self.df_episode_buffer = pd.DataFrame()
        self.callback = callback
        self.env_index = env_index
        self.log_counter = 0
        self.log_interval = log_interval
    
        
    def update_state(self):
        
        if self.callback.dones[self.env_index] == True:
            if self.callback.infos[self.env_index].get("TimeLimit.truncated", False):
                terminal_observation = self.callback.infos[self.env_index].get("terminal_observation")
                self.callback.velocity[self.env_index] = terminal_observation[OBS_INDEX.VELOCITY_BEGIN:OBS_INDEX.VELOCITY_END+1]
                self.callback.distance[self.env_index] = terminal_observation[OBS_INDEX.DISTANCE_BEGIN:OBS_INDEX.DISTANCE_END+1]
                self.callback.angle_diff[self.env_index] = terminal_observation[OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1]
                # self.callback.pos[self.env_index] = terminal_observation[OBS_INDEX.REL_POS_BEGIN:OBS_INDEX.REL_POS_END+1]
            self._add_stats_to_buffer()
            self._flush_buffer()
            self.log_counter = 0
        else:
            if self.log_counter % self.log_interval==0:
                self._add_stats_to_buffer()
            self.log_counter += 1
    

    def _flush_buffer(self):
        
        self.callback.df_episodes_all = pd.concat([self.callback.df_episodes_all,
                                                   self.df_episode_buffer])
        self.callback.df_episodes_all.to_csv(self.callback.df_episodes_all_path,
                                             index=False)
        self.df_episode_buffer.drop(self.df_episode_buffer.index,
                                                    inplace=True)
        self.callback.evaluate_model = True
 
    def _add_stats_to_buffer(self):
        row: dict[str] = {
            "episode":self.callback.infos[self.env_index]['episode_number'],
            "env":self.env_index,
            "rel_time": time.time() - self.callback.training_start,
            "learning_step": self.callback.model.num_timesteps,
            "episode_mujoco_time": self.callback.infos[self.env_index]['episode_mujoco_time'],
            "episode_env_step":self.callback.infos[self.env_index]['episode_env_step'],
            'dist':self.callback.distance[self.env_index, 0],
            'angle_diff':self.callback.angle_diff[self.env_index, 0],
            'pos_X':self.callback.global_pos[self.env_index, 0],
            'pos_Y':self.callback.global_pos[self.env_index, 1],
            'reward':self.callback.rewards[self.env_index],
            'episode_mean_reward':self.callback.infos[self.env_index]['episode_mean_reward'],
            'velocity':self.callback.velocity[self.env_index, 0],
            'action_engine': self.callback.actions[self.env_index, 0],
            'action_angle': self.callback.actions[self.env_index, 1],
            }
        self.df_episode_buffer = pd.concat([self.df_episode_buffer,
                                                pd.DataFrame.from_dict([row])])

class  CSVCallback(BaseCallback):
    def __init__(self, out_logdir, verbose=0, log_interval=20, max_saved_models = 10, window_size = 100, **kwargs):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.evaluate_model = False
        self.out_logdir = out_logdir
        self.max_saved_models = max_saved_models
        self.saved_models  = []
        self.window_size = window_size
        self.training_start = time.time()
        
    def _init_callback(self):
        
        self.df_episodes_all_path = self.ep_logdir = Path(self.out_logdir).joinpath(EPISODES_ALL).resolve()
        
        self.df_episodes_all:pd.DataFrame = pd.DataFrame()
        self.episode_buffers: list[EpisodeStatsBuffer] = []
        
        for i in range(self.training_env.num_envs):
            buffer = EpisodeStatsBuffer(self, i, self.log_interval)
            self.episode_buffers.append(buffer)

        self.best_reward = 0
            
    def _save_model(self, reward):    
        model_filename = str(Path(self.out_logdir, 'models', f'model-rew_{str(round(reward,3)).replace(".","_")}-step_{self.num_timesteps}'))
        self.saved_models.append(model_filename)
        
        print(f"Saving model at mean reward: {reward:0.3f}, step: {self.num_timesteps}")
        self.model.save(model_filename)
        
        if len(self.saved_models) > self.max_saved_models:
            oldest_model = self.saved_models.pop(0)
            os.remove(oldest_model+".zip")
                
    def _on_training_start(self) -> None:
        self.training_start = time.time()
        return super()._on_training_start()    
                
    def _on_step(self) -> bool:
        self.infos = self.locals['infos']
        self.dones = self.locals['dones']
        self.observations = self.locals['new_obs']
        self.rewards = self.locals['rewards']
        
        self.actions = self.locals.get('clipped_actions')
        if self.actions is None:
            self.actions = self.locals['actions']
        
        self.extra_observations = np.array([info.get("extra_obs") for info in self.infos])  
        
        self.velocity = self.observations[:,OBS_INDEX.VELOCITY_BEGIN:OBS_INDEX.VELOCITY_END+1]
        self.distance = self.observations[:,OBS_INDEX.DISTANCE_BEGIN:OBS_INDEX.DISTANCE_END+1]
        self.angle_diff = self.observations[:,OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1]
        self.rel_pos = self.observations[:,OBS_INDEX.REL_POS_BEGIN:OBS_INDEX.REL_POS_END+1]
        
        self.global_pos = self.extra_observations[:,EXTRA_OBS_INDEX.GLOBAL_POS_BEGIN:EXTRA_OBS_INDEX.GLOBAL_POS_END+1]
            
        for env_index in range(self.training_env.num_envs):
            self.episode_buffers[env_index].update_state()
        
        if self.evaluate_model is True:
            # SAVE BEST MODEL
            self.evaluate_model = False
            last_reward = np.mean(self.df_episodes_all.groupby(['episode', 'env'])['episode_mean_reward'].last().to_numpy()[-self.window_size:])
            if  last_reward > self.best_reward:
                self.best_reward= last_reward
                self._save_model(self.best_reward)
            
            
        return True
    
    def _on_training_end(self):
        load_generate_csvs(self.out_logdir)