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
            self._flush_summary()
            self._flush_buffer()
            self.log_counter = 0
        else:
            if self.log_counter % self.log_interval==0:
                self._add_stats_to_buffer()
            self.log_counter += 1
    
    def _flush_summary(self):
        summary_row = {
            "episode":self.callback.infos[self.env_index]['episode_number'],
            "env":self.env_index,
            
            "learning_step_min": self.df_episode_buffer.iloc[0,:]['learning_step'],
            "learning_step_max": self.df_episode_buffer.iloc[-1,:]['learning_step'],
            
            "episode_mean_reward":self.df_episode_buffer.iloc[-1,:]['episode_mean_reward'],
            "episode_mujoco_time":self.df_episode_buffer.iloc[-1,:]['episode_mujoco_time'],
            }
        self.callback.df_episodes_summary = pd.concat([self.callback.df_episodes_summary,
                                                       pd.DataFrame.from_dict([summary_row])])
        
        self.callback.df_episodes_summary.to_csv(self.callback.df_episodes_summary_path,
                                                 index=False)
        
        
    def _flush_buffer(self):
        
        self.callback.df_episodes_all = pd.concat([self.callback.df_episodes_all,
                                                   self.df_episode_buffer])
        
        self.df_episode_buffer.drop(self.df_episode_buffer.index,
                                                    inplace=True)
        
        self.callback.df_episodes_all.to_csv(self.callback.df_episodes_all_path,
                                             index=False)
 
    def _add_stats_to_buffer(self):
        row: dict[str] = {
            "episode":self.callback.infos[self.env_index]['episode_number'],
            "env":self.env_index,
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
    def __init__(self, out_logdir, verbose=0, log_interval=20, max_saved_models = 5, window_size = 100, **kwargs):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.last_n_episodes = 0
        self.out_logdir = out_logdir
        self.max_saved_models = max_saved_models
        self.saved_models  = []
        self.window_size = window_size
        
    def _init_callback(self):
        
        self.df_episodes_summary_path = self.ep_logdir = Path(self.out_logdir).joinpath("episodes_summary.csv").resolve()
        self.df_episodes_all_path = self.ep_logdir = Path(self.out_logdir).joinpath("episodes_all.csv").resolve()
        self.df_training_stats_path = self.ep_logdir = Path(self.out_logdir).joinpath("training_stats.csv").resolve()
        
        self.df_episodes_summary:pd.DataFrame = pd.DataFrame()
        self.df_episodes_all:pd.DataFrame = pd.DataFrame()
        self.df_training_stats:pd.DataFrame = pd.DataFrame()
        
        self.episode_buffers: list[EpisodeStatsBuffer] = []
        
        for i in range(self.training_env.num_envs):
            buffer = EpisodeStatsBuffer(self, i, self.log_interval)
            self.episode_buffers.append(buffer)

        self.best_reward = 0
    
    def _log_training_stats(self):
        
        if len(self.df_episodes_summary) >= self.window_size:
            window = self.df_episodes_summary.iloc[-self.window_size:]
        else:
            window = self.df_episodes_summary
            
        row: dict[str] = {
            "learning_step": self.num_timesteps,
            "timestamp": time.time(),
            "mean_time_end": window['episode_mujoco_time'].mean(),
            "mean_reward": window['episode_mean_reward'].mean()
            }
        
        self.df_training_stats = pd.concat([self.df_training_stats,
                                                pd.DataFrame.from_dict([row])])
        self.df_training_stats.to_csv(self.df_training_stats_path,
                                             index=False)
        
        # TENSORBOARD
        self.logger.record('episodes_rolling_mean/time_end', row['mean_time_end'])
        self.logger.record('episodes_rolling_mean/reward', row['mean_reward'])
            
    def _save_model(self, reward=None):
        if reward is None:
            reward = self.df_episodes_summary['episode_mean_reward'].mean()
            
        model_filename = str(Path(self.out_logdir, 'models', f'model-rew_{str(round(reward,3)).replace(".","_")}-step_{self.num_timesteps}-ep_{self.df_episodes_summary.shape[0]}'))
        self.saved_models.append(model_filename)
        
        print(f"Saving model at mean reward: {reward:0.3f}, step: {self.num_timesteps}, ep: {self.df_episodes_summary.shape[0]}")
        self.model.save(model_filename)
        
        if len(self.saved_models) > self.max_saved_models:
            oldest_model = self.saved_models.pop(0)
            os.remove(oldest_model+".zip")
                
    def _on_training_start(self) -> None:
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
        
        
        # UPDATE TRAINING LOGS IF TOTAL EP NUMBER CHANGED
        if self.last_n_episodes != len(self.df_episodes_all):
            self.last_n_episodes = len(self.df_episodes_all)
            self._log_training_stats()
            
            # SAVE BEST MODEL
            last_reward = self.df_training_stats.iloc[-1,:]['mean_reward']
            if  last_reward > self.best_reward:
                self.best_reward= last_reward
                self._save_model(self.best_reward)

        return True
    
    def _on_training_end(self):
        self._save_model()