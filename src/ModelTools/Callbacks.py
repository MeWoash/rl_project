# autopep8: off

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
        
    def flush_all(self):
        self._add_stats_to_buffer()
        self._flush_summary()
        self._flush_buffer()
        self.log_counter = 0
        
    def update_state(self):
        
        if self.callback.dones[self.env_index] == True:
            if self.callback.infos[self.env_index].get("TimeLimit.truncated", False):
                terminal_observation = self.callback.infos[self.env_index].get("terminal_observation")
                self.callback.velocity[self.env_index] = terminal_observation[OBS_INDEX.VELOCITY_BEGIN:OBS_INDEX.VELOCITY_END+1]
                self.callback.distance[self.env_index] = terminal_observation[OBS_INDEX.DISTANCE_BEGIN:OBS_INDEX.DISTANCE_END+1]
                self.callback.angle_diff[self.env_index] = terminal_observation[OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1]
                # self.callback.pos[self.env_index] = terminal_observation[OBS_INDEX.REL_POS_BEGIN:OBS_INDEX.REL_POS_END+1]
            self.flush_all()
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
            
            "episode_cum_reward":self.df_episode_buffer.iloc[-1,:]['episode_cum_reward'],
            "episode_mean_reward":self.df_episode_buffer.iloc[-1,:]['episode_mean_reward'],
            "episode_mujoco_time":self.df_episode_buffer.iloc[-1,:]['episode_mujoco_time'],
            
            "dist_min":self.df_episode_buffer['dist'].min(),
            "dist_mean":self.df_episode_buffer['dist'].mean(),
            "pos_X_mean":self.df_episode_buffer['pos_X'].mean(),
            "pos_Y_mean":self.df_episode_buffer['pos_Y'].mean()
            }
        self.callback.df_episodes_summary = pd.concat([self.callback.df_episodes_summary,
                                                       pd.DataFrame.from_dict([summary_row])])
        
        self.callback.df_episodes_summary.to_csv(self.callback.df_episodes_summary_path,
                                                 index=False)
        
        self.callback._log_episode_end_stats(summary_row)
        self.callback._save_best_model(summary_row['episode_mean_reward'])
        
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
            "episode_mujoco_time": round(self.callback.infos[self.env_index]['episode_mujoco_time']*1000),
            "episode_env_step":self.callback.infos[self.env_index]['episode_env_step'],
            'dist':self.callback.distance[self.env_index, 0],
            'angle_diff':self.callback.angle_diff[self.env_index, 0],
            'pos_X':self.callback.global_pos[self.env_index, 0],
            'pos_Y':self.callback.global_pos[self.env_index, 1],
            'reward':self.callback.rewards[self.env_index],
            'episode_cum_reward':self.callback.infos[self.env_index]['episode_cumulative_reward'],
            'episode_mean_reward':self.callback.infos[self.env_index]['episode_mean_reward'],
            'velocity':self.callback.velocity[self.env_index, 0],
            'action_engine': self.callback.actions[self.env_index, 0],
            'action_angle': self.callback.actions[self.env_index, 1],
            }
        self.df_episode_buffer = pd.concat([self.df_episode_buffer,
                                                pd.DataFrame.from_dict([row])])

class  CSVCallback(BaseCallback):
    def __init__(self, out_logdir, verbose=0, log_interval=20, **kwargs):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.iteration = 0
        self.out_logdir = out_logdir
        
    def _init_callback(self):
        
        self.df_episodes_summary_path = self.ep_logdir = Path(self.out_logdir).joinpath("episodes_summary.csv")
        self.df_episodes_all_path = self.ep_logdir = Path(self.out_logdir).joinpath("episodes_all.csv")
        
        self.df_episodes_summary:pd.DataFrame = pd.DataFrame()
        self.df_episodes_all:pd.DataFrame = pd.DataFrame()
        self.df_training_stats:pd.DataFrame = pd.DataFrame()
        
        self.episode_buffers: list[EpisodeStatsBuffer] = []
        
        for i in range(self.training_env.num_envs):
            buffer = EpisodeStatsBuffer(self, i, self.log_interval)
            self.episode_buffers.append(buffer)

        self.best_reward = 0
    
    def _log_episode_end_stats(self, row):
        self.logger.record('observation/episode_mean_reward',row['episode_mean_reward'])
        self.logger.record('observation/dist_mean', row['dist_mean'])
        self.logger.record('observation/time_max', row['episode_mujoco_time'])
    
    def _save_best_model(self, new_reward):
        if new_reward > self.best_reward:
            self.best_reward= new_reward
            print(f"New best mean reward: {self.best_reward:0.3f} at step: {self.num_timesteps}")
            self.model.save(Path(self.out_logdir,'models', f'best_model_rew-{int(self.best_reward*1000)}_step-{self.num_timesteps}'))
                
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
        
        # autopep8: off
        self.velocity = self.observations[:,OBS_INDEX.VELOCITY_BEGIN:OBS_INDEX.VELOCITY_END+1]
        self.distance = self.observations[:,OBS_INDEX.DISTANCE_BEGIN:OBS_INDEX.DISTANCE_END+1]
        self.angle_diff = self.observations[:,OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1]
        self.rel_pos = self.observations[:,OBS_INDEX.REL_POS_BEGIN:OBS_INDEX.REL_POS_END+1]
        
        self.global_pos = self.extra_observations[:,EXTRA_OBS_INDEX.GLOBAL_POS_BEGIN:EXTRA_OBS_INDEX.GLOBAL_POS_END+1]
            
        # autopep8: on
        for env_index in range(self.training_env.num_envs):
            self.episode_buffers[env_index].update_state()
                

        return True
    
    # def _on_training_end(self):
    #     self.df_episodes_summary.to_csv(self.df_episodes_summary_path, index=False)
    #     self.df_episodes_all.to_csv(self.df_episodes_all_path, index=False)