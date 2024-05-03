from ast import List
from cv2 import log
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from CustomEnvs.CarParking import ObsIndex
import numpy as np
import pandas as pd
from ModelTools.PostProcess import *
            
          
class  CSVCallback(BaseCallback):
    def __init__(self, verbose=0, log_interval=50, **kwargs):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.iteration = 0
        
    def _init_callback(self):
        self.logdir=self.logger.get_dir()
        
        self.df_episodes_summary_path = self.ep_logdir = Path(self.logdir).joinpath("episodes_summary.csv")
        self.df_episodes_all_path = self.ep_logdir = Path(self.logdir).joinpath("episodes_all.csv")
        
        self.df_episodes_summary:pd.DataFrame = pd.DataFrame()
        self.df_episodes_all:pd.DataFrame = pd.DataFrame()
        
        self.df_episode_buffer: list[pd.DataFrame] = []
        for i in range(self.training_env.num_envs):
            self.df_episode_buffer.append(pd.DataFrame())
            
        self.best_reward = 0
    
    def _buffer_episode_stats(self, env_index) -> None:
        
        ep = self.infos[env_index]['episode_number']
        ep_time = round(self.infos[env_index]['episode_mujoco_time']*1000)
        ep_step_env = self.infos[env_index]['episode_env_step']
        ep_cum_rew = self.infos[env_index]['episode_cumulative_reward']
        ep_norm_cum_rew = self.infos[env_index]['episode_norm_cumulative_reward']
        
        row: dict[str] = {
            "episode":ep,
            "env":env_index,
            "learning_step": self.model.num_timesteps,
            "episode_mujoco_time":ep_time,
            "episode_env_step":ep_step_env,
            'dist':self.distance[env_index, 0],
            'angle_diff':self.angle_diff[env_index, 0],
            'pos_X':self.pos[env_index, 0],
            'pos_Y':self.pos[env_index, 1],
            'reward':self.rewards[env_index],
            'episode_cum_reward':ep_cum_rew,
            'episode_norm_cum_reward':ep_norm_cum_rew,
            'velocity':self.velocity[env_index, 0],
            'action_engine': self.actions[env_index, 0],
            'action_angle': self.actions[env_index, 1],
            }
        self.df_episode_buffer[env_index] = pd.concat([self.df_episode_buffer[env_index],
                                                pd.DataFrame.from_dict([row])])
            
    def _check_episodes(self):
        for env_index in range(self.training_env.num_envs):
            if self.dones[env_index] == True:
                # BUFFER TERMINAL OBSERVATION IF WASNT LOGGED
                if self.iteration % self.log_interval != 0:
                    self._buffer_episode_stats(env_index)
                
                ep = self.infos[env_index]['episode_number']
                row = {
                    "episode":ep,
                    "env":env_index,
                    
                    "learning_step_min": self.df_episode_buffer[env_index].iloc[0,:]['learning_step'],
                    "learning_step_max": self.df_episode_buffer[env_index].iloc[-1,:]['learning_step'],
                    
                    "episode_cum_reward":self.df_episode_buffer[env_index].iloc[-1,:]['episode_cum_reward'],
                    "episode_norm_cum_reward":self.df_episode_buffer[env_index].iloc[-1,:]['episode_norm_cum_reward'],
                    "episode_mujoco_time":self.df_episode_buffer[env_index].iloc[-1,:]['episode_mujoco_time'],
                    
                    "dist_min":self.df_episode_buffer[env_index]['dist'].min(),
                    "dist_mean":self.df_episode_buffer[env_index]['dist'].mean(),
                    "pos_X_mean":self.df_episode_buffer[env_index]['pos_X'].mean(),
                    "pos_Y_mean":self.df_episode_buffer[env_index]['pos_Y'].mean()
                    }
                
                self.df_episodes_all = pd.concat([self.df_episodes_all,
                                                  self.df_episode_buffer[env_index]])
                self.df_episodes_summary = pd.concat([self.df_episodes_summary,
                                                      pd.DataFrame.from_dict([row])])
                self.df_episode_buffer[env_index].drop(self.df_episode_buffer[env_index].index, inplace=True)
                
                self.df_episodes_summary.to_csv(self.df_episodes_summary_path, index=False)
                self.df_episodes_all.to_csv(self.df_episodes_all_path, index=False)
                
                self._log_episode_end_stats(row)
                self._save_best_model(row['episode_norm_cum_reward'])
    
    def _log_episode_end_stats(self, row):
        self.logger.record('observation/episode_norm_cum_reward',row['episode_norm_cum_reward'])
        self.logger.record('observation/time_max', row['episode_mujoco_time'])
    
    def _save_best_model(self, new_reward):
        if new_reward > self.best_reward:
            self.best_reward= new_reward
            self.model.save(Path(self.logdir,'models', f'best_model_rew-{int(self.best_reward*1000)}_step-{self.num_timesteps}'))
                
    def _on_step(self) -> bool:
        self.infos = self.locals['infos']
        self.dones = self.locals['dones']
            
        if self.iteration % self.log_interval == 0:
            self.observations = self.locals['new_obs']
            self.rewards = self.locals['rewards']
            self.actions = self.locals['actions']
            
            # autopep8: off
            self.velocity = self.observations[:,ObsIndex.VELOCITY_BEGIN:ObsIndex.VELOCITY_END+1]
            self.distance = self.observations[:,ObsIndex.DISTANCE_BEGIN:ObsIndex.DISTANCE_END+1]
            self.angle_diff = self.observations[:,ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1]
            # self.contact = self.observations[:,ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1]
            # self.range = self.observations[:,ObsIndex.RANGE_BEGIN:ObsIndex.RANGE_END+1]
            self.pos = self.observations[:,ObsIndex.POS_BEGIN:ObsIndex.POS_END+1]
            # self.eul = self.observations[:,ObsIndex.CAR_YAW_BEGIN:ObsIndex.CAR_YAW_END+1]
            
            # autopep8: on
            for env_index in range(self.training_env.num_envs):
                self._buffer_episode_stats(env_index)
                
        self.iteration+=1
        self._check_episodes()
        # print(f"step: {self.locals['n_steps']}/{self.locals['n_rollout_steps']}")
        return True
    
    def _on_training_end(self):
        self.df_episodes_summary.to_csv(self.df_episodes_summary_path, index=False)
        self.df_episodes_all.to_csv(self.df_episodes_all_path, index=False)