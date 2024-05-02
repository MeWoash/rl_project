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
    
    def _buffer_overall_stats(self):
        mean_reward = np.mean(self.rewards)
        self.logger.record_mean('observations/mean_reward', mean_reward)
        
        mean_dist = np.mean(self.distance)
        self.logger.record_mean('observations/mean_dist', mean_dist)
    
    def _buffer_episode_stats(self):
        for i in range(self.training_env.num_envs):
            ep = self.infos[i]['episode_number']
            ep_time = round(self.infos[i]['episode_time'], 0)*1000
            
            row: dict[str] = {
                "episode":ep,
                "env":i,
                'episode_time':ep_time,
                'dist':self.distance[i, 0],
                'angle_diff':self.angle_diff[i, 0],
                'pos_X':self.pos[i, 0],
                'pos_Y':self.pos[i, 1],
                'reward':self.rewards[i],
                'velocity':self.velocity[i, 0]
                }
            self.df_episode_buffer[i] = pd.concat([self.df_episode_buffer[i],
                                                   pd.DataFrame.from_dict([row])])
            
    def _check_episodes(self):
        for i in range(self.training_env.num_envs):
            if self.dones[i] == True:
                ep = self.infos[i]['episode_number']
                row = {
                    "episode":ep,
                    "env":i,
                    "reward_sum":self.df_episode_buffer[i]['reward'].sum(),
                    "reward_mean":self.df_episode_buffer[i]['reward'].mean(),
                    "episode_time_max":self.df_episode_buffer[i]['episode_time'].max(),
                    "dist_min":self.df_episode_buffer[i]['dist'].min(),
                    "dist_mean":self.df_episode_buffer[i]['dist'].mean(),
                    "pos_X_mean":self.df_episode_buffer[i]['pos_X'].mean(),
                    "pos_Y_mean":self.df_episode_buffer[i]['pos_Y'].mean()
                    }
                
                self.df_episodes_all = pd.concat([self.df_episodes_all,
                                                  self.df_episode_buffer[i]])
                self.df_episodes_summary = pd.concat([self.df_episodes_summary,
                                                      pd.DataFrame.from_dict([row])])
                self.df_episode_buffer[i].drop(self.df_episode_buffer[i].index, inplace=True)
                
                self.df_episodes_summary.to_csv(self.df_episodes_summary_path, index=False)
                self.df_episodes_all.to_csv(self.df_episodes_all_path, index=False)
                
                self._save_best_model(row['reward_mean'])
                
    def _save_best_model(self, new_reward):
        if new_reward > self.best_reward:
            self.best_reward= new_reward
            self.model.save(Path(self.logdir,'models', f'best_model_rew-{round(self.best_reward*1000,3)}_step-{self.num_timesteps}'))
                
    def _on_step(self) -> bool:
        self.infos = self.locals['infos']
        self.dones = self.locals['dones']
            
        if self.iteration % self.log_interval == 0:
            self.observations = self.locals['new_obs']
            self.rewards = self.locals['rewards']
            
            # autopep8: off
            self.velocity = self.observations[:,ObsIndex.VELOCITY_BEGIN:ObsIndex.VELOCITY_END+1]
            self.distance = self.observations[:,ObsIndex.DISTANCE_BEGIN:ObsIndex.DISTANCE_END+1]
            self.angle_diff = self.observations[:,ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1]
            # self.contact = self.observations[:,ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1]
            # self.range = self.observations[:,ObsIndex.RANGE_BEGIN:ObsIndex.RANGE_END+1]
            self.pos = self.observations[:,ObsIndex.POS_BEGIN:ObsIndex.POS_END+1]
            # self.eul = self.observations[:,ObsIndex.CAR_YAW_BEGIN:ObsIndex.CAR_YAW_END+1]
            
            # autopep8: on
            self._buffer_overall_stats()
            self._buffer_episode_stats()
        self.iteration+=1
        self._check_episodes()
        # print(f"step: {self.locals['n_steps']}/{self.locals['n_rollout_steps']}")
        return True
    
    def _on_training_end(self):
        self.df_episodes_summary.to_csv(self.df_episodes_summary_path, index=False)
        self.df_episodes_all.to_csv(self.df_episodes_all_path, index=False)