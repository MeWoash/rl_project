from ast import List
from cv2 import log
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from CustomEnvs.CarParking import ObsIndex
import numpy as np
import pandas as pd
from ModelTools.PostProcess import *

class  CustomMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, log_interval=100, **kwargs):
        super().__init__(verbose)
        self.loggers: list[SummaryWriter] = []
        self.log_interval = log_interval
        self.iteration = 0
        
    def _init_callback(self):
        self.logdir=self.logger.get_dir()
        # infos = self.training_env.get_attr('info')
        
        for i in range(self.training_env.num_envs):
            # ep = infos[i]['episode_number']
            path = self._create_log_name(self.logdir, i, 0)
            # print("Sublog:",path)
            self.loggers.append(SummaryWriter(path))
    
    def _create_log_name(self, logdir, nth_vec, episode):
        return Path(logdir).joinpath(f"./ep-{episode}_env-{nth_vec}")
    
    def _overall_stats(self):
        mean_reward = np.mean(self.rewards)
        self.logger.record_mean('observations/mean_reward', mean_reward)
    
    def _episode_stats(self):
        for i in range(self.training_env.num_envs):
            
            ep_time = round(self.infos[i]['episode_time'], 3)*1000
            
            self.loggers[i].add_scalar("episode/dist_to_target", self.distance[i, 0], ep_time)
            self.loggers[i].add_scalar("episode/reward", self.rewards[i], ep_time)
            self.loggers[i].add_scalar("episode/pos_X", self.pos[i, 0], ep_time)
            self.loggers[i].add_scalar("episode/pos_Y", self.pos[i, 1], ep_time)
            self.loggers[i].add_scalar("episode/velocity", self.velocity[i, 0], ep_time)
            self.loggers[i].add_scalar("episode/angle_diff", self.angle_diff[i, 0], ep_time)
            # self.loggers[i].flush()
            
    def _check_episodes(self):
        for i in range(self.training_env.num_envs):
            if self.dones[i] == True:
                ep = self.infos[i]['episode_number']
                path = self._create_log_name(self.logdir, i, ep)
                # self.loggers[i].flush()
                self.loggers[i].close()
                # del self.loggers[i]
                self.loggers[i] = SummaryWriter(path)
    
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
            # self.eul = self.observations[:,ObsIndex.EUL_BEGIN:ObsIndex.EUL_END+1]
            
            # autopep8: on
            self._overall_stats()
            self._episode_stats()
        self.iteration+=1
        self._check_episodes()
        
        return True
    
    def _on_training_end(self):
        for i in range(self.training_env.num_envs):
            self.loggers[i].close()
            
            
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
    
    def _create_log_name(self, logdir, nth_vec, episode):
        return Path(logdir).joinpath(f"./ep-{episode}_env-{nth_vec}.csv")
    
    def _overall_stats(self):
        mean_reward = np.mean(self.rewards)
        self.logger.record_mean('observations/mean_reward', mean_reward)
    
    def _episode_stats(self):
        for i in range(self.training_env.num_envs):
            ep = self.infos[i]['episode_number']
            ep_time = round(self.infos[i]['episode_time'], 3)*1000
            
            row: dict[str] = {
                "episode":ep,
                "env":i,
                'step':ep_time,
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
                    "step_max":self.df_episode_buffer[i]['step'].max(),
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
            # self.eul = self.observations[:,ObsIndex.EUL_BEGIN:ObsIndex.EUL_END+1]
            
            # autopep8: on
            self._overall_stats()
            self._episode_stats()
        self.iteration+=1
        self._check_episodes()
        
        return True
    
    def _on_training_end(self):
        self.df_episodes_summary.to_csv(self.df_episodes_summary_path, index=False)
        self.df_episodes_all.to_csv(self.df_episodes_all_path, index=False)
        
        do_basic_analysis(self.logdir)