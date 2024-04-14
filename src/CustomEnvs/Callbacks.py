from ast import List
from cv2 import log
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from CustomEnvs.CarParking import ObsIndex
import numpy as np

class  CustomMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, log_interval=100, **kwargs):
        super().__init__(verbose)
        self.loggers: List[SummaryWriter] = []
        self.log_interval = log_interval
        self.iteration = 0
        
    def _init_callback(self):
        self.logdir=self.logger.get_dir()
        # infos = self.training_env.get_attr('info')
        
        for i in range(self.training_env.num_envs):
            # ep = infos[i]['episode_number']
            path = self._create_log_name(self.logdir, i, 0)
            print("Sublog:",path)
            self.loggers.append(SummaryWriter(path))
    
    def _create_log_name(self, logdir, nth_vec, episode):
        return Path(logdir).joinpath(f"./ep-{episode}_env-{nth_vec}")
    
    def _overall_stats(self):
        mean_reward = np.mean(self.rewards)
        self.logger.record_mean('observations/mean_reward', mean_reward)
    
    def _episode_stats(self):
        for i in range(self.training_env.num_envs):
            self.loggers[i].add_scalar("episode/dist_to_target", self.distance[i, 0], self.infos[i]['episode_time'])
            self.loggers[i].add_scalar("episode/reward", self.rewards[i], self.infos[i]['episode_time'])
            self.loggers[i].add_scalar("episode/pos_X", self.pos[i, 0], self.infos[i]['episode_time'])
            self.loggers[i].add_scalar("episode/pos_Y", self.pos[i, 1], self.infos[i]['episode_time'])   
            # self.loggers[i].flush()
            
    def _check_episodes(self):
        for i in range(self.training_env.num_envs):
            if self.dones[i] == True:
                ep = self.infos[i]['episode_number']
                path = self._create_log_name(self.logdir, i, ep)
                # self.loggers[i].flush()
                self.loggers[i].close()
                self.loggers[i] = SummaryWriter(path)
    
    def _on_step(self) -> bool:
        
        if self.iteration % self.log_interval == 0:
            self.observations = self.locals['new_obs']
            self.rewards = self.locals['rewards']
            self.infos = self.locals['infos']
            self.dones = self.locals['dones']
            
            # autopep8: off
            
            # self.velocity = self.observations[:,ObsIndex.VELOCITY_BEGIN:ObsIndex.VELOCITY_END+1]
            self.distance = self.observations[:,ObsIndex.DISTANCE_BEGIN:ObsIndex.DISTANCE_END+1]
            # self.angle_diff = self.observations[:,ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1]
            # self.contact = self.observations[:,ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1]
            # self.range = self.observations[:,ObsIndex.RANGE_BEGIN:ObsIndex.RANGE_END+1]
            self.pos = self.observations[:,ObsIndex.POS_BEGIN:ObsIndex.POS_END+1]
            # self.eul = self.observations[:,ObsIndex.EUL_BEGIN:ObsIndex.EUL_END+1]
            
            # autopep8: on
            self._overall_stats()
            self._episode_stats()
            self._check_episodes()
        self.iteration+=1
        
        return True