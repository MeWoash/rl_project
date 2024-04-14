from stable_baselines3.common.callbacks import BaseCallback

from CustomEnvs.CarParking import ObsIndex
import numpy as np

class  CustomMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, **kwargs):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # rewards = self.training_env.get_attr('reward')
        # print(rewards)
        observations = np.array(self.training_env.get_attr('observation'))
        mean_obs = observations.mean(axis=0)
        
        velocity = mean_obs[ObsIndex.VELOCITY_BEGIN:ObsIndex.VELOCITY_END+1]
        distance = mean_obs[ObsIndex.DISTANCE_BEGIN:ObsIndex.DISTANCE_END+1]
        angle_diff = mean_obs[ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1]
        contact = mean_obs[ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1]
        range = mean_obs[ObsIndex.RANGE_BEGIN:ObsIndex.RANGE_END+1]
        pos = mean_obs[ObsIndex.POS_BEGIN:ObsIndex.POS_END+1]
        eul = mean_obs[ObsIndex.EUL_BEGIN:ObsIndex.EUL_END+1]

        self.logger.record('observations/dist_to_target', distance[0])
        self.logger.record('observations/mean_X', pos[0])
        self.logger.record('observations/mean_Y', pos[1])
        return True
    
    
class TensorboardPerEpisodeCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super(TensorboardPerEpisodeCallback, self).__init__(verbose)
        self.logger = None
        self.log_path = log_path
        self.total_rewards = []
        self.episode_number = 0  # Licznik epizodów

    def _init_callback(self):
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(self.log_path)

    def _on_step(self):
        self.total_rewards.append(self.locals['rewards'][0])
        if self.locals['dones'][0]:  # Jeśli epizod się zakończył
            self.episode_number += 1  # Zwiększ licznik epizodów
            episode_reward = sum(self.total_rewards)
            # Rejestruj średnią nagrodę dla epizodu przy użyciu numeru epizodu jako klucza czasu
            self.logger.add_scalar('average_reward', episode_reward / len(self.total_rewards), self.episode_number)
            self.total_rewards = []  # Resetuj nagrody dla nowego epizodu
        return True

    def _on_training_end(self):
        self.logger.close()