import gymnasium
import CustomEnvs # Registering envs
from stable_baselines3.common.env_util import make_vec_env



if __name__ == "__main__":
    
    env = gymnasium.make('CustomEnvs/CarParkingEnv-v0')
    obs = env.reset()
    while True:
        action = None
        obs, rewards, dones, info = env.step(action)
        env.render("human")