import gymnasium
import CustomEnvs # Registering envs
from stable_baselines3.common.env_util import make_vec_env



if __name__ == "__main__":
    
    env = gymnasium.make('CustomEnvs/CarParkingEnv-v0', render_mode="human")
    obs = env.reset()
    
    
    while True:

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation)
        # env.render()