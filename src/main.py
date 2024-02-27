import gymnasium
import CustomEnvs # Registering envs
from stable_baselines3.common.env_util import make_vec_env



if __name__ == "__main__":
    
    env = gymnasium.make('Ant-v4', render_mode="human")
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        env.step(action)
        env.render()