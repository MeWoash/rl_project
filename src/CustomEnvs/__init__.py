from gymnasium.envs.registration import register
from CustomEnvs.CarParking import CarParkingEnv

register(
    id="CustomEnvs/CarParkingEnv-v0",
    entry_point="CustomEnvs:CarParkingEnv",
    max_episode_steps=10000,
)
