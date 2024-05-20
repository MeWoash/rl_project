from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC, PPO

TRAIN_VECTOR = [
    {
        "name": "next_test",
        "modelConstructor" : PPO,
        "total_timesteps" : 20_000,
        "seed": 0,
        "normalize": False,
        "model_kwargs" : {
            "device":"cuda"
        },
        "make_env_kwargs" : {
            "vec_env_cls": SubprocVecEnv,
            "n_envs": 1,
            "env_kwargs":{
                "render_mode": "none",
                "enable_random_spawn": True,
                "enable_spawn_noise": True
            }
        }
    }
]