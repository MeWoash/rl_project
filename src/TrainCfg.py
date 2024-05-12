from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, SAC, PPO

TRAIN_VECTOR = [
    {
        "name": "test",
        "modelConstructor" : A2C,
        "total_timesteps" : 10_000,
        "seed": 0,
        "normalize": False,
        "model_kwargs" : {
            "device":"cpu"
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