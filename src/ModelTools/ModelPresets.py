import stable_baselines3
from stable_baselines3 import A2C, SAC, TD3
from ModelTools.ModelWrapper import ModelWrapper

A2C_PRESET = {
    "createEnvArgs":
        {
            "envCreationFunciton": ModelWrapper.create_vec_env,
            "envArgs":
            {
                "n_envs": 4,
                "render_mode": "human"
            }

        },
    "createModelArgs":
        {
            "modelContructor": A2C,
            "modelArgs":
                {
                    "policy": "MlpPolicy",
                    "device": "cuda"
                },
        },
    "learnModelArgs":
        {
            "n_steps": 1_000_000
        }
}

SAC_PRESET = {
    "createEnvArgs":
        {
            "envCreationFunciton": ModelWrapper.create_vec_env,
            "envArgs":
            {
                "n_envs": 16,
                "render_mode": "human"
            }

        },
    "createModelArgs":
        {
            "modelContructor": SAC,
            "modelArgs":
                {
                    "policy": stable_baselines3.sac.policies.MlpPolicy,
                    "device": "cuda"
                },
        },
    "learnModelArgs":
        {
            "n_steps": 250_000
        }
}

TD3_PRESET = {
    "createEnvArgs":
        {
            "envCreationFunciton": ModelWrapper.create_vec_env,
            "envArgs":
            {
                "n_envs": 16,
                "render_mode": "rgb_array"
            }

        },
    "createModelArgs":
        {
            "modelContructor": TD3,
            "modelArgs":
                {
                    "policy": stable_baselines3.td3.policies.MlpPolicy,
                    "device": "cuda"
                },
        },
    "learnModelArgs":
        {
            "n_steps": 200_000
        }
}
