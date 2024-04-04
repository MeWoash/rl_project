from ModelTools.ModelWrapper import ModelWrapper
from ModelTools.ModelPresets import *

PRESETS = [
    # A2C_PRESET,
    # SAC_PRESET,
    TD3_PRESET
]


if __name__ == "__main__":

    for preset in PRESETS:

        modelWrapper = ModelWrapper()
        env_create_f = preset['createEnvArgs']['envCreationFunciton']

        env_create_f(modelWrapper, **preset['createEnvArgs']['envArgs'])
        modelWrapper.create_model(**preset['createModelArgs'])

        modelWrapper.learn_model(**preset['learnModelArgs'])
        modelWrapper.save_model()
