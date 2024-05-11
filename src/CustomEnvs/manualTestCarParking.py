# autopep8: off
from pathlib import Path
import sys
from pynput import keyboard
import numpy as np
import gymnasium
from gymnasium.spaces import Box
import math

sys.path.append(str(Path(__file__,'..','..').resolve()))
from MJCFGenerator.Generator import generate_MJCF 
from CustomEnvs.CarParking import *
from MJCFGenerator.Config import *
# autopep8: on

current_action = np.array([0.0, 0.0])

WHEEL_ANGLE_LIMIT_RADIANS = [math.radians(WHEEL_ANGLE_LIMIT[0]), math.radians(WHEEL_ANGLE_LIMIT[1])]

def on_press(key):
    global current_action
    try:
        if key == keyboard.Key.up:
            current_action[ACTION_INDEX.ENGINE] = 1.0
        elif key == keyboard.Key.down:
            current_action[ACTION_INDEX.ENGINE] = -1.0
            
        if key == keyboard.Key.left:
            current_action[ACTION_INDEX.ANGLE] = WHEEL_ANGLE_LIMIT_RADIANS[0]
        elif key == keyboard.Key.right:
            current_action[ACTION_INDEX.ANGLE] = WHEEL_ANGLE_LIMIT_RADIANS[1]
            
    except BaseException as e:
        pass
        # print(e)


def on_release(key):
    global current_action
    try:
        if key in [keyboard.Key.up, keyboard.Key.down]:
            current_action[0] = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            current_action[1] = 0
        else:
            pass
    except BaseException as e:
        print(e)


# Start listening to the keyboard
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


def describe_obs(obs):
    d = f"""
speed:      {obs[OBS_INDEX.VELOCITY_BEGIN:OBS_INDEX.VELOCITY_END+1]}    
dist:       {obs[OBS_INDEX.DISTANCE_BEGIN:OBS_INDEX.DISTANCE_END+1]}
adiff:      {obs[OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1]}
contact:    {obs[OBS_INDEX.CONTACT_BEGIN:OBS_INDEX.CONTACT_END+1]}
range:      {obs[OBS_INDEX.RANGE_BEGIN:OBS_INDEX.RANGE_END+1]}
pos:        {obs[OBS_INDEX.REL_POS_BEGIN:OBS_INDEX.REL_POS_END+1]}
eul:        {obs[OBS_INDEX.YAW_BEGIN:OBS_INDEX.YAW_END+1]}"""

    return d


if __name__ == "__main__":
    generate_MJCF()
    env = CarParkingEnv(render_mode="human")
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    env.reset(0)
    while True:
        observation, reward, terminated, truncated, info = env.step(
            current_action)
        # print(describe_obs(observation))
        if terminated or truncated:
            env.reset()

