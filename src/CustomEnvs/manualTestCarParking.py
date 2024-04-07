from unittest.mock import Base
from pynput import keyboard
import numpy as np
import gymnasium
from gymnasium.spaces import Box
import math
from CarParking import CarParkingEnv, ObsIndex


current_action = np.array([0.0, 0.0])


def on_press(key):
    global current_action
    try:
        if key == keyboard.Key.up:
            current_action[0] = 1.0
        elif key == keyboard.Key.down:
            current_action[0] = -1.0
        if key == keyboard.Key.left:
            current_action[1] = -1
        elif key == keyboard.Key.right:
            current_action[1] = 1
    except BaseException as e:
        print(e)


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
speed:      {obs[ObsIndex.VELOCITY_BEGIN:ObsIndex.VELOCITY_END+1]}    
dist:       {obs[ObsIndex.DISTANCE_BEGIN:ObsIndex.DISTANCE_END+1]}
adiff:      {obs[ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1]}
contact:    {obs[ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1]}
range:      {obs[ObsIndex.RANGE_BEGIN:ObsIndex.RANGE_END+1]}
pos:        {obs[ObsIndex.POS_BEGIN:ObsIndex.POS_END+1]}
eul:        {obs[ObsIndex.EUL_BEGIN:ObsIndex.EUL_END+1]}"""

    return d


if __name__ == "__main__":
    env = CarParkingEnv(render_mode="human")
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    while True:
        observation, reward, terminated, truncated, info = env.step(
            current_action)
        env.render()
        if terminated or truncated:
            env.reset()
