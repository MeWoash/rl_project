from unittest.mock import Base
from pynput import keyboard
import numpy as np
import gymnasium
from gymnasium.spaces import Box
import math
import CarParking


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
speed:      {obs[0]}    
dist:       {obs[1]}
adiff:      {obs[2]}
contact:    {obs[3]}
range:      {obs[4:11]}
pos:        {obs[12:15]}
eul:        {obs[16:]}"""

    return d


if __name__ == "__main__":
    env = CarParking.CarParkingEnv(render_mode="human")
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    i = 0
    while True:
        observation, reward, terminated, truncated, info = env.step(
            current_action)
        if i % 200 == 0:
            described = describe_obs(observation)
            print(described)
            print(f"reward: {reward}")
        env.render()
        if terminated or truncated:
            env.reset()
        i += 1
