# autopep8: off
from pathlib import Path
import sys
from pynput import keyboard
import numpy as np
import gymnasium
from gymnasium.spaces import Box
import math

sys.path.append(str(Path(__file__,'..','..').resolve()))
import MJCFGenerator.Config as mjcf_cfg
from MJCFGenerator.Generator import generate_MJCF 
from CustomEnvs.CarParking import CarParkingEnv
from CustomEnvs.Indexes import ACTION_INDEX
import MJCFGenerator.Config as mjcf_cfg
# autopep8: on

current_action = np.array([0.0, 0.0])

WHEEL_ANGLE_LIMIT_RADIANS = [math.radians(mjcf_cfg.WHEEL_ANGLE_LIMIT[0]), math.radians(mjcf_cfg.WHEEL_ANGLE_LIMIT[1])]

def on_press(key):
    global current_action
    try:
        if key == keyboard.Key.up:
            current_action[ACTION_INDEX.ENGINE] = 1.0
        elif key == keyboard.Key.down:
            current_action[ACTION_INDEX.ENGINE] = -1.0
            
        if key == keyboard.Key.left:
            current_action[ACTION_INDEX.ANGLE] = -1 # WHEEL_ANGLE_LIMIT_RADIANS[0]
        elif key == keyboard.Key.right:
            current_action[ACTION_INDEX.ANGLE] = 1 # WHEEL_ANGLE_LIMIT_RADIANS[1]
            
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


def main():
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

if __name__ == "__main__":
    main()

