from math import ceil
import sys
import cv2
import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from pathlib import Path


sys.path.append(str(Path(__file__,'..','..').resolve()))
from CustomEnvs.Indexes import *
from CustomEnvs.CarParking import *
from MJCFGenerator.Config import *

from ModelTools.Utils import *
from PostProcessing.Utils import timeit
from PostProcessing.PlotGenerators import *
from PostProcessing.VideoGenerators import *
from PathsConfig import *
# autopep8: on



def evaluate_reward_function(reward_function, reward_args, **kwargs):
    reward, reward_info = reward_function(**reward_args)
    return reward
    
def prepare_data_for_reward_function():
    reward_params = {
            "dist_weight": 0.25,
            "angle_diff_weight": 0.75,
            "dist_scale":2,
            "max_step_reward": 1
        }
    parking_point = np.array(PARKING_SPOT_KWARGS['pos'][:2])
    
    x_values = np.linspace(MAP_BOUNDARY[0][0], MAP_BOUNDARY[0][1], 50)
    y_values = np.linspace(MAP_BOUNDARY[0][0], MAP_BOUNDARY[0][1], 50)
            
    mesh = np.meshgrid(x_values, y_values)
    XY = np.vstack([mesh[0].ravel(),mesh[1].ravel()]).T
    
    observation = np.zeros(OBS_INDEX.OBS_SIZE, dtype=np.float32)
    angles = np.linspace(0, len(observation[OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1])*math.pi, 4)
    
    
    for spawn_kwargs in CAR_SPAWN_KWARGS:
        spawn_point = np.array(spawn_kwargs['pos'][:2])
        init_car_distance = np.linalg.norm(spawn_point - parking_point)
        for angle in angles:
            rewards = []
            for xy in XY:
                car_dist_to_parking = np.linalg.norm(xy - parking_point)
                
                observation[OBS_INDEX.DISTANCE_BEGIN:OBS_INDEX.DISTANCE_END+1] = car_dist_to_parking
                observation[OBS_INDEX.ANGLE_DIFF_BEGIN:OBS_INDEX.ANGLE_DIFF_END+1] = angle/2, angle/2
                
                reward, reward_info = calculate_reward(observation, init_car_distance, reward_params)
                # print(reward)
                rewards.append(reward)
            rewards = np.array(rewards).reshape(mesh[0].shape)
            
            plt.figure(figsize=(10, 8))
            plt.pcolormesh(mesh[0], mesh[1], rewards, shading='auto')
            plt.colorbar()
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title(f'angle diff = {angle}')
            plt.show()

    

if __name__ == "__main__":
    prepare_data_for_reward_function()