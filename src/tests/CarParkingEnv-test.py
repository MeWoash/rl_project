import pygame
import numpy as np

import sys
import os

SRC_ROOT_DIR = os.path.normpath(os.path.join(__file__,'..','..',))
sys.path.append(SRC_ROOT_DIR)

import CustomEnvs
from CustomEnvs.CarParking import CarParkingEnv

if __name__ == "__main__":
    env = CarParkingEnv(render_mode="human")
