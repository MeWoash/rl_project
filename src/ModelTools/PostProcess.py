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

# autopep8: off
sys.path.append(str(Path(__file__,'..','..').resolve()))
from ModelTools.Utils import timeit
from ModelTools.PlotGenerators import *
from ModelTools.VideoGenerators import *
# autopep8: on

class test:
    def __init__(self) -> None:
        fig, ax = plt.subplots(1, 1)
# import MJCFGenerator

def do_basic_analysis(log_dir):
    df_summary, df_episodes = load_dfs(log_dir)
    
    # generate_heatmap_video(df_episodes, log_dir, "heatmap.mp4", sigma=1, bins=101)
    # generate_trajectory_video(df_episodes, log_dir, "trajectories.mp4")
    
    
    _, filltered = list(batch_by_episodes(df_episodes, 10))[0]
    
    heat = PlotHeatMap()
    traj = PlotTrajectory()
    wrapper = PlotWrapper([2,1],[heat, traj])
    wrapper.plot(filltered)
    
    plt.show()

@timeit
def do_basic_analysis_timed(log_dir):
    do_basic_analysis(log_dir)

if __name__ == "__main__":
    log_dir = rf"D:\kody\rl_project\out\logs\A2C\A2C_2"
    do_basic_analysis_timed(log_dir)