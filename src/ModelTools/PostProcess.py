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


def do_basic_analysis(log_dir):
    df_summary, df_episodes = load_dfs(log_dir)
    
    # generate_heatmap_video(df_episodes, log_dir, "heatmap.mp4", sigma=1, bins=101)
    # generate_trajectory_video(df_episodes, log_dir, "trajectories.mp4")
    
    _, filltered = list(generator_episodes(df_episodes, 10))[0]
    
    vidGen:VideoGenerator = VideoGenerator(PlotWrapper([PlotHeatMap(), PlotTrajectory()]), log_dir, frame_size=(1080,1080), dpi=200)
    
    vidGen.generate_video(df_episodes, "trajectories.mp4", "Episodes trajectories")
    vidGen.generate_video(df_episodes, "best_trajectories.mp4", "Best runs Episodes trajectories", generator_function=generator_episodes_best)
    
    best = get_n_best_rewards(df_episodes, 20)
    fig, axs = plt.subplots(1, 2, figsize=(10,7))
    _,_ = PlotWrapper([PlotTrajectory(), PlotHeatMap()], fig, axs).plot(best)
    
    # plt.show()

@timeit
def do_basic_analysis_timed(log_dir):
    do_basic_analysis(log_dir)

if __name__ == "__main__":
    log_dir = rf"D:\kody\rl_project\out\logs\A2C\A2C_1"
    do_basic_analysis_timed(log_dir)