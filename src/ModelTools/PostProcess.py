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


def do_basic_analysis(log_dir: str):
    media_dir = Path(log_dir, "media")
    media_dir.mkdir(exist_ok=True)
    
    df_summary, df_episodes = load_dfs(log_dir)
    
    _, filltered = list(generator_episodes(df_episodes, 10))[0]
    
    
    # VIDEO GENERATION
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    wrapper = PlotWrapper([PlotHeatMap(sigma=2, bins=51), PlotTrajectory(), PlotBestTrajectory(n_best=1), PlotBestActions(n_best=1, legend=True)], fig, axs)
    vidGen:VideoGenerator = VideoGenerator(wrapper, media_dir, frame_size=(1080, 1080), dpi=100)
    vidGen.generate_video(df_episodes, "trajectories.mp4", "Episodes trajectories")
    
    # BEST ACTIONS
    fig, ax = PlotWrapper([PlotBestActions(n_best=1, legend=True)]).plot(df_episodes)
    generate_fig_file(fig, media_dir, "best_actions")
    
    #HEATMAP
    fig, ax = PlotWrapper([PlotHeatMap(sigma=2, bins=51)]).plot(df_episodes)
    generate_fig_file(fig, media_dir, "heat_map")
    
    #PlotBestTrajectory
    fig, ax = PlotWrapper([PlotBestTrajectory(n_best=5)]).plot(df_episodes)
    generate_fig_file(fig, media_dir, "best_trajectories")
    
    # ALL IN ONE
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    wrapper = PlotWrapper([PlotHeatMap(sigma=2, bins=51), PlotTrajectory(), PlotBestTrajectory(n_best=1), PlotBestActions(n_best=1, legend=True)], fig, axs)
    wrapper.plot(df_episodes)
    generate_fig_file(fig, media_dir, "mixed_stats")
    
@timeit
def do_basic_analysis_timed(log_dir):
    do_basic_analysis(log_dir)

if __name__ == "__main__":
    log_dir = rf"out\learning\SAC\SAC_2"
    do_basic_analysis_timed(log_dir)