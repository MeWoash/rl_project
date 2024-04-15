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

# import MJCFGenerator

def generate_best_rewards_plot(df_episodes, n_best = 20):
    n_best = 20
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    best = get_n_best_rewards(df_episodes, n_best)
    generate_trajectory_plot(best, axs = axs[0])
    generate_heatmap_plot(best, axs = axs[1], sigma=1, bins=101)

    axs[0].scatter(5,5, c='r', s=50)
    axs[1].scatter(5,5, c='r', s=50)
    if n_best > 20:
        axs[0].legend().remove()
    fig.tight_layout()
    fig.suptitle(f"{n_best} best rewarded episodes")
    
    return fig, axs


def do_basic_analysis(log_dir):
    df_summary, df_episodes = load_dfs(log_dir)
    
    generate_heatmap_video(df_episodes, log_dir, "heatmap.mp4", sigma=1, bins=101)
    generate_trajectory_video(df_episodes, log_dir, "trajectories.mp4")

if __name__ == "__main__":
    log_dir = rf"D:\kody\rl_project\out\logs\A2C\A2C_2"
    
    do_basic_analysis(log_dir)
    
    # df_summary, df_episodes = load_dfs(log_dir)
    
    
    
