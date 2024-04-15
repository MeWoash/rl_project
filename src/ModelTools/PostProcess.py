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

def do_basic_analysis(log_dir):
    
    df_summary, df_episodes = load_dfs(log_dir)
    generate_heatmap_video(df_episodes, log_dir)
    
    fig_rew, axs_rew = generate_best_rewards_plot(df_episodes, 20)
    fig_rew.savefig(str(Path(log_dir,'best_rewards.png')))

if __name__ == "__main__":
    log_dir = rf"D:\kody\rl_project\out\logs\A2C\A2C_2"
    
    do_basic_analysis(log_dir)
    
    # df_summary, df_episodes = load_dfs(log_dir)
    
    
    
