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
    df_summary, df_episodes = load_dfs(log_dir)
    media_dir = Path(log_dir, "media")
    
    # _, filltered = list(generator_episodes(df_episodes, 10))[0]
    
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    wrapper = PlotWrapper([PlotHeatMap(), PlotTrajectory(), PlotBestTrajectory()], fig, axs)
    
    vidGen:VideoGenerator = VideoGenerator(wrapper, media_dir, frame_size=(720, 720), dpi=100)
    vidGen.generate_video(df_episodes, "trajectories.mp4", "Episodes trajectories")
    

@timeit
def do_basic_analysis_timed(log_dir):
    do_basic_analysis(log_dir)

if __name__ == "__main__":
    log_dir = rf"D:\kody\rl_project\out\logs\A2C\A2C_2"
    do_basic_analysis_timed(log_dir)