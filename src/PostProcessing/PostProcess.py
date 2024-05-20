# autopep8: off
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
from ModelTools.Utils import *
from PostProcessing.Utils import timeit
from PostProcessing.PlotGenerators import *
from PostProcessing.VideoGenerators import *
from PathsConfig import *
# autopep8: on


def generate_media(log_dir: str):
    media_dir = Path(log_dir, "media")
    media_dir.mkdir(exist_ok=True)
    
    df_episodes_all, df_episode_stats, df_training_stats = load_generate_csvs(log_dir)
    
    
    # # BEST ACTIONS
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    PlotBestActions(ax, n_best=1, legend=True).plot(df_episodes_all)
    generate_fig_file(fig, media_dir, "best_actions")
    
    # #HEATMAP
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    PlotHeatMap(ax, sigma=2, bins=51).plot(df_episodes_all)
    generate_fig_file(fig, media_dir, "heat_map")
    
    # #PlotBestTrajectory
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    PlotBestTrajectory(ax, n_best=5, legend=True).plot(df_episodes_all)
    generate_fig_file(fig, media_dir, "best_trajectories")
    
    #PlotBestRewardCurve
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    PlotBestTrainingReward(df_training_stats, ax=ax, relative=True).plot(df_episodes_all)
    generate_fig_file(fig, media_dir, "best_rewards")
    
    # ALL IN ONE
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    wrapper = PlotWrapper([PlotHeatMap(sigma=2, bins=51),
                           PlotBestTrainingReward(df_training_stats, relative=True),
                           PlotBestTrajectory(n_best=1, legend=True),
                           PlotBestActions(n_best=1, legend=True)],
                          fig, axs)
    wrapper.plot(df_episodes_all)
    generate_fig_file(fig, media_dir, "mixed_stats")
    
    # VIDEO GENERATION
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    wrapper = PlotWrapper([PlotHeatMap(sigma=2, bins=51),
                           PlotBestTrainingReward(df_training_stats, relative=True),
                           PlotBestTrajectory(n_best=1, legend=True),
                           PlotBestActions(n_best=1, legend=True)],
                          fig, axs)
    vidGen:VideoGenerator = VideoGenerator(wrapper, media_dir, frame_size=(1920, 1080), dpi=100)
    vidGen.generate_video(df_episodes_all, "trajectories.mp4", "Episodes trajectories")
    
@timeit
def generate_media_timed(log_dir):
    generate_media(log_dir)


def main():
    # log_dir = str(Path(OUT_LEARNING_DIR,'SAC','SAC_1'))
    last_modified = str(Path(get_last_modified_file(OUT_LEARNING_DIR,'.csv'),'..').resolve())
    generate_media_timed(last_modified)

if __name__ == "__main__":
    main()