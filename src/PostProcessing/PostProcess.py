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


def generate_model_media(log_dir: str):
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


def generate_models_comparison():
    files = get_all_files(OUT_LEARNING_DIR, "training_stats.csv")
    
    
    fig, axs = plt.subplots(2, 1, figsize=(10,7))
    
    for key, value in files.items():
        df_training_stats = pd.read_csv(value)
        name = Path(value, "..").resolve().stem

        y = df_training_stats['episode_mean_reward'].to_numpy()
        x_rel = df_training_stats['rel_time'].to_numpy()
        axs[0].plot(x_rel, y, label=name)
       
        x_steps = df_training_stats['learning_step'].to_numpy()
        axs[1].plot(x_steps, y, label=name)
    
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('mean reward')
    axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    
    axs[1].set_xlabel('learning steps')
    axs[1].set_ylabel('mean reward')
    axs[1].grid(True)
    axs[1].legend()
    
    plt.savefig(Path(OUT_LEARNING_DIR,"models_comparison.png"))

@timeit
def generate_model_media_timed(log_dir):
    generate_model_media(log_dir)
    
@timeit
def generate_models_comparison_timed():
    generate_models_comparison()


def main_generate_model_media():
    # log_dir = str(Path(OUT_LEARNING_DIR,'SAC','SAC_1'))
    last_modified = str(Path(get_last_modified_file(OUT_LEARNING_DIR,'.csv'),'..').resolve())
    generate_model_media_timed(last_modified)
    
def main_compare_models():
    generate_models_comparison_timed()

if __name__ == "__main__":
    main_generate_model_media()
    main_compare_models()