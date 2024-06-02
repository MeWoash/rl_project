# autopep8: off
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from pathlib import Path
import matplotlib.ticker as ticker
import numpy as np

sys.path.append(str(Path(__file__,'..','..').resolve()))
from ModelTools.Utils import *
from PostProcessing.Utils import timeit, generate_fig_file, time_formatter, exclude_outliers
from PostProcessing.PlotGenerators import PlotBestActions, PlotHeatMap, PlotBestTrajectory, PlotBestTrainingReward, PlotWrapper
from PostProcessing.VideoGenerators import VideoGenerator
import PathsConfig as paths_cfg
# autopep8: on

@timeit
def generate_model_media(log_dir: str):
    media_dir = Path(log_dir, "media")
    media_dir.mkdir(exist_ok=True)
    
    df_episodes_all, df_episode_stats, df_training_stats = load_generate_csvs(log_dir)
    
    
    # # BEST ACTIONS
    # fig, ax = plt.subplots(1, 1, figsize=(10,7))
    # PlotBestActions(ax, n_best=1, legend=True).plot(df_episodes_all)
    # generate_fig_file(fig, media_dir, "best_actions")
    
    # #HEATMAP
    # fig, ax = plt.subplots(1, 1, figsize=(10,7))
    # PlotHeatMap(ax, sigma=2, bins=51).plot(df_episodes_all)
    # generate_fig_file(fig, media_dir, "heat_map")
    
    # #PlotBestTrajectory
    # fig, ax = plt.subplots(1, 1, figsize=(10,7))
    # PlotBestTrajectory(ax, n_best=5, legend=True).plot(df_episodes_all)
    # generate_fig_file(fig, media_dir, "best_trajectories")
    
    # #PlotBestRewardCurve
    # fig, ax = plt.subplots(1, 1, figsize=(10,7))
    # PlotBestTrainingReward(df_training_stats, ax=ax, relative=True).plot(df_episodes_all)
    # generate_fig_file(fig, media_dir, "best_rewards")
    
    # ALL IN ONE
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    wrapper = PlotWrapper([PlotHeatMap(sigma=2, bins=51),
                           PlotBestTrainingReward(df_training_stats, relative=True, n_best=5),
                           PlotBestTrajectory(n_best=5, legend=True),
                           PlotBestActions(n_best=1, legend=True)],
                          fig, axs)
    wrapper.plot(df_episodes_all)
    generate_fig_file(fig, media_dir, "mixed_stats")
    
    # VIDEO GENERATION
    # fig, axs = plt.subplots(2, 2, figsize=(15,10))
    # wrapper = PlotWrapper([PlotHeatMap(sigma=2, bins=51),
    #                        PlotBestTrainingReward(df_training_stats, relative=True),
    #                        PlotBestTrajectory(n_best=1, legend=True),
    #                        PlotBestActions(n_best=1, legend=True)],
    #                       fig, axs)
    # vidGen:VideoGenerator = VideoGenerator(wrapper, media_dir, frame_size=(1920, 1080), dpi=100)
    # vidGen.generate_video(df_episodes_all, "trajectories.mp4", "Episodes trajectories")

def generate_all_model_media(path_dir=paths_cfg.OUT_LEARNING_DIR):
    dirs =  [str(Path(file,"..")) for file in get_all_files(path_dir, "episodes_all.csv")]
    
    for dir in dirs:
        generate_model_media(dir)

@timeit
def generate_models_comparison():
    all_dfs = load_generate_all_csvs()
    
    fig, axs = plt.subplots(2, 1, figsize=(10,7))
    
    for dir, (df_episodes_all, df_episodes_summary, df_training_stats) in all_dfs.items():
        name = Path(dir).resolve().stem

        y = df_training_stats['episode_mean_reward'].to_numpy()
        y_smoothed = gaussian_filter1d(y, sigma=500)
        
        
        x_rel = df_training_stats['rel_time'].to_numpy()
        line_rel = axs[0].plot(x_rel, y, alpha = 0.3)[0]
        axs[0].plot(x_rel, y_smoothed, label=name, color=line_rel.get_color(), alpha = 1)
       
        x_steps = df_training_stats['learning_step'].to_numpy()
        line_steps = axs[1].plot(x_steps, y, alpha = 0.3)[0]
        axs[1].plot(x_steps, y_smoothed, label=name, color = line_steps.get_color(), alpha = 1)
    
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('mean reward')
    axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    
    axs[1].set_xlabel('learning steps')
    axs[1].set_ylabel('mean reward')
    axs[1].grid(True)
    axs[1].legend()
    
    fig.savefig(Path(paths_cfg.OUT_LEARNING_DIR,"models_comparison.png"))

@timeit
def generate_models_comparison_2(lower_percentile=1, upper_percentile=99):
    all_dfs = load_generate_all_csvs()
    
    fig, axs = plt.subplots(2, 1, figsize=(10,7))
    
    for dir, (df_episodes_all, df_episodes_summary, df_training_stats) in all_dfs.items():
        name = Path(dir).resolve().stem

        y = df_episodes_summary['episode_mean_reward'].to_numpy()
        x_rel = df_episodes_summary['rel_time'].to_numpy()
        x_steps = df_episodes_summary['learning_step_max'].to_numpy()
        
        mask = (y >= np.percentile(y, lower_percentile)) & (y <= np.percentile(y, upper_percentile))
        y_filtered = y[mask]
        x_rel_filtered = x_rel[mask]
        x_steps_filtered = x_steps[mask]
        
        y_filtered_smoothed = gaussian_filter1d(y_filtered, sigma=500)
        
        line_rel = axs[0].plot(x_rel_filtered, y_filtered, alpha = 0.3)[0]
        axs[0].plot(x_rel_filtered, y_filtered_smoothed, label=name, color=line_rel.get_color(), alpha = 1)
       
        line_steps = axs[1].plot(x_steps_filtered, y_filtered, alpha = 0.3)[0]
        axs[1].plot(x_steps_filtered, y_filtered_smoothed, label=name, color = line_steps.get_color(), alpha = 1)
    
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('mean reward')
    axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    
    axs[1].set_xlabel('learning steps')
    axs[1].set_ylabel('mean reward')
    axs[1].grid(True)
    axs[1].legend()
    
    fig.savefig(Path(paths_cfg.OUT_LEARNING_DIR,"models_comparison.png"))


@timeit
def generate_models_comparison_3():
    all_dfs = load_generate_all_csvs()
    
    fig, axs = plt.subplots(2, 1, figsize=(10,7))
    
    for dir, (df_episodes_all, df_episodes_summary, df_training_stats) in all_dfs.items():
        name = Path(dir).resolve().stem

        y = df_episodes_summary['episode_mean_reward'].to_numpy()
        y_smoothed = gaussian_filter1d(y, sigma=500)
        
        
        x_rel = df_episodes_summary['rel_time'].to_numpy()
        # line_rel = axs[0].plot(x_rel, y, alpha = 0.3)[0]
        axs[0].plot(x_rel, y_smoothed, label=name)#, color=line_rel.get_color())
       
        x_steps = df_episodes_summary['learning_step_max'].to_numpy()
        # line_steps = axs[1].plot(x_steps, y, alpha = 0.3)[0]
        axs[1].plot(x_steps, y_smoothed, label=name)#, color = line_steps.get_color())
    
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('mean reward')
    axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    
    axs[1].set_xlabel('learning steps')
    axs[1].set_ylabel('mean reward')
    axs[1].grid(True)
    axs[1].legend()
    
    fig.savefig(Path(paths_cfg.OUT_LEARNING_DIR,"models_comparison.png"))
    
if __name__ == "__main__":
    last_modified = str(Path(get_last_modified_file(paths_cfg.OUT_LEARNING_DIR,'.csv'),'..').resolve())
    generate_model_media(last_modified)
    
    generate_models_comparison()