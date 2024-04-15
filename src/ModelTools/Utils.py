from math import ceil
from pathlib import Path
import time
from typing import Callable

import cv2
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

EPISODES_FILE_NAME = 'episodes_all.csv'
SUMMARY_FILE_NAME = 'episodes_summary.csv'
MAP_SIZE = [20, 20, 20, 5]  # MJCFGenerator.Generator._map_length
MAP_BOUNDARY = [[-MAP_SIZE[0]/2, MAP_SIZE[0]/2],[-MAP_SIZE[1]/2, MAP_SIZE[1]/2]] # X, Y


def timeit(func):
    """Measure function time"""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function {func.__name__} finished in: {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

def load_dfs(log_dir:str):
    log_dir:Path = Path(log_dir).resolve()
    
    summary_path = log_dir.joinpath(SUMMARY_FILE_NAME)
    episodes_path = log_dir.joinpath(EPISODES_FILE_NAME)
    
    df_summary = pd.read_csv(summary_path, index_col=['episode','env'])
    df_all = pd.read_csv(episodes_path, index_col=['episode','env'])
    
    return df_summary, df_all

def batch_by_episodes(df, episode_divide_factor=100):
    max_ep = df.index.max()[0]
    data_sorted = df.sort_index()
    
    n_episodes = ceil(max_ep/episode_divide_factor)
    idxs = list(range(1, max_ep+n_episodes+1, n_episodes))
    if idxs[-1]>max_ep+1:
        idxs[-1]=max_ep+1

    for i in range(len(idxs)-1):
        lower_bound = idxs[i]
        upper_bound = idxs[i + 1]
        
        yield (lower_bound, upper_bound), data_sorted.loc[(slice(lower_bound, upper_bound), slice(None)), :]
        
def group_by_episodes(df):
    for indexes, grouped in df.groupby(level=df.index.names):
        yield indexes, grouped
        
def get_n_best_rewards(df, n_episodes=10):
    grouped = df.groupby(by=df.index.names)
    acc = grouped.sum().sort_values(by='reward', ascending=False)
    indexes = acc[:n_episodes].index.to_list()
    best = df.loc[indexes]
    return best

def generate_n_combined_plots(df, function_spec:list[Callable, dict], axs=None, **kwargs):
    if axs is None:
        fig, axs = plt.subplots(1, len(function_spec))
    else:
        assert len(axs) == len(function_spec)
        fig = axs.get_figure()
    
    for i, ax in enumerate(axs):
        function_spec[i][0](df, axs=ax,**function_spec[i][1])
        
    return fig, axs
        

def generate_video_from_plot_function(df, plot_function, dir, filename="out.mp4", plot_function_kwargs={}):
    dpi = 100
    frame_size = (480, 480)
    fig_size = (frame_size[0] / dpi, frame_size[1] / dpi)
    
    fig, axs = plt.subplots(1,1, figsize=fig_size, dpi=dpi)
    output_file = str(Path(dir).joinpath(filename))
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    arr = []
  
    for (lower_bound, upper_bound), filtered in batch_by_episodes(df, 100):
  
        plot_function(filtered, axs=axs, **plot_function_kwargs)
        axs.set_title(f"episode: <{lower_bound}, {upper_bound})")
        fig.canvas.draw()

        rgba_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)

        w, h = fig.canvas.get_width_height()
        rgba_image = rgba_image.reshape((h, w, 4))

        rgb_image = rgba_image[:, :, :3]
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        bgr_image = cv2.resize(bgr_image, frame_size)
        
        out.write(bgr_image)
        axs.cla()

    out.release()
    plt.close(fig)
    return arr

def generate_fig_file(fig:Figure, dir:str, filename:str="out.png") -> None:
    p = str(Path(dir,filename))
    fig.savefig(p)
