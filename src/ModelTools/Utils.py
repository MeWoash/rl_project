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

def generator_episodes(df, episode_divide_factor=100):
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
        
def generator_episodes_best(df, best = 1):
    data_sorted = df.sort_index()
    for (lower_bound, upper_bound), batch in generator_episodes(data_sorted):
        yield (lower_bound, upper_bound), get_n_best_rewards(batch, best)
        
def group_by_episodes(df):
    for indexes, grouped in df.groupby(level=df.index.names):
        yield indexes, grouped
        
def get_n_best_rewards(df, n_episodes=10):
    grouped = df.groupby(by=df.index.names)
    acc = grouped.sum().sort_values(by='reward', ascending=False)
    indexes = acc[:n_episodes].index.to_list()
    best = df.loc[indexes]
    return best

def generate_fig_file(fig:Figure, dir:str, filename:str="out.png") -> None:
    p = str(Path(dir,filename))
    fig.savefig(p)