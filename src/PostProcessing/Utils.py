# autopep8: off

from math import ceil
import os
from pathlib import Path
import sys
import time

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__,'..','..').resolve()))
import MJCFGenerator.Config as mjcf_cfg


# autopep8: on
MAP_BOUNDARY = [[-mjcf_cfg.MAP_LENGTH
[0]/2, mjcf_cfg.MAP_LENGTH
[0]/2],[-mjcf_cfg.MAP_LENGTH
[1]/2, mjcf_cfg.MAP_LENGTH
[1]/2]] # X, Y


def get_last_modified_file(directory_path, suffix=".zip"):
    latest_time = 0
    latest_file = None

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(suffix):
                filepath = os.path.join(root, filename)
                if os.path.isfile(filepath):
                    file_mtime = os.path.getmtime(filepath)
                    
                    if file_mtime > latest_time:
                        latest_time = file_mtime
                        latest_file = filepath

    if latest_file:
        print(f"Last modified {suffix} file: {latest_file}")
    else:
        print(f"No {suffix} files found.")
    return latest_file

def timeit(func):
    """Measure function time"""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function {func.__name__} finished in: {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

def generator_episodes_by_learning_step(df, df_training, episode_divide_factor=100):
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
    for indexes, grouped in df.groupby(level=['episode','env']):
        yield indexes, grouped
        
def group_by_envs(df):
    for index, grouped in df.groupby(level='env'):
        yield index, grouped
        
def get_n_best_rewards(df, n_episodes=10):
    grouped = df.groupby(by=['episode','env'])
    acc = grouped.last().sort_values(by='episode_mean_reward', ascending=False)
    indexes = acc[:n_episodes].index.to_list()
    best = df.loc[indexes]
    return best

def generate_fig_file(fig:Figure, dir:str, filename:str="out.png") -> None:
    p = str(Path(dir,filename))
    fig.savefig(p)

def time_formatter(x, pos):
    if x == 0:
        return "0s"
    
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    seconds = int(x % 60)
    
    out = f""
    if hours > 0:
        out+= f"{hours}h"
        if minutes > 0:
            out+= f" {minutes}m"
    elif minutes > 0:
        out+= f" {minutes}m"
        if seconds > 0:
            out+= f" {seconds}s"
    else:
        out+= f" {seconds}s"
    
    return out

@timeit
def divide_by_steps(df_all, N=5, min_step = None, max_step = None):
    df = df_all.copy()
    
    _min = df['learning_step'].min()
    _max =df['learning_step'].max()
    
    if min_step is None or min_step < _min:
        min_step = _min
    if max_step is None or max_step > _max:
        max_step = _max
        
    range_step = max_step - min_step

    bins = [min_step + i * (range_step / N) for i in range(N)] + [max_step]
    df['part'] = pd.cut(df['learning_step'], bins=bins, labels=False, include_lowest=True)
    
    parts = [df[df['part'] == i].copy().drop(columns=['part']) for i in range(N)]

    return parts, bins