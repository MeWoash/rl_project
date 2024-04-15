from cv2 import log
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from pathlib import Path

from tensorboard import summary

EPISODES_FILE_NAME = 'episodes_all.csv'
SUMMARY_FILE_NAME = 'episodes_summary.csv'

def load_dfs(log_dir:str):
    log_dir:Path = Path(log_dir).resolve()
    
    summary_path = log_dir.joinpath(SUMMARY_FILE_NAME)
    episodes_path = log_dir.joinpath(EPISODES_FILE_NAME)
    
    df_summary = pd.read_csv(summary_path, index_col=['episode','env'])
    df_all = pd.read_csv(episodes_path, index_col=['episode','env'])
    
    return df_summary, df_all

def plot_trajectory(df):
    fig, axs = plt.subplots(1,1)
    
    for indexes, grouped in df.groupby(level=df.index.names):
        axs.plot(grouped['pos_Y'].to_numpy(), grouped['pos_X'].to_numpy(), label=f"ep-{indexes[0]}_en-{indexes[1]}")


    axs.grid(True)
    axs.set_xlim([-10,10])
    axs.set_ylim([-10,10])
    
    axs.invert_xaxis()
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    return fig, axs

def create_heatmap(df, bins = 50, sigma= 3):
    
    fig, axs = plt.subplots(1,1)
    
    heatmap, xedges, yedges = np.histogram2d(df['pos_Y'], df['pos_X'], bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    axs.set_xlim([-10,10])
    axs.set_ylim([-10,10])
    axs.invert_xaxis()
    
    axs.imshow(heatmap.T, extent=extent, origin='lower')
    
    return fig, axs

def get_rows_by_ep_env(df, indexes:list[list[int, int]] = []):
    return df.loc[indexes]

def get_n_best_rewards(df, n_episodes=10):
    grouped = df.groupby(by=df.index.names)
    acc = grouped.sum().sort_values(by='reward', ascending=False)
    indexes = acc[:n_episodes].index.to_list()
    best = df.loc[indexes]
    return best

if __name__ == "__main__":
    log_dir = rf"D:\kody\rl_project\out\logs\A2C\A2C_1"
    df_summary, df_episodes = load_dfs(log_dir)
    
    
