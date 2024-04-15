from math import ceil
import sys
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
from ModelTools.Utils import *
# autopep8: on

# import MJCFGenerator


def generate_trajectory_plot(df, axs:Axes = None):
    
    if axs is None:
        fig, axs = plt.subplots(1,1)
    else:
        fig = axs.get_figure()
    
    for indexes, grouped in df.groupby(level=df.index.names):
        axs.plot(grouped['pos_X'].to_numpy(), grouped['pos_Y'].to_numpy(), label=f"ep-{indexes[0]}_en-{indexes[1]}")

    axs.grid(True)
    axs.set_xlim(MAP_BOUNDARY[0])
    axs.set_ylim(MAP_BOUNDARY[1])
    
    axs.set_xlabel('y')
    axs.set_ylabel('x')
    axs.set_aspect('equal')
    
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    return fig, axs

def generate_heatmap_plot(df, sigma=1, bins=101, axs:Axes = None):
    
    if axs is None:
        fig, axs = plt.subplots(1,1)
    else:
        fig = axs.get_figure()
    
    heatmap, xedges, yedges = np.histogram2d(df['pos_X'], df['pos_Y'], bins=bins, range=MAP_BOUNDARY)
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    axs.set_xlim(MAP_BOUNDARY[0])
    axs.set_ylim(MAP_BOUNDARY[1])
    
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_aspect('equal')
    
    axs.imshow(heatmap.T, extent=extent, origin='lower',interpolation='nearest', cmap = matplotlib.colormaps['plasma'])
    
    return fig, axs


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