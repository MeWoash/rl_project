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


def generate_trajectory_plot(df, axs:Axes = None, legend = False, **kwargs):
    
    if axs is None:
        fig, axs = plt.subplots(1,1)
    else:
        fig = axs.get_figure()
    
    for indexes, grouped in group_by_episodes(df):
        x = grouped['pos_X'].to_numpy()
        y = grouped['pos_Y'].to_numpy()
        axs.plot(x, y, label=f"ep-{indexes[0]}_en-{indexes[1]}")

    axs.grid(True)
    axs.set_xlim(MAP_BOUNDARY[0])
    axs.set_ylim(MAP_BOUNDARY[1])
    
    axs.set_xlabel('y')
    axs.set_ylabel('x')
    axs.set_aspect('equal')
    
    if legend:
        axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    return fig, axs

def generate_heatmap_plot(df, sigma=1, bins=101, axs:Axes = None, **kwargs):
    
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


