import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from pathlib import Path


if __name__ == "__main__":
    episode_summary_path = rf"D:\kody\rl_project\out\logs\A2C\A2C_2\episodes_summary.csv"
    episode_all_path = rf"D:\kody\rl_project\out\logs\A2C\A2C_2\episodes_all.csv"
    
    df_summary = pd.read_csv(episode_summary_path, index_col=False)
    df_all = pd.read_csv(episode_all_path, index_col=False)
    
    plt.scatter(df_all['pos_Y'], df_all['pos_X'], alpha=0.1)
    plt.scatter(5, 5, s=200, c='r')
    
    plt.grid()
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.gca().invert_xaxis()
    plt.show()
    
    
    heatmap, xedges, yedges = np.histogram2d(df_all['pos_Y'], df_all['pos_X'], bins=50)
    heatmap = gaussian_filter(heatmap, sigma=3)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.gca().invert_xaxis()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.scatter(5, 5, s=200, c='r')
    plt.show()