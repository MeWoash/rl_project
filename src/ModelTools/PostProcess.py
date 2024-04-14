import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm


def load_episodes(df_summary):
    df = pd.DataFrame()
    for file_name in df_summary['file']:
        new_df = pd.read_csv(file_name)
        df = pd.concat([df,new_df])
    return df


if __name__ == "__main__":
    summary_path = rf"D:\kody\rl_project\out\logs\A2C\A2C_1\episodes_summary.csv"
    df_summary = pd.read_csv(summary_path)
    
    df_all = load_episodes(df_summary)
    
    # plt.scatter(df_all['pos_Y'], df_all['pos_X'], s=40, alpha=0.1)
    # plt.scatter(5, 5, s=200, c='r')
    
    # plt.grid()
    # plt.xlim([-10,10])
    # plt.ylim([-10,10])
    # plt.gca().invert_xaxis()
    # plt.show()
    
    
    heatmap, xedges, yedges = np.histogram2d(df_all['pos_Y'], df_all['pos_X'], bins=100)
    heatmap = gaussian_filter(heatmap, sigma=3)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.gca().invert_xaxis()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.scatter(5, 5, s=200, c='r')
    plt.show()