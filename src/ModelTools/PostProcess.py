import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from pathlib import Path


def load_episodes(df_summary, save_to_file = True):
    
    p = Path(df_summary['file'][0]).joinpath("../../episodes_all.csv")
    df = pd.DataFrame()
    if p.exists():
        df = pd.read_csv(p)
    else:
        for index, row in df_summary.iterrows():
            new_df = pd.read_csv(row['file'], index_col=False)
            new_df['episode']=row['episode']
            new_df['env']=row['env']
            df = pd.concat([df,new_df])
        df.reset_index(inplace=True, drop=True)
        df.to_csv(p)
    return df


if __name__ == "__main__":
    summary_path = rf"D:\kody\rl_project\out\logs\A2C\A2C_1\episodes_summary.csv"
    df_summary = pd.read_csv(summary_path)
    
    df_all = load_episodes(df_summary)
    
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