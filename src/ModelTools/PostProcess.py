from cv2 import log
import cv2
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from pathlib import Path

# import MJCFGenerator

EPISODES_FILE_NAME = 'episodes_all.csv'
SUMMARY_FILE_NAME = 'episodes_summary.csv'
MAP_SIZE = [20, 20, 20, 5]  # MJCFGenerator.Generator._map_length
MAP_BOUNDARY = [[-MAP_SIZE[0]/2, MAP_SIZE[0]/2],[-MAP_SIZE[1]/2, MAP_SIZE[1]/2]] # X, Y

def load_dfs(log_dir:str):
    log_dir:Path = Path(log_dir).resolve()
    
    summary_path = log_dir.joinpath(SUMMARY_FILE_NAME)
    episodes_path = log_dir.joinpath(EPISODES_FILE_NAME)
    
    df_summary = pd.read_csv(summary_path, index_col=['episode','env'])
    df_all = pd.read_csv(episodes_path, index_col=['episode','env'])
    
    return df_summary, df_all

def plot_trajectory(df, axs:Axes = None):
    
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

def create_heatmap(df, sigma=1, bins=101, axs:Axes = None):
    
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
    
    axs.imshow(heatmap.T, extent=extent, origin='lower',interpolation='nearest', cmap = cm.get_cmap('plasma'))
    
    return fig, axs

def generate_heatmap_video(df, dir, filename="heatmap.mp4", n_intervals=100, sigma=1, bins=101):
    dpi = 100
    frame_size = (480, 480)
    fig_size = (frame_size[0] / dpi, frame_size[1] / dpi)
    
    fig, axs = plt.subplots(1,1, figsize=fig_size, dpi=dpi)
    output_file = str(Path(dir).joinpath(filename))

    frame_size = (480, 480)
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    max_ep = df.index.max()[0]
    arr = []
    data_sorted = df.sort_index()
    
    if n_intervals >  max_ep:
        n_intervals = max_ep
    intervals = np.linspace(1, max_ep, n_intervals).astype(int)
    print(intervals)

    # Filtrowanie DataFrame dla każdego przedziału
    results = []
    for i in range(len(intervals) - 1):
        lower_bound = intervals[i]
        upper_bound = intervals[i + 1]

        filtered = data_sorted.loc[(slice(lower_bound, upper_bound), slice(None)), :]
        heatmap, xedges, yedges = np.histogram2d(filtered['pos_X'],filtered['pos_Y'], bins=bins, range=MAP_BOUNDARY)
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Create the image
        im = axs.imshow(heatmap.T, extent=extent, origin='lower', interpolation='nearest', cmap='plasma')
        
        axs.set_xlim(MAP_BOUNDARY[0])
        axs.set_ylim(MAP_BOUNDARY[1])
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.set_aspect('equal')
        axs.set_title(f"episode: {lower_bound}-{upper_bound}")
        
        fig.canvas.draw()  # Render the figure

        # Extract the image as an RGB array, convert to BGR
        rgb_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb_image = rgb_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Resize image to match the video frame size
        bgr_image = cv2.resize(bgr_image, frame_size)
        
        out.write(bgr_image)
        axs.cla()  # Clear the axis for the next frame

    out.release()
    plt.close(fig)
    return arr

def get_rows_by_ep_env(df, indexes:list[list[int, int]] = []):
    return df.loc[indexes]

def get_n_best_rewards(df, n_episodes=10):
    grouped = df.groupby(by=df.index.names)
    acc = grouped.sum().sort_values(by='reward', ascending=False)
    indexes = acc[:n_episodes].index.to_list()
    best = df.loc[indexes]
    return best


def do_basic_analysis(log_dir):
    
    df_summary, df_episodes = load_dfs(log_dir)
    
    generate_heatmap_video(df_episodes,log_dir)
    # create_heatmap(df_episodes)

if __name__ == "__main__":
    log_dir = rf"D:\kody\rl_project\out\logs\A2C\A2C_1"
    do_basic_analysis(log_dir)
    
