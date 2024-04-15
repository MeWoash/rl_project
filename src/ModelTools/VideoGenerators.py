from math import ceil
import sys
import cv2
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
from ModelTools.PlotGenerators import generate_heatmap_plot
from ModelTools.Utils import *
# autopep8: on

# import MJCFGenerator


@timeit
def generate_heatmap_video(df, dir, filename="heatmap.mp4", sigma=1, bins=101):
    dpi = 100
    frame_size = (480, 480)
    fig_size = (frame_size[0] / dpi, frame_size[1] / dpi)
    
    fig, axs = plt.subplots(1,1, figsize=fig_size, dpi=dpi)
    output_file = str(Path(dir).joinpath(filename))
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    arr = []
  
    for filtered, upper_bound, lower_bound in batch_by_episodes(df, 100):
  
        generate_heatmap_plot(filtered, axs=axs)
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