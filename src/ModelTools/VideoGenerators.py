from math import ceil
from pathlib import Path
import time
from typing import Callable

import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import sys

# autopep8: off
sys.path.append(str(Path(__file__,'..','..').resolve()))
from ModelTools.Utils import *
from ModelTools.PlotGenerators import *
# autopep8: on


class VideoGenerator():
    def __init__(self,
                plot_wrapper: PlotWrapper,
                dir:str,
                filename:str="out.mp4",
                frame_size = (480, 480),
                fps = 10,
                dpi = 100) -> None:
        
        self.plot_wrapper:PlotWrapper = plot_wrapper
        self.frame_size= frame_size
        self.fps = fps
        self.dpi = dpi
        self.fig_size = (frame_size[0] / dpi, frame_size[1] / dpi)

        self.fig: Figure
        self.axes: list[Axes]
        self.fig, self.axes = plt.subplots(*self.plot_wrapper.subplot_layout, figsize=self.fig_size, dpi=dpi)
        
        plot_wrapper.assign_axes(axes = self.axes)
        
        self.output_file = str(Path(dir, filename).resolve())

    def generate_video(self,df):
        
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file, fourcc, self.fps, self.frame_size)
        arr = []
    
        for (lower_bound, upper_bound), filtered in batch_by_episodes(df, 100):
    
            self.plot_wrapper.plot(filtered)
            self.fig.suptitle(f"episode: <{lower_bound}, {upper_bound})")
            self.fig.canvas.draw()

            rgba_image = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)

            w, h = self.fig.canvas.get_width_height()
            rgba_image = rgba_image.reshape((h, w, 4))

            rgb_image = rgba_image[:, :, :3]
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            bgr_image = cv2.resize(bgr_image, self.frame_size)
            
            out.write(bgr_image)
            for ax in self.axes:
                ax.cla()

        out.release()
        plt.close(self.fig)
        return arr