# autopep8: off

from math import ceil
from pathlib import Path
from typing import Generator

import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import sys

sys.path.append(str(Path(__file__,'..','..').resolve()))
from PostProcessing.Utils import generator_episodes, divide_by_steps
from PostProcessing.PlotGenerators import PlotWrapper
# autopep8: on


class VideoGenerator():
    def __init__(self,
                plot_wrapper: PlotWrapper,
                dir:str,
                frame_size = (1080, 1080),
                fps = 10,
                length = 3,
                dpi = 100) -> None:
        
        self._plot_wrapper:PlotWrapper = plot_wrapper
        self._frame_size= frame_size
        self._fps = fps
        self._dpi = dpi
        self._length = length
        fig_size = (frame_size[0] / dpi, frame_size[1] / dpi)

        self._fig: Figure = self._plot_wrapper.fig
        self._axes: np.ndarray[Axes] = self._plot_wrapper.axes
        
        self._fig.set_size_inches(fig_size)
        self._fig.set_dpi(dpi)
        self._dir = dir

    def generate_video(self,
                       df: pd.DataFrame,
                       file_name,
                       fig_title = "",
                       generator_function: Generator = generator_episodes):
        
        output_file = str(Path(self._dir, file_name).resolve())
        
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self._fps, self._frame_size)
        arr = []

        n_parts = self._fps * self._length
        parts, bins = divide_by_steps(df, n_parts)
        
        for i in range(n_parts):
        
            self._plot_wrapper.plot(parts[i])
            
            title = f"step: {int(bins[i+1])}"
            self._fig.suptitle(title)
            self._fig.canvas.draw()

            rgba_image = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
            
            w, h = self._fig.canvas.get_width_height()
            rgba_image = rgba_image.reshape((h, w, 4))

            rgb_image = rgba_image[:, :, :3]
            
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            bgr_image = cv2.resize(bgr_image, self._frame_size)
               
            out.write(bgr_image)
            for ax in self._axes:
                ax.cla()

        out.release()
        plt.close(self._fig)
        return arr