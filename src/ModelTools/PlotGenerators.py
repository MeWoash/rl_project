from abc import ABC, abstractmethod
import enum
from math import ceil
import sys
from typing import Any, Tuple, Type, TypeVar, Union
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

class PlotBaseAbstract(ABC):
    def __init__(self,
                 ax = None,
                 ) -> None:
        
        self._fig: Figure = None
        self._ax: Axes = ax
        self._ax_label = ""
        
        if ax is None:
            self._fig, self._ax = plt.subplots(1,1)
        else:
            self._fig = ax.get_figure()

    @property
    def ax_label(self) -> Axes:
        return self._ax_label
    
    @ax_label.setter
    def ax_label(self, val):
        self._ax_label = val
    
    @property
    def ax(self) -> Axes:
        return self._ax


    @ax.setter
    def ax(self, ax: Axes):
        self._ax = ax
        plt.close(self._fig)
        self._fig = self._ax.get_figure()


    @property
    def fig(self) -> Axes:
        """
        To avoid situation when plot has assgigned ax which does not belong to figure, it doesnt have fig setter.

        Returns:
            Axes: Figure
        """
        return self._fig


    @abstractmethod
    def plot(self):
        self._ax.set_title(self._ax_label)


class PlotTrajectory(PlotBaseAbstract):
    def __init__(self,
                 ax= None,
                 legend = False
                 )-> None:
        super().__init__(ax)
        self.legend = legend
        self.ax_label = "trajectory"
        
    def plot(self, df) -> Tuple[Figure | Axes]:
        super().plot()
        for indexes, grouped in group_by_episodes(df):
            x = grouped['pos_X'].to_numpy()
            y = grouped['pos_Y'].to_numpy()
            self._ax.plot(x, y, label=f"ep-{indexes[0]}_en-{indexes[1]}")

            self._ax.grid(True)
            self._ax.set_xlim(MAP_BOUNDARY[0])
            self._ax.set_ylim(MAP_BOUNDARY[1])
            
            self._ax.set_xlabel('x')
            self._ax.set_ylabel('y')
            self._ax.set_aspect('equal')
        
        if self.legend:
            self._ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
        return self._fig, self._ax
    
    
class PlotBestTrajectory(PlotTrajectory):
    def __init__(self,
                 ax= None,
                 legend = False,
                 n_best = 1
                 ) -> None:
        super().__init__(ax)
        self.legend = legend
        self.n_best = n_best
        self.ax_label = f"top {n_best} trajectory"
        
    def plot(self, df):
        best = get_n_best_rewards(df, self.n_best)
        return super().plot(best)
        
class PlotHeatMap(PlotBaseAbstract):
    def __init__(self,
                 ax=None,
                 sigma=1,
                 bins=101
                 ) -> None:
        super().__init__(ax)
        self.sigma = sigma
        self.bins = bins
        self.ax_label = "trajectory heatmap"
        
    def plot(self, df) -> Tuple[Figure | Axes]:
        super().plot()
        heatmap, xedges, yedges = np.histogram2d(df['pos_X'], df['pos_Y'], bins=self.bins, range=MAP_BOUNDARY)
        heatmap = gaussian_filter(heatmap, sigma=self.sigma)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        self._ax.set_xlim(MAP_BOUNDARY[0])
        self._ax.set_ylim(MAP_BOUNDARY[1])
        
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.set_aspect('equal')
        
        self._ax.imshow(heatmap.T, extent=extent, origin='lower',interpolation='nearest', cmap = matplotlib.colormaps['plasma'])
        
        return self._fig, self._ax


class PlotWrapper():
    def __init__(self,
                 generators: list[Type[PlotBaseAbstract]] | None = None,
                 fig: Figure | None = None,
                 axes: list[Axes] | None = None
                 ):
                 
        self._generators = generators
        self.assign_axes(fig, axes)
    
    @property
    def fig(self) -> Figure:
        if self._fig is None:
            raise Exception("Fig has not been assigned yet.")
        return self._fig
    
    @property
    def axes(self)-> np.ndarray[Axes]:
        if self._axes is None:
            raise Exception("Axes havent been assigned yet.")
        return self._axes
    
    def assign_axes(self,
                    fig: Figure | None = None,
                    axes: np.ndarray[Axes] | np.ndarray[np.ndarray[Axes]] | None = None):
        
        if fig is None or axes is None:
            fig, axes = plt.subplots(1, len(self._generators), squeeze=False)
        
        self._fig = fig
        self._axes: np.ndarray[Axes] = axes.flatten()
        
        # IMPORTANT CHECK
        assert len(self._generators) <= len(self._axes)
        
        for i, gen in enumerate(self._generators):
            gen.ax = self._axes[i]

    def plot(self, df):
        for gen in self._generators:
            gen.plot(df)
        
        return self._fig, self._axes

