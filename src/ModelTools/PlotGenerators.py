from abc import ABC, abstractmethod
import enum
from math import ceil
import sys
from typing import Type, TypeVar, Union
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
    def __init__(self, ax = None) -> None:
        self._fig: Figure = None
        self._ax: Axes = ax
        
        if ax is None:
            self._fig, self._ax = plt.subplots(1,1)
        else:
            self._fig = ax.get_figure()


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
        return self._fig


    @abstractmethod
    def plot(self, df):
        raise NotImplementedError("Plot not implemented in derived class!")


class PlotTrajectory(PlotBaseAbstract):
    def __init__(self, ax= None, legend = False) -> None:
        super().__init__(ax)
        self.legend = legend
        
    def plot(self, df):
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
        
class PlotHeatMap(PlotBaseAbstract):
    def __init__(self, ax=None, sigma=1, bins=51) -> None:
        super().__init__(ax)
        self.sigma = sigma
        self.bins = bins
        
    def plot(self, df):
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
    def __init__(self, subplot_layout = None, generators: Union[list[Type[PlotBaseAbstract]],None] = None) -> None:
        self._fig: Figure = None
        self._axes: list[Axes] = []
        self._generators:list[Type[PlotBaseAbstract]] = []
        
        if generators is not None and subplot_layout is not None:
            self.create_wrapper(subplot_layout, generators)
    
    def create_wrapper(self, subplot_layout, generators: list[Type[PlotBaseAbstract]]):
        self._generators = generators
        assert len(self._generators) <= subplot_layout[0]*subplot_layout[1]
        
        self._fig, self._axes = plt.subplots(*subplot_layout, squeeze=True)
        
        if len(self._axes) == 1:
            self._axes = [self._axes]
        
        for i, gen in enumerate(self._generators):
            print(self._axes[i])
            gen.ax = self._axes[i]
    
    def plot(self, df):

        for gen in self._generators:
            gen.plot(df)
        
        return self._fig, self._axes

