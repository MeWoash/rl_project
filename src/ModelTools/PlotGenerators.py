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
from scipy.interpolate import interp1d

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
        self.ax_label = f"top {n_best} rewarded trajectory"
        
    def plot(self, df):
        best = get_n_best_rewards(df, self.n_best)
        return super().plot(best)


class PlotRewardCurve(PlotBaseAbstract):
    def __init__(self,
                 df_episodes_all,
                 ax= None,
                 legend = False
                 )-> None:
        super().__init__(ax)
        self.legend = legend
        self.ax_label = "rewards"
        self.df_episodes_all = df_episodes_all
        
    def plot(self, df) -> Tuple[Figure | Axes]:
        super().plot()
        
        for index, grouped in group_by_envs(self.df_episodes_all):
            
            learning_step = grouped['learning_step'].to_numpy()
            episode_norm_cum_reward = grouped['episode_norm_cum_reward'].to_numpy()
            
            line  = self._ax.plot(learning_step,
                          episode_norm_cum_reward,
                          label=f"env-{index}",
                          color='blue',
                          alpha=0.4)
            
            
        for indexes, grouped in group_by_episodes(df):
            learning_step = grouped['learning_step'].to_numpy()
            episode_norm_cum_reward = grouped['episode_norm_cum_reward'].to_numpy()
            
            self._ax.plot(learning_step,
                          episode_norm_cum_reward,
                          label=f"ep-{indexes[0]}_env-{indexes[1]}",
                          linewidth=2)

        self._ax.grid(True)
        self._ax.set_xlabel('learning step')
        self._ax.set_ylabel('episode normalized cumulative reward')
        
        if self.legend:
            self._ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
        return self._fig, self._ax
     
class PlotBestRewardCurve(PlotRewardCurve):
    def __init__(self,
                 df_episodes_all,
                 ax= None,
                 legend = False,
                 n_best = 1
                 ) -> None:
        super().__init__(df_episodes_all, ax, legend)
        self.n_best = n_best
        self.ax_label = f"top {n_best} on cumulative reward"
        
    def plot(self, df):
        best = get_n_best_rewards(df, self.n_best)
        return super().plot(best)

class PlotActions(PlotBaseAbstract):
    def __init__(self,
                 ax= None,
                 legend = False
                 )-> None:
        super().__init__(ax)
        self.legend = legend
        self.ax_label = "actions"
        
    def plot(self, df) -> Tuple[Figure | Axes]:
        super().plot()
        for indexes, grouped in group_by_episodes(df):
            action_engine = grouped['action_engine'].to_numpy()
            action_angle = grouped['action_angle'].to_numpy()
            x = grouped['episode_mujoco_time'].to_numpy()
            
            color_engine = 'blue'
            color_angle = 'green'
            
            self._ax.plot(x, action_engine, label="engine", linestyle='-', color=color_engine, alpha=1)
            self._ax.fill_between(x, action_engine, color=color_engine, alpha=0.3)
            
            self._ax.plot(x, action_angle, label="angle", linestyle='-', color=color_angle, alpha=1)
            self._ax.fill_between(x, action_angle, color=color_angle, alpha=0.3)
            
            # INTERPOLATE
            # x_new = np.linspace(x.min(), x.max(), 30)
            # interp_engine = interp1d(x, action_engine, kind='cubic')(x_new)
            # interp_angle = interp1d(x, action_angle, kind='cubic')(x_new)
            # self._ax.plot(x_new, interp_engine, label="engine (interp)", color=color_engine)
            # self._ax.plot(x_new, interp_angle, label="angle (interp)", color=color_angle)

        self._ax.grid(True)
        
        self._ax.set_xlabel('env time [ms]')
        self._ax.set_ylabel('y')
        
        if self.legend:
            self._ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
        return self._fig, self._ax
    
class PlotBestActions(PlotActions):
    def __init__(self,
                 ax= None,
                 legend = False,
                 n_best = 1
                 ) -> None:
        super().__init__(ax)
        self.legend = legend
        self.n_best = n_best
        self.ax_label = f"top {n_best} rewarded action"
        
        
    def plot(self, df):
        best = get_n_best_rewards(df, self.n_best)
        return super().plot(best)
        
class PlotHeatMap(PlotBaseAbstract):
    def __init__(self,
                 ax=None,
                 sigma=2,
                 bins=51
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
            fig, axes = plt.subplots(1, len(self._generators), squeeze=False, figsize=(10,7))
        
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

