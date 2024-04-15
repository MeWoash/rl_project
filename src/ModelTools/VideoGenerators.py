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
from ModelTools.PlotGenerators import *
from ModelTools.Utils import *
# autopep8: on

# import MJCFGenerator

def generate_heatmap_video(df, dir, filename="heatmap.mp4", sigma=1, bins=101):
    generate_video_from_plot_function(df, generate_heatmap_plot, dir, filename, plot_function_kwargs={"sigma":sigma, "bins":bins})
    
def generate_trajectory_video(df, dir, filename="heatmap.mp4", sigma=1, bins=101):
    generate_video_from_plot_function(df, generate_trajectory_plot, dir, filename, plot_function_kwargs={"legend":False})