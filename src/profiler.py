import cProfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__,'..','..').resolve()))
from ModelTools.ModelManager import train_models
from PathsConfig import *

if __name__ == "__main__":

    cProfile.run("train_models()", str(Path(__file__,"..","program.prof").resolve()))
    #snakeviz .\src\program.prof