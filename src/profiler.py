import cProfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__,'..','..').resolve()))
from ModelTools.ModelManager import train_models
import PathsConfig as paths_cfg

if __name__ == "__main__":

    cProfile.run("train_models()", str(Path(paths_cfg.OUT_LEARNING_DIR,"program.prof").resolve()))
    #snakeviz .\out\learning\program.prof