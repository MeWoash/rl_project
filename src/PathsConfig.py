from pathlib import Path

# ================= PATHS ======================

ROOT_WS_DIR = Path(__file__, '..', '..').resolve()

EPISODES_ALL = "episodes_all.csv"
EPISODE_STATS = "episodes_stats.csv"
TRAINING_STATS = "training_stats.csv"

SRC_DIR = str(ROOT_WS_DIR.joinpath('src').resolve())
ASSET_DIR = str(ROOT_WS_DIR.joinpath("assets").resolve())
OUT_LEARNING_DIR = str(ROOT_WS_DIR.joinpath("out", "learning").resolve())
MEDIA_DIR = str(ROOT_WS_DIR.joinpath("out","media").resolve())
MJCF_OUT_DIR = str(ROOT_WS_DIR.joinpath("out","mjcf").resolve())
MJCF_MODEL_NAME = "out.xml"