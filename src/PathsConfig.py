from pathlib import Path

# ================= PATHS ======================

ROOT_WS_DIR = Path(__file__, '..', '..').resolve()

ASSET_DIR = str(ROOT_WS_DIR.joinpath("assets"))
OUT_RL_DIR = str(ROOT_WS_DIR.joinpath("out", "learning"))
MEDIA_DIR = str(ROOT_WS_DIR.joinpath("out","media"))
MJCF_OUT_DIR = str(ROOT_WS_DIR.joinpath("out","mjcf"))
MJCF_MODEL_NAME = "out.xml"