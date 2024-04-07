import sys
from pathlib import Path

SELF_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SELF_DIR))

try:
    from Generator import Generator
    from Globals import *
except BaseException as e:
    print(e)
    raise Exception
