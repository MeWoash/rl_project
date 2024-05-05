# autopep8: off
import sys
from pathlib import Path


SELF_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SELF_DIR))

from Generator import *
from Globals import *

