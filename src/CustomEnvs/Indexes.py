from enum import IntEnum
import sys
from pathlib import Path

sys.path.append(str(Path(__file__,'..','..').resolve()))


# INCLUSIVE INDEXES
class ACTION_INDEX(IntEnum):
    # 1 val
    ENGINE = 0
    # 1 val
    ANGLE = 1
    # SUM
    ACTION_SIZE = 2


class OBS_INDEX(IntEnum):
    # 1 val
    VELOCITY_BEGIN = 0
    VELOCITY_END = 0
    # 1 val
    DISTANCE_BEGIN = 1
    DISTANCE_END = 1
    # 2 val
    CAR_ANGLE_DIFF_BEGIN = 2
    CAR_ANGLE_DIFF_END = 2
    # 2 val 
    REL_POS_BEGIN = 3
    REL_POS_END = 4
    # 1 val
    CAR_ANGLE_BEGIN = 5
    CAR_ANGLE_END = 5
    # 1 val
    HITCH_ANGLE_BEGIN = 6
    HITCH_ANGLE_END = 6

    # 10 val
    RANGE_BEGIN = 7
    RANGE_END = 16
    # SUM
    OBS_SIZE = 17
    
    
class EXTRA_OBS_INDEX(IntEnum):
    # 2 val
    GLOBAL_POS_BEGIN = 0
    GLOBAL_POS_END = 1
    # 2 val
    CONTACT_BEGIN = 2
    CONTACT_END = 3
    
    OBS_SIZE = 4