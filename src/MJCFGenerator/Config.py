# ======================== PARKING_SPOT ====================
PARKING_NAME = "parkingSpot"
PARKING_POS = [4, 5, 0]
PARKING_SPOT_PADDINGS = (1.2, 1.5)
PARKING_LINE_WIDTH = 0.15
PARKING_LINE_HEIGHT_SIZE = 0.001

# ======================= WHEEL ============================
WHEEL_FRICTION = [1.2, 0.01, 0.02]
WHEEL_MASS = 30
WHEEL_ANGLE_LIMIT = (-45, 45)
WHEEL_RADIUS = 0.3
WHEEL_THICKNESS = 0.2

# ======================== CAR =============================
CAR_NAME = "mainCar"
CAR_DIMS = (2, 1, 0.25)
CAR_MASS = 1500
CAR_WHEEL_AXIS_SPACING = 0.6
CAR_WHEEL_SPACING = 0.79
CAR_WHEEL_MOUNT_HEIGHT = -0.25
CAR_SPAWN_HEIGHT = WHEEL_RADIUS + CAR_DIMS[2] * abs(CAR_WHEEL_MOUNT_HEIGHT)
CAR_COLOR = (0.8, 0.102, 0.063, 1)
CAR_N_RANGE_SENSORS = 5

# ======================== TRAILER =============================
TRAILER_NAME = "mainTrailer"
TRAILER_DIMS = (1.75, 1, 0.25, 0.7)
TRAILER_MASS = 1500
TRAILER_WHEEL_AXIS_SPACING = 0.6
TRAILER_WHEEL_SPACING = 0.79
TRAILER_WHEEL_MOUNT_HEIGHT = -0.25
TRAILER_COLOR = (0.235, 0.761, 0.196, 1)
TRAILER_N_RANGE_SENSORS = 5

# ====================== GENERAL ==============================
CUSTOM_OBSTACLES =\
[
    {"size":[0.5, 2, 1],  "pos":[4, 1, 0]},
    {"size":[0.5, 2, 1],  "pos":[4, -7, 0]},
    {"size":[5, 1, 1],  "pos":[0, 8, 0]},
    {"size":[1, 1, 1],  "pos":[-2, -1, 0]}
]
SPAWN_POINTS: list[dict[str, list[int]]] =\
[ # Position Z will be overwritten to match surface level. 
    {"pos": [-5, -5, CAR_SPAWN_HEIGHT], "euler": [0, 0, -30]},
    {"pos": [-6, -5, CAR_SPAWN_HEIGHT], "euler": [0, 0, 90]},
    {"pos": [7, -5, CAR_SPAWN_HEIGHT], "euler": [0, 0, 90]},
    {"pos": [5, -3, CAR_SPAWN_HEIGHT], "euler": [0, 0, 180]},
    {"pos":[-7, -3, CAR_SPAWN_HEIGHT], "euler":[0, 0, 60]},
    {"pos":[-6, 3, CAR_SPAWN_HEIGHT], "euler":[0, 0, -30]},
    {"pos": [8, 8, CAR_SPAWN_HEIGHT], "euler": [0, 0, 40]},
    {"pos": [1, -6, CAR_SPAWN_HEIGHT], "euler": [0, 0, 90]},
    # {"pos":[-3, 5, CAR_SPAWN_HEIGHT], "euler":[0, 0, -5]},
    # {"pos":[8, 3.5, CAR_SPAWN_HEIGHT], "euler":[0, 0, -15]}
]


MAP_LENGTH = [20, 20, 20, 5]

SENSORS_MAX_RANGE = 5

RENDER_OFF_HEIGHT = 720
RENDER_OFF_WIDTH = 1280