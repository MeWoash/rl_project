import random
import mujoco.viewer
import mujoco
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import math
from typing import Annotated, Tuple
import sys


SELF_DIR_PATH = Path(__file__).parent
PATTERN_DIR = SELF_DIR_PATH / "patterns"
MAIN_MODEL = SELF_DIR_PATH / "generated.xml"
GENERATOR_PARTS = SELF_DIR_PATH/"GeneratorParts"

try:
    sys.path.append(str(GENERATOR_PARTS))
    from GeneratorParts.CarGenerator import CarGenerator
    from GeneratorParts.WorldGenerator import WorldGenerator
    from GeneratorParts.CameraGenerator import CameraGenerator
except:
    raise ImportError


class MujocoXMLGenerator():
    PATTERN = """\
    <mujoco>
        <worldbody>
        <light dir="0 0 -1" pos="0 0 1000" diffuse="1 1 1"/>
        </worldbody> 
    </mujoco>"""

    def __init__(self) -> None:
        self._initTree()

    def _initTree(self):
        self.root = ET.fromstring(MujocoXMLGenerator.PATTERN)
        self.tree = ET.ElementTree(self.root)
        self.worldbodyNode = self.tree.find("worldbody")

    def createGround(self, groundSize: Tuple):
        worldGenerator = WorldGenerator("ground", *groundSize)
        worldGenerator.attachToMujoco(self.root)

    def createCamera(self, groundSize):
        cameraGenerator: CameraGenerator = CameraGenerator(
            "TopDownCam", groundSize[0], groundSize[1])
        cameraGenerator.attachToMujoco(self.root)

    def createCar(self):
        carGenerator = CarGenerator("car1",
                                    2,
                                    1,
                                    0.5)
        carGenerator.attachToMujoco(self.root)

    def saveTree(self):
        self.tree.write(MAIN_MODEL)

    def runSimulation(self):
        mujoco.viewer.launch_from_path(str(MAIN_MODEL))


if __name__ == "__main__":

    generator = MujocoXMLGenerator()
    GROUND_SIZE: Tuple[int, int, int] = (20, 20, 10)
    generator.createGround(GROUND_SIZE)
    generator.createCamera(GROUND_SIZE)
    generator.createCar()
    generator.saveTree()
    generator.runSimulation()
