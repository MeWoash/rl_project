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
XMLS_PATH = SELF_DIR_PATH.joinpath("../xmls")
MAIN_MODEL: Path = XMLS_PATH.joinpath("generated.xml")
GENERATOR_PARTS: Path = SELF_DIR_PATH.joinpath("GeneratorParts")

try:
    sys.path.append(str(GENERATOR_PARTS))
    from GeneratorParts.CarGenerator import CarGenerator
    from GeneratorParts.WorldGenerator import WorldGenerator
    from GeneratorParts.CameraGenerator import CameraGenerator
    from GeneratorParts.TargetGenerator import TargetGenerator
except:
    raise ImportError


class MujocoXMLGenerator():
    PATTERN = """\
    <mujoco>
        <worldbody>
        </worldbody>
    </mujoco>"""

    def __init__(self) -> None:
        self._initTree()

    def _initTree(self):
        self.root = ET.fromstring(MujocoXMLGenerator.PATTERN)
        self.tree = ET.ElementTree(self.root)
        self.worldbodyNode = self.tree.find("worldbody")

    def saveTree(self, outputPath=MAIN_MODEL):
        ET.indent(self.tree, space="    ", level=0)
        self.tree.write(outputPath)

    def runSimulation(self, outputPath=MAIN_MODEL):
        mujoco.viewer.launch_from_path(str(outputPath))


def sampleWorldCreation() -> MujocoXMLGenerator:

    GROUND_SIZE: Tuple[int, int, int] = (20, 20, 20, 5)
    CAR_SIZE: Tuple[int, int, int] = (2, 1, 0.25)

    generator = MujocoXMLGenerator()
    # GROUND
    worldGenerator = WorldGenerator("ground", *GROUND_SIZE)
    worldGenerator.attachToMujoco(generator.root)

    # CAMERA
    cameraGenerator: CameraGenerator = CameraGenerator(
        "TopDownCam", GROUND_SIZE[0], GROUND_SIZE[1])
    cameraGenerator.attachToMujoco(generator.root)

    # CAR
    carPos = (0, 0, )
    carGenerator = CarGenerator("mainCar", *CAR_SIZE, )
    carGenerator.attachToMujoco(generator.root)

    # TARGET - PARKING SPOT
    targetGenerator = TargetGenerator(CAR_SIZE,
                                      targetPos=(5, 5, 0))
    targetGenerator.attachToMujoco(generator.root)

    generator.saveTree()
    return generator


if __name__ == "__main__":

    generator: MujocoXMLGenerator = sampleWorldCreation()
    generator.runSimulation()
