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
except:
    raise ImportError


# def generateTopDownCamProperties(cameraName: str, xlen: float, ylen: float, camFov=90):
#     camHeight = calculateCameraHeight(xlen, ylen, camFov)
#     prop = {
#         "cam_name": f"{cameraName}",
#         "cam_pos": f"0 0 {camHeight}",
#         "cam_euler": "0 0 -90",
#         "cam_fov": f"{camFov}"
#     }
#     return prop


def _generateRandomCoords(x, y):
    x_coord = random.uniform(-x / 2, x / 2)
    y_coord = random.uniform(-y / 2, y / 2)
    return x_coord, y_coord


def generateWallsProperties(name: str, xlen: float, ylen: float, hlen: float, tlen: float = 0.1):
    x = xlen / 2
    y = ylen / 2
    h = hlen / 2
    t = tlen
    prop = {
        "ground_name": f"{name}",
        "board_size": f"{x} {y} {t}",
        "left_size": f"{t} {y} {h}",
        "left_pos": f"{-(x + t)} 0 {h}",
        "right_size": f"{t} {y} {h}",
        "right_pos": f"{x + t} 0 {h}",
        "front_size": f"{x} {t} {h}",
        "front_pos": f"0 {y + t} {h}",
        "back_size": f"{x} {t} {h}",
        "back_pos": f"0 {-(y + t)} {h}",
    }
    return prop


# def calculateCameraHeight(x, y, fov_y_degrees):
#     fov_y_radians = math.radians(fov_y_degrees)
#     h = max(x, y) / math.tan(fov_y_radians / 2)
#     return h


# def createElement(tagName: str, attributes: dict = {}):
#     element = ET.Element(tagName, attributes)
#     return element


# def createAppendElement(nodeToAppend: ET.Element, tagName: str, attributes: dict = {}):
#     element = createElement(tagName, attributes)
#     nodeToAppend.append(element)
#     return nodeToAppend


# def createNodeFromXMLPattern(patternPath: str, patternVariables, extraAttributes = {}):
#     with open(patternPath, 'r') as file:
#         templateXML = file.read()
#     replacedXML = templateXML.format(**patternVariables)
#     newNode: ET.Element = ET.fromstring(replacedXML)
#     newNode.attrib.update(extraAttributes)
#     return


# def addNodeToTree(nodeToAppend: ET.ElementTree, appendedNode: ET.ElementTree):
#     if appendedNode.tag == "FAKE_GROUP":
#         for child in appendedNode:
#             nodeToAppend.append(appendedNode)
#     else:
#         nodeToAppend.append(appendedNode)
#     return nodeToAppend


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

    def createGround(self):
        worldGenerator = WorldGenerator("ground", 100, 100, 10)
        worldGenerator.attachToMujoco(self.root)

    def createCar(self):
        carGenerator = CarGenerator("car1",
                                    2,
                                    1,
                                    0.5)
        carGenerator.attachToMujoco(self.root)

    def _saveTree(self):
        self.tree.write(MAIN_MODEL)

    def runSimulation(self):
        self._saveTree()
        mujoco.viewer.launch_from_path(str(MAIN_MODEL))


if __name__ == "__main__":

    generator = MujocoXMLGenerator()
    generator.createGround()
    generator.createCar()
    generator.runSimulation()
