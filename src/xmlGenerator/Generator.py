import random
import mujoco.viewer
import mujoco
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import math

SELF_DIR_PATH = Path(__file__).parent
PATTERN_DIR = SELF_DIR_PATH/"patterns"
MAIN_MODEL = SELF_DIR_PATH/"main.xml"


def generate_random_coords(x, y):
    x_coord = random.uniform(-x / 2, x / 2)
    y_coord = random.uniform(-y / 2, y / 2)
    return (x_coord, y_coord)

def getWallsProperties(name, x, y, h, t):
    prop = {
        "ground_name": f"{name}",
        "board_size": f"{x} {y} {t}",
        "left_size":f"{t} {y} {h}",
        "left_pos":f"{-(x+t)} 0 {h}",
        "right_size":f"{t} {y} {h}",
        "right_pos":f"{x+t} 0 {h}",
        "front_size":f"{x} {t} {h}",
        "front_pos":f"0 {y+ t} {h}",
        "back_size":f"{x} {t} {h}",
        "back_pos":f"0 {-(y+t)} {h}",
    }
    return prop

def calculateCameraHeight(x, y, fov_y_degrees):
    fov_y_radians = math.radians(fov_y_degrees)
    h = max(x, y) / math.tan(fov_y_radians / 2)
    return h

def createElement(tagName: str, attributes: dict = {}):
    element = ET.Element(tagName, attributes)
    return element

def createAppendElement(nodeToAppend: ET.Element, tagName: str, attributes: dict = {}):
    element = createElement(tagName, attributes)
    nodeToAppend.append(element)
    return nodeToAppend

def _readXMLReplacePlaceHolders(patternPath:str, patternVariables):
    with open(patternPath, 'r') as file:
        templateXML = file.read()
    replacedXML = templateXML.format(**patternVariables)
    return ET.fromstring(replacedXML)

def addPatternToTree(nodeToAppend:ET.ElementTree, patternPath: str, patternVariables: dict = {}, extraAttributes = {}):
    newNode = _readXMLReplacePlaceHolders(patternPath, patternVariables)
    newNode.attrib.update(extraAttributes)
    if newNode.tag == "FAKE_GROUP":
        for child in newNode:
            nodeToAppend.append(child)
    else:
        nodeToAppend.append(newNode)
    return newNode

class MujocoXMLGenerator():
    def __init__(self) -> None:
        self._initTree()

    def _initTree(self):
        self.tree = ET.parse(PATTERN_DIR/"mainPattern.xml")
        self.root = self.tree.getroot()
        self.worldbodyNode = self.root.find("worldbody")

    def saveTree(self):
        self.tree.write(MAIN_MODEL)

    def runSimulation(self):
        self.saveTree()
        # self.model = mujoco.MjModel.from_xml_path(MAIN_MODEL)
        mujoco.viewer.launch_from_path(str(MAIN_MODEL))

if __name__ == "__main__":
    generator = MujocoXMLGenerator()


    BOARD_X = 10
    BOARD_Y = 15
    GROUND_PROP = getWallsProperties("main_ground", BOARD_X, BOARD_Y, 2, 0.1)
    CAMERA_FOV = 45
    CAMERA_Z = calculateCameraHeight(BOARD_X, BOARD_Y, CAMERA_FOV)
    CAR_INIT_CORDS = generate_random_coords(BOARD_X, BOARD_Y)
    XML_ELEMENTS = [
        [generator.root, PATTERN_DIR/"boardAssetPattern.xml"],
        [generator.worldbodyNode, PATTERN_DIR/"boardBodyPattern.xml", GROUND_PROP],
        [generator.worldbodyNode, PATTERN_DIR/"cameraPattern.xml", {"cam_name": "TopDownCam",
                                                                    "cam_pos": f"0 0 {CAMERA_Z}",
                                                                    "cam_euler": "0 0 -90",
                                                                    "cam_fov": str(CAMERA_FOV)}],
        [generator.worldbodyNode, PATTERN_DIR/"carPattern.xml", {"car_name":"car1",
                                                                "car_pos":f"{CAR_INIT_CORDS[0]} {CAR_INIT_CORDS[1]} 0.5"}],                                              
    ]

    for args in XML_ELEMENTS:
        addPatternToTree(*args)



    generator.runSimulation()