import random
import mujoco.viewer
import mujoco
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import math
from typing import Annotated

SELF_DIR_PATH = Path(__file__).parent
PATTERN_DIR = SELF_DIR_PATH/"patterns"
MAIN_MODEL = SELF_DIR_PATH/"main.xml"

def generateCarProperties(carName: str,
                          chassisLength: Annotated[float, "length in meters"],
                          chassisWidth: Annotated[float, "width in meters"],
                          chassisHeight: Annotated[float, "height in meters"],
                          carMass: Annotated[float, "Car mass in kgs"] = 1500,
                          wheelRadius: Annotated[float, "Car wheel/tire radius in meters"] = 0.5,
                          wheelThickness:  Annotated[float, "wheel/tire thickness in meters"] = 0.2,
                          wheelMass: Annotated[float, "wheel/tire mass in kgs"] = 30,
                          wheelAxisSpacing: Annotated[float, "spacing of axis from center to edge, range: 0 to 1"] = 0.6,
                          wheelSpacing: Annotated[float, "Distance between wheels, range: 0 to 1"] = 1,
                          wheelMountHeight: Annotated[float, "range: -1 to 1"] = 0,
                          lightsSpacing: Annotated[float, "range: 0 to 1"] = 0.6):
    
    chassisXSize = chassisLength/2
    chassisYsize = chassisWidth/2
    chassisZsize = chassisHeight/2



    prop = {
        "car_name":f"{carName}",
        "car_pos": "0 0 5",
        "car_size": f"{chassisXSize} {chassisYsize} {chassisZsize}",
        "car_mass": f"{carMass}",
        
        "car_wheel_size": f"{wheelRadius} {wheelThickness/2}",
        "car_wheel_mass": f"{wheelMass}",
        "left_front_wheel_pos": f"{chassisXSize*wheelAxisSpacing} {-chassisYsize*wheelSpacing} {chassisHeight*wheelMountHeight}",
        "right_front_wheel_pos": f"{chassisXSize*wheelAxisSpacing} {chassisYsize*wheelSpacing} {chassisHeight*wheelMountHeight}",
        "left_back_wheel_pos": f"{-chassisXSize*wheelAxisSpacing} {-chassisYsize*wheelSpacing} {chassisHeight*wheelMountHeight}",
        "right_back_wheel_pos": f"{-chassisXSize*wheelAxisSpacing} {chassisYsize*wheelSpacing} {chassisHeight*wheelMountHeight}",

        "car_front_lights_pos": f"{chassisXSize} 0 0",
        "car_front_right_light_pos":f"0 {-chassisYsize*lightsSpacing} 0",
        "car_front_left_light_pos":f"0 {chassisYsize*lightsSpacing} 0",

    }


    return prop

def generateTopDownCamProperties(cameraName:str, xlen, ylen, camFov=90):
    camHeight = calculateCameraHeight(xlen, ylen, camFov)
    prop= {
        "cam_name": f"{cameraName}",
        "cam_pos": f"0 0 {camHeight}",
        "cam_euler": "0 0 -90",
        "cam_fov": f"{camFov}"
    }
    return prop


def _generateRandomCoords(x, y):
    x_coord = random.uniform(-x / 2, x / 2)
    y_coord = random.uniform(-y / 2, y / 2)
    return (x_coord, y_coord)

def generateWallsProperties(name, xlen, ylen, hlen, tlen = 0.1):
    x = xlen/2
    y = ylen/2
    h = hlen/2
    t = tlen
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


    BOARD_LENGTH = 20
    BOARD_WIDTH = 20
    BOARD_WALLS_HEIGHT = 2
    GROUND_PROP = generateWallsProperties("main_ground", BOARD_LENGTH, BOARD_WIDTH, BOARD_WALLS_HEIGHT)
    CAM_PROPERTIES = generateTopDownCamProperties("TopDownCam1", BOARD_LENGTH, BOARD_WIDTH, 45)
    CAR_PROPERTIES = generateCarProperties("car1", 5, 2, 0.25)
    XML_ELEMENTS = [
        [generator.root, PATTERN_DIR/"boardAssetPattern.xml"],
        [generator.worldbodyNode, PATTERN_DIR/"boardBodyPattern.xml", GROUND_PROP],
        [generator.worldbodyNode, PATTERN_DIR/"cameraPattern.xml", CAM_PROPERTIES],
        [generator.worldbodyNode, PATTERN_DIR/"carPattern.xml", CAR_PROPERTIES],                                              
    ]

    for args in XML_ELEMENTS:
        addPatternToTree(*args)



    generator.runSimulation()