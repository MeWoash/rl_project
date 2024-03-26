import math
from typing import Any
import xml.etree.ElementTree as ET
from GeneratorBaseClass import BaseGenerator


def calculate_direction(light_pos, target_pos):
    # Calculate the vector components from light to target
    dx = target_pos[0] - light_pos[0]
    dy = target_pos[1] - light_pos[1]
    dz = target_pos[2] - light_pos[2]

    # Calculate the length of the vector
    length = math.sqrt(dx**2 + dy**2 + dz**2)

    # Normalize the vector
    dx /= length
    dy /= length
    dz /= length

    return dx, dy, dz

# <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
# <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0"/>


class WorldGenerator(BaseGenerator):
    TEMPLATES: dict[str, str] = {
        "asset": """\
            <asset>
                <texture name="sky_texture" type="skybox" file="assets/Wispy_Sky-Night_03-512x512.png"/>

                <texture name="wall_texture" type="2d" file="assets/Fuzzy_Sky-Night_01-512x512.png"/>
                <material name="wall_material" rgba = "0 0 0 0"/>

                <texture name="ground_texture" type="2d" file="assets/ground.png"/>
                <material name="ground_material" texture="ground_texture" texrepeat="25 25"/>
                
            </asset>""",

        "ground": '<geom name="{ground_name}_bottom" type="plane" size="{board_size}" material="ground_material" friction="1.0 0.005 0.0001"/>',
        "top": '<geom name="{ground_name}_top" type="box" size="{board_size}" pos="{top_pos}" material="wall_material"/>',
        "wall1": '<geom name="{ground_name}_left" type="box" size="{wall1_size}" pos="{wall1_pos}" material="wall_material"/>',
        "wall2": '<geom name="{ground_name}_right" type="box" size="{wall2_size}" pos="{wall2_pos}" material="wall_material"/>',
        "wall3": '<geom name="{ground_name}_front" type="box" size="{wall3_size}" pos="{wall3_pos}" material="wall_material"/>',
        "wall4": '<geom name="{ground_name}_back" type="box" size="{wall4_size}" pos="{wall4_pos}" material="wall_material"/>',

        "mainLight": '<light dir="0 0 -1" pos="0 0 100" diffuse="1 1 1" castshadow="true"/>',
        "light1": '<light dir="{light1_dir}" pos="{light1_pos}" diffuse="{light_dffuse}" castshadow="false"/>',
        "light2": '<light dir="{light2_dir}" pos="{light2_pos}" diffuse="{light_dffuse}" castshadow="false"/>',
        "light3": '<light dir="{light3_dir}" pos="{light3_pos}" diffuse="{light_dffuse}" castshadow="false"/>',
        "light4": '<light dir="{light4_dir}" pos="{light4_pos}" diffuse="{light_dffuse}" castshadow="false"/>',
    }

    def __init__(self,
                 name: str,
                 xlen: float,
                 ylen: float,
                 hlen: float,
                 tlen: float = 0.1) -> None:
        super().__init__()
        self.name: str = name
        self.xlen = xlen
        self.ylen = ylen
        self.hlen = hlen
        self.tlen = tlen
        self._calculateProperties()

    def _calculateProperties(self) -> None:

        x: float = self.xlen / 2
        y: float = self.ylen / 2
        h: float = self.hlen / 2
        t: float = self.tlen / 2

        light1Pos = (-x, y, t)
        light2Pos = (x, y, t)
        light3Pos = (-x, -y, t)
        light4Pos = (x, -y, t)

        targetPos = (0, 0, 5)
        light1Dir = calculate_direction(light1Pos, targetPos)
        light2Dir = calculate_direction(light2Pos, targetPos)
        light3Dir = calculate_direction(light3Pos, targetPos)
        light4Dir = calculate_direction(light4Pos, targetPos)

        lightDiffuse = (0.3, 0.3, 0.3)

        self.props: dict[str, str] = {
            "ground_name": f"{self.name}",
            "board_size": f"{x} {y} {t}",

            "wall1_size": f"{t} {y} {h}",
            "wall1_pos": f"{-(x + t)} 0 {h}",

            "wall2_size": f"{t} {y} {h}",
            "wall2_pos": f"{x + t} 0 {h}",

            "wall3_size": f"{x} {t} {h}",
            "wall3_pos": f"0 {y + t} {h}",

            "wall4_size": f"{x} {t} {h}",
            "wall4_pos": f"0 {-(y + t)} {h}",

            "top_pos": f"0 0 {self.hlen}",

            "light_dffuse": f"{lightDiffuse[0]} {lightDiffuse[1]} {lightDiffuse[2]}",

            "light1_pos": f"{light1Pos[0]} {light1Pos[1]} {light1Pos[2]}",
            "light2_pos": f"{light2Pos[0]} {light2Pos[1]} {light2Pos[2]}",
            "light3_pos": f"{light3Pos[0]} {light3Pos[1]} {light3Pos[2]}",
            "light4_pos": f"{light4Pos[0]} {light4Pos[1]} {light4Pos[2]}",

            "light1_dir": f"{light1Dir[0]} {light1Dir[1]} {light1Dir[2]}",
            "light2_dir": f"{light2Dir[0]} {light2Dir[1]} {light2Dir[2]}",
            "light3_dir": f"{light3Dir[0]} {light3Dir[1]} {light3Dir[2]}",
            "light4_dir": f"{light4Dir[0]} {light4Dir[1]} {light4Dir[2]}",
        }

    def generateNodes(self) -> dict:
        return super().generateNodes()

    def attachToMujoco(self, mujocoNode: ET.Element) -> None:
        nodeDict = self.generateNodes()
        mujocoNode.insert(0, nodeDict['asset'])

        worldBodyNode: ET.Element = mujocoNode.find("worldbody")
        worldBodyNode.append(nodeDict['ground'])
        worldBodyNode.append(nodeDict['wall1'])
        worldBodyNode.append(nodeDict['wall2'])
        worldBodyNode.append(nodeDict['wall3'])
        worldBodyNode.append(nodeDict['wall4'])
        worldBodyNode.append(nodeDict['top'])
        worldBodyNode.append(nodeDict['mainLight'])
        worldBodyNode.append(nodeDict['light1'])
        worldBodyNode.append(nodeDict['light2'])
        worldBodyNode.append(nodeDict['light3'])
        worldBodyNode.append(nodeDict['light4'])
