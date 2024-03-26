import math
from typing import Dict, Tuple
import xml.etree.ElementTree as ET
from GeneratorBaseClass import BaseGenerator


class TargetGenerator(BaseGenerator):
    TEMPLATES = {
        "targetAsset": """\
            <asset>
                <material name="target_material" rgba="1 1 1 1"/>
            </asset>""",

        "targetBody": """\
                <body name="target_space" pos="{target_pos}">
                    <site name="parking_spot_center" type="cylinder" size="0.1 0.001" material="target_material"/>
                    <geom name="parking_area" type="box" size="{target_size}" pos="0 0 0" friction="1.0 0.005 0.0001" material="target_material" rgba="0 0 0 0" />
                    <geom name="parking_edge_1" type="box" size="{parking_edge_1_size}" pos="{parking_edge_1_pos}" friction="1.0 0.005 0.0001" material="target_material"/>
                    <geom name="parking_edge_2" type="box" size="{parking_edge_2_size}" pos="{parking_edge_2_pos}" friction="1.0 0.005 0.0001" material="target_material"/>
                    <geom name="parking_edge_3" type="box" size="{parking_edge_3_size}" pos="{parking_edge_3_pos}" friction="1.0 0.005 0.0001" material="target_material"/>
                    <geom name="parking_edge_4" type="box" size="{parking_edge_4_size}" pos="{parking_edge_4_pos}" friction="1.0 0.005 0.0001" material="target_material"/>
                </body>"""
    }

    def __init__(self,
                 carSize,
                 targetPos=(0, 0, 0),
                 targetPaddings=(1.4, 1.7),
                 lineWidth=0.2
                 ):
        self.carSize = carSize
        self.targetPaddings = targetPaddings
        self.targetPos = targetPos
        self.lineWidth = lineWidth
        self._calculateProperties()

    def _calculateProperties(self) -> None:
        targetXSize = self.carSize[0]/2*self.targetPaddings[0]
        targetYSize = self.carSize[1]/2*self.targetPaddings[1]
        lineWidthSize = self.lineWidth/2
        lineHeightSize = 0.001

        self.props = {
            "target_pos": f"{self.targetPos[0]} {self.targetPos[1]} {self.targetPos[2]}",
            "target_size": f"{targetXSize} {targetYSize} {lineHeightSize}",

            "parking_edge_1_size": f"{lineWidthSize} {targetYSize} {lineHeightSize}",
            "parking_edge_1_pos": f"{targetXSize - lineWidthSize} 0 0",

            "parking_edge_2_size": f"{targetXSize} {lineWidthSize} {lineHeightSize}",
            "parking_edge_2_pos": f"0 {targetYSize - lineWidthSize} 0",

            "parking_edge_3_size": f"{targetXSize} {lineWidthSize} {lineHeightSize}",
            "parking_edge_3_pos": f"0 {-(targetYSize - lineWidthSize)} 0",

            "parking_edge_4_size": f"{lineWidthSize} {targetYSize} {lineHeightSize}",
            "parking_edge_4_pos": f"{-(targetXSize - lineWidthSize)} 0 0",
        }

    def attachToMujoco(self, mujocoNode: ET.Element) -> None:
        nodesDict = self.generateNodes()
        mujocoNode.find('worldbody').append(nodesDict['targetBody'])
        mujocoNode.insert(0, nodesDict['targetAsset'])

    def generateNodes(self) -> Dict[str, ET.Element]:
        return super().generateNodes()
