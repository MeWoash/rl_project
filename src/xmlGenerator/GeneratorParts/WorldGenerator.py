from typing import Any
import xml.etree.ElementTree as ET
from GeneratorBaseClass import BaseGenerator


class WorldGenerator(BaseGenerator):
    TEMPLATES: dict[str, str] = {
        "texture": """\
            <asset>
                <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
                <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0"/>
            </asset>""",
        "ground": '<geom name="{ground_name}_bottom" type="plane" size="{board_size}" material="grid" friction="1.0 0.005 0.0001"/>',
        "wall1": '<geom name="{ground_name}_left" type="box" size="{left_size}" pos="{left_pos}" material="grid"/>',
        "wall2": '<geom name="{ground_name}_right" type="box" size="{right_size}" pos="{right_pos}" material="grid"/>',
        "wall3": '<geom name="{ground_name}_front" type="box" size="{front_size}" pos="{front_pos}" material="grid"/>',
        "wall4": '<geom name="{ground_name}_back" type="box" size="{back_size}" pos="{back_pos}" material="grid"/>'
    }

    def __init__(self,
                 name: str,
                 xlen: float,
                 ylen: float,
                 hlen: float,
                 tlen: float = 0.1) -> None:
        super().__init__()
        self.name: str = name
        self.x: float = xlen / 2
        self.y: float = ylen / 2
        self.h: float = hlen / 2
        self.t: float = tlen
        self._calculateProperties()

    def _calculateProperties(self) -> None:
        self.props: dict[str, str] = {
            "ground_name": f"{self.name}",
            "board_size": f"{self.x} {self.y} {self.t}",
            "left_size": f"{self.t} {self.y} {self.h}",
            "left_pos": f"{-(self.x + self.t)} 0 {self.h}",
            "right_size": f"{self.t} {self.y} {self.h}",
            "right_pos": f"{self.x + self.t} 0 {self.h}",
            "front_size": f"{self.x} {self.t} {self.h}",
            "front_pos": f"0 {self.y + self.t} {self.h}",
            "back_size": f"{self.x} {self.t} {self.h}",
            "back_pos": f"0 {-(self.y + self.t)} {self.h}",
        }

    def generateNodes(self) -> dict:
        return super().generateNodes()

    def attachToMujoco(self, mujocoNode: ET.Element) -> None:
        nodeDict = self.generateNodes()
        mujocoNode.append(nodeDict['texture'])

        worldBodyNode: ET.Element = mujocoNode.find("worldbody")
        worldBodyNode.append(nodeDict['ground'])
        worldBodyNode.append(nodeDict['wall1'])
        worldBodyNode.append(nodeDict['wall2'])
        worldBodyNode.append(nodeDict['wall3'])
        worldBodyNode.append(nodeDict['wall4'])
