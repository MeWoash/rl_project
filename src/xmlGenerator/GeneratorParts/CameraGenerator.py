import math
from typing import Dict, Tuple
import xml.etree.ElementTree as ET
from GeneratorBaseClass import BaseGenerator


def calculateCameraHeight(x, y, fov_y_degrees):
    fov_y_radians = math.radians(fov_y_degrees)
    h = max(x, y) / 2*math.tan(fov_y_radians / 2)
    return h


class CameraGenerator(BaseGenerator):
    TEMPLATES = {
        "cameraNode": """<camera name="{cam_name}" pos="{cam_pos}" euler="{cam_euler}" fovy="{cam_fov}"/>"""
    }

    def __init__(self, cameraName: str, xlen: float, ylen: float, camFov=90):
        self.cameraName = cameraName
        self.xlen = xlen
        self.ylen = ylen
        self.camFov = camFov
        self._calculateProperties()

    def _calculateProperties(self) -> None:
        camHeight = calculateCameraHeight(self.xlen, self.ylen, self.camFov)
        self.props = {
            "cam_name": f"{self.cameraName}",
            "cam_pos": f"0 0 {camHeight}",
            "cam_euler": "0 0 -90",
            "cam_fov": f"{self.camFov}"
        }

    def attachToMujoco(self, mujocoNode: ET.Element) -> None:
        nodesDict = self.generateNodes()
        mujocoNode.find('worldbody').append(nodesDict['cameraNode'])

    def generateNodes(self) -> Dict[str, ET.Element]:
        return super().generateNodes()
