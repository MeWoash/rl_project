from typing import Tuple
import xml.etree.ElementTree as ET
from GeneratorBaseClass import BaseGenerator

# reflectance
# shininess - matte -> glossy look
# emission - light emited by material
# specular - mirror-like reflection of light sources


class WheelGenerator(BaseGenerator):
    TEMPLATES = {
        "wheelAsset": """\
            <asset>
                <material name="wheel_material" rgba="{wheel_color}" reflectance="0" shininess="0" emission="0" specular="0.1"/>
            </asset>""",

        "wheelNode": """\
            <body name="{wheel_name}" pos="{wheel_pos}" zaxis="0 1 0">
                <joint name="{wheel_name}_joint_roll"/>
                <geom type="cylinder" size="{wheel_size}" material="wheel_material" mass="{wheel_mass}" friction="{wheel_friction}"/>
            </body>""",

        "steeringPart": """
            <joint name="{wheel_name}_joint_steer" type="hinge" axis="0 1 0" limited="true" range="{wheel_angle_limit}"/>"""
    }

    def __init__(self,
                 wheel_name="default_wheel_name",
                 wheel_pos=(0, 0, 0),
                 wheel_size=(0.5, 0.2),
                 wheel_mass=30,
                 wheel_friction=(1, 1e-3, 1e-3),
                 is_steering=False,
                 wheel_angle_limit=(-45, 45)) -> None:
        """
        Wheel generator class.

        Args:
            wheel_name (str, optional): Wheel name. Defaults to "default_wheel_name".
            wheel_pos (tuple, optional): Wheel position[m]. Defaults to (0, 0, 0).
            wheel_size (tuple, optional):Wheel size ~ Radius, Thickness [m]. Defaults to (0.5 0.2).
            wheel_mass (int, optional): Wheel mass[kg]. Defaults to 30.
            wheel_friction (tuple, optional): Wheel friction. Defaults to (1, 1e-3, 1e-3).
            is_steering (bool, optional): Is Steering wheel. Defaults to False.
            wheel_angle_limit (tuple, optional): wheelAngleLimit. Defaults to (-45, 45) degrees.
        """
        super().__init__()
        self.wheelName = wheel_name
        self.wheelPos = wheel_pos
        self.wheelSize = wheel_size
        self.wheelMass = wheel_mass
        self.wheelFriction = wheel_friction
        self.isSteering = is_steering
        self.wheelAngleLimit = wheel_angle_limit
        self.wheelColor = (0, 0, 0, 1)
        self._calculateProperties()

    def with_wheelName(self, atr):
        self.wheelName = atr
        return self

    def with_wheelPos(self, atr):
        self.wheelPos = atr
        return self

    def with_wheelSize(self, atr):
        self.wheelSize = atr
        return self

    def with_wheelMass(self, atr):
        self.wheelMass = atr
        return self

    def with_wheelFriction(self, atr):
        self.wheelFriction = atr
        return self

    def with_isSteering(self, atr):
        self.isSteering = atr
        return self

    def _calculateProperties(self):
        self.props = {
            "wheel_name": self.wheelName,
            "wheel_pos": f"{self.wheelPos[0]} {self.wheelPos[1]} {self.wheelPos[2]}",
            "wheel_size": f"{self.wheelSize[0]} {self.wheelSize[1]}",
            "wheel_mass": f"{self.wheelMass}",
            "wheel_friction": f"{self.wheelFriction[0]} {self.wheelFriction[1]} {self.wheelFriction[2]}",
            "wheel_angle_limit": f"{self.wheelAngleLimit[0]} {self.wheelAngleLimit[1]}",
            "wheel_color": f"{self.wheelColor[0]} {self.wheelColor[1]} {self.wheelColor[2]} {self.wheelColor[3]}"
        }
        return self

    def generateNodes(self) -> dict:
        nodesDict = super().generateNodes()

        if self.isSteering:
            nodesDict['wheelNode'].insert(0, nodesDict['steeringPart'])

        del nodesDict['steeringPart']
        return nodesDict

    def attachToMujoco(self, mujocoNode: ET.Element):
        """
        Wheel should be only attached in CarGenerator.
        """
        raise NotImplementedError


if __name__ == "__main__":
    whl = WheelGenerator()
    a, = whl.generateNodes()
    ET.dump(a)
