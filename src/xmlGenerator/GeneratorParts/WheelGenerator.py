import xml.etree.ElementTree as ET
from GeneratorBaseClass import BaseGenerator


class WheelGenerator(BaseGenerator):
    TEMPLATES = {
        "wheelNode": """\
        <body name="{wheel_name}" pos="{wheel_pos}" zaxis="0 1 0">
            <joint name="{wheel_name}_joint_steer" type="hinge" axis="0 1 0" limited="true" range="-45 45"/>
            <joint name="{wheel_name}_joint_roll"/>
            <geom type="cylinder" size="{wheel_size}" rgba=".5 .5 1 1" mass="{wheel_mass}" friction="{wheel_friction}"/>
        </body>"""}

    def __init__(self,
                 wheel_name="default_wheel_name",
                 wheel_pos="0 0 0",
                 wheel_size="0.5 0.2",
                 wheel_mass="30",
                 wheel_friction="0.5") -> None:
        """
        Wheel generator class.

        Args:
            wheel_name (str, optional): Wheel name. Defaults to "default_wheel_name".
            wheel_pos (str, optional): Wheel position[m]. Defaults to "0 0 0".
            wheel_size (str, optional): Wheel size ~ Radius, Thickness [m]. Defaults to "0.5 0.2".
            wheel_mass (str, optional): Wheel mass[kg]. Defaults to "30".
            wheel_friction (str, optional): Wheel friction. Defaults to "0.5".
        """
        super().__init__()
        self.wheelName = wheel_name
        self.wheelPos = wheel_pos
        self.wheelSize = wheel_size
        self.wheelMass = wheel_mass
        self.wheelFriction = wheel_friction
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

    def wheel_friction(self, atr):
        self.wheelFriction = atr
        return self

    def _calculateProperties(self):
        self.props = {
            "wheel_name": self.wheelName,
            "wheel_pos": self.wheelPos,
            "wheel_size": self.wheelSize,
            "wheel_mass": self.wheelMass,
            "wheel_friction": self.wheelFriction
        }
        return self

    def generateNodes(self) -> dict:
        return super().generateNodes()

    def attachToMujoco(self, mujocoNode: ET.Element):
        """
        Wheel should be only attached in CarGenerator.
        """
        raise NotImplementedError


if __name__ == "__main__":
    whl = WheelGenerator()
    a, = whl.generateNodes()
    ET.dump(a)
