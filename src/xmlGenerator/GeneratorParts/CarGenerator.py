import xml.etree.ElementTree as ET
from GeneratorBaseClass import BaseGenerator
from WheelGenerator import WheelGenerator
from typing import Dict, Tuple


class CarGenerator(BaseGenerator):
    TEMPLATES: dict[str, str] = {
        "carBody": """\
        <body name="{car_name}" pos="{car_pos}">
            <freejoint/>
            <body name="{car_name}_front_lights"  pos="{car_front_lights_pos}"></body>
            <body name="{car_name}_chassis">
                <geom name="{car_name}_chassis_geom" type="box" size="{car_size}" mass="{car_mass}"/>
                <!-- WHEELS HERE -->
            </body>
        </body>"""
    }

    def __init__(
        self,
        carName: str,
        chassisLength: float,
        chassisWidth: float,
        chassisHeight: float,
        carMass: float = 1500,
        wheelRadius: float = 0.5,
        wheelThickness: float = 0.2,
        wheelMass: float = 30,
        wheelFriction: Tuple[float, float, float] = (1, 1e-3, 1e-3),
        wheelAxisSpacing: float = 0.6,
        wheelSpacing: float = 1,
        wheelMountHeight: float = 0,
        lightsSpacing: float = 0.6,
    ) -> None:
        """
        Car Generator class.

        Args:
            carName (str): Car name
            chassisLength (float): length [m]
            chassisWidth (float): width [m]
            chassisHeight (float): height [m]
            carMass (float, optional): Car mass [kg]. Defaults to 1500kg.

            wheelRadius (float, optional): Car wheel/tire radius [m]. Defaults to 0.5m.
            wheelThickness (float, optional): "wheel/tire thickness [m]. Defaults to 0.2.
            wheelMass (float, optional): wheel/tire mass [kg]. Defaults to 30.
            wheelFriction (Tuple[float, float, float], optional): Sliding, Torsional, Rolling friction. Defaults to (1e3, 1e-3, 1e-3).

            wheelAxisSpacing (float, optional): spacing of axis from center to edge, range: 0 to 1. Defaults to 0.6.
            wheelSpacing (float, optional): Distance between wheels, range: 0 to 1. Defaults to 1.
            wheelMountHeight (float, optional): range: -1 to 1. Defaults to 0.
            lightsSpacing (float, optional): range: 0 to 1. Defaults to 0.6.
        """
        super().__init__()
        self.carName: str = carName
        self.chassisLength: float = chassisLength
        self.chassisWidth: float = chassisWidth
        self.chassisHeight: float = chassisHeight
        self.carMass: float = carMass
        self.wheelRadius: float = wheelRadius
        self.wheelThickness: float = wheelThickness
        self.wheelMass: float = wheelMass
        self.wheelFriction: Tuple[float, float, float] = wheelFriction
        self.wheelAxisSpacing: float = wheelAxisSpacing
        self.wheelSpacing: float = wheelSpacing
        self.wheelMountHeight: float = wheelMountHeight
        self.lightsSpacing: float = lightsSpacing
        self._calculateProperties()

    def _calculateProperties(self) -> None:
        """
        Calculate internal properties based on properties provided in constructor. Call always after changing any property.
        """
        chassisXSize = self.chassisLength / 2
        chassisYsize = self.chassisWidth / 2
        chassisZsize = self.chassisHeight / 2

        self.props = {
            "car_name": f"{self.carName}",
            "car_pos": "0 0 5",
            "car_size": f"{chassisXSize} {chassisYsize} {chassisZsize}",
            "car_mass": f"{self.carMass}",
            "car_wheel_size": (self.wheelRadius, self.wheelThickness / 2),
            "car_wheel_mass": self.wheelMass,
            "car_wheel_friction": (self.wheelFriction[0], self.wheelFriction[1], self.wheelFriction[2]),
            "left_front_wheel_pos": (chassisXSize * self.wheelAxisSpacing, -chassisYsize * self.wheelSpacing, self.chassisHeight * self.wheelMountHeight),
            "right_front_wheel_pos": (chassisXSize * self.wheelAxisSpacing, chassisYsize * self.wheelSpacing, self.chassisHeight * self.wheelMountHeight),
            "left_back_wheel_pos": (-chassisXSize * self.wheelAxisSpacing, -chassisYsize * self.wheelSpacing, self.chassisHeight * self.wheelMountHeight),
            "right_back_wheel_pos": (-chassisXSize * self.wheelAxisSpacing, chassisYsize * self.wheelSpacing, self.chassisHeight * self.wheelMountHeight),
            "car_front_lights_pos": f"{chassisXSize} 0 0",
            "car_front_right_light_pos": f"0 {-chassisYsize * self.lightsSpacing} 0",
            "car_front_left_light_pos": f"0 {chassisYsize * self.lightsSpacing} 0",
        }

    def generateNodes(self) -> Dict[str, ET.Element]:
        """
        Generate Class nodes from Pattern.

        Returns:
            list: List of nodes.
        """
        nodeDict: Dict[str, ET.Element] = super().generateNodes()

        carChassis: ET.Element = nodeDict["carBody"].find(
            f".//body[@name='{self.carName}_chassis']")

        wheelGen = WheelGenerator(
            wheel_mass=self.props["car_wheel_mass"],
            wheel_size=self.props["car_wheel_size"],
            wheel_friction=self.props["car_wheel_friction"],
        )

        carChassis.append(
            *wheelGen
            .with_wheelName(f"{self.carName}_wheel_front_left")
            .with_wheelPos(self.props["left_front_wheel_pos"])
            .with_isSteering(True)
            ._calculateProperties()
            .generateNodes()
            .values()
        )
        carChassis.append(
            *wheelGen
            .with_wheelName(f"{self.carName}_wheel_front_right")
            .with_wheelPos(self.props["right_front_wheel_pos"])
            .with_isSteering(True)
            ._calculateProperties()
            .generateNodes()
            .values()
        )
        carChassis.append(
            *wheelGen
            .with_wheelName(f"{self.carName}_wheel_back_left")
            .with_wheelPos(self.props["left_back_wheel_pos"])
            .with_isSteering(False)
            ._calculateProperties()
            .generateNodes()
            .values()
        )
        carChassis.append(
            *wheelGen
            .with_wheelName(f"{self.carName}_wheel_back_right")
            .with_wheelPos(self.props["right_back_wheel_pos"])
            .with_isSteering(False)
            ._calculateProperties()
            .generateNodes()
            .values()
        )

        return nodeDict

    def attachToMujoco(self, mujocoNode: ET.Element) -> None:
        carBody, = self.generateNodes().values()
        worldBody = mujocoNode.find("worldbody")
        worldBody.append(carBody)


if __name__ == "__main__":
    car = CarGenerator("car1",
                       2,
                       1,
                       0.5)
    a, = car.generateNodes()
    ET.dump(a)
