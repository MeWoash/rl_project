import math
from platform import node
import xml.etree.ElementTree as ET
from GeneratorBaseClass import BaseGenerator
from WheelGenerator import WheelGenerator
from typing import Dict, Tuple

import numpy as np


class CarGenerator(BaseGenerator):
    TEMPLATES: dict[str, str] = {

        "asset": """\
            <asset>
                <material name="{car_name}_chassis_material" rgba="{car_chassis_color}" reflectance="0" shininess="1" emission="0.0" specular="0.0"/>
            </asset>""",

        "carBody": """\
        <body name="{car_name}" pos="{car_pos}">
            <freejoint/>
            <site name="{car_name}_center" type="cylinder" size="0.1 0.001"/>
            <body name="{car_name}_front_lights"  pos="{car_front_lights_pos}"></body>
            <body name="{car_name}_chassis">
                <geom name="{car_name}_chassis_geom" material="{car_name}_chassis_material" type="box" size="{car_size}" mass="{car_mass}"/>
                <site name="{car_name}_chassis_site" type="box" size="{car_size}" rgba="0 0 0 0"/>
                <!-- WHEELS HERE -->
            </body>
        </body>""",

        "backWheelsTendon": """\
            <tendon>
                <fixed name="{car_name}_back_wheels_tendon">
                    <joint joint="{car_name}_wheel3_joint_roll" coef="1000" />
                    <joint joint="{car_name}_wheel4_joint_roll" coef="1000" />
                </fixed>
            </tendon>""",

        "carControls": """\
            <actuator>
                <motor name="{car_name}_engine_power" tendon="{car_name}_back_wheels_tendon" ctrlrange="-1 1" />
                <position name="{car_name}_wheel1_angle" joint="{car_name}_wheel1_joint_steer" kp="1000" ctrlrange="{wheel_control_angle_range}"/>
                <position name="{car_name}_wheel2_angle" joint="{car_name}_wheel2_joint_steer" kp="1000" ctrlrange="{wheel_control_angle_range}"/>    
            </actuator>""",

        "carSensors": """\
            <sensor>
                <touch name="{car_name}_chassis_touch_sensor" site="{car_name}_chassis_site" cutoff="{max_sensor_val}"/>
                <velocimeter name="{car_name}_speed_sensor" site="{car_name}_center" cutoff="10"/>
                <framepos name="{car_name}_posGlobal_sensor" objtype="site" objname="{car_name}_center"/>
                <framepos name="{car_name}_posTarget_sensor" objtype="site" objname="{car_name}_center" reftype="site" refname="parking_spot_center"/>
            </sensor>"""
    }

    def __init__(
        self,
        carName: str,
        chassisLength: float,
        chassisWidth: float,
        chassisHeight: float,
        carPosition=(0, 0, 2),
        carMass: float = 1500,
        wheelRadius: float = 0.3,
        wheelThickness: float = 0.2,
        wheelMass: float = 30,
        wheelFriction: Tuple[float, float, float] = (1, 1e-3, 1e-3),
        wheelAngleLimit: Tuple[float, float, float] = (-45, 45),
        wheelAxisSpacing: float = 0.6,
        wheelSpacing: float = 1,
        wheelMountHeight: float = -0.25,
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
        self.carPosition = carPosition
        self.carMass: float = carMass
        self.wheelRadius: float = wheelRadius
        self.wheelThickness: float = wheelThickness
        self.wheelMass: float = wheelMass
        self.wheelFriction: Tuple[float, float, float] = wheelFriction
        self.wheelAngleLimit: Tuple[float, float] = wheelAngleLimit
        self.wheelAxisSpacing: float = wheelAxisSpacing
        self.wheelSpacing: float = wheelSpacing
        self.wheelMountHeight: float = wheelMountHeight
        self.lightsSpacing: float = lightsSpacing
        self.carChassisColorRGBA = (0.8, 0.102, 0.063, 1)
        self.maxSensorVal = 5
        self._calculateProperties()

    def _calculateProperties(self) -> None:
        """
        Calculate internal properties based on properties provided in constructor. Call always after changing any property.
        """
        chassisXSize = self.chassisLength / 2
        chassisYsize = self.chassisWidth / 2
        chassisZsize = self.chassisHeight / 2
        wheelControlRange = (math.radians(
            self.wheelAngleLimit[0]), math.radians(self.wheelAngleLimit[1]))

        self.props = {
            "car_name": f"{self.carName}",
            "car_pos": f"{self.carPosition[0]} {self.carPosition[1]} {self.carPosition[2]}",
            "car_size": f"{chassisXSize} {chassisYsize} {chassisZsize}",
            "car_mass": f"{self.carMass}",
            "car_wheel_size": (self.wheelRadius, self.wheelThickness / 2),
            "car_wheel_mass": self.wheelMass,
            "car_wheel_friction": self.wheelFriction,
            "car_wheel_angle_limit": self.wheelAngleLimit,
            "car_chassis_color": f"{self.carChassisColorRGBA[0]} {self.carChassisColorRGBA[1]} {self.carChassisColorRGBA[2]} {self.carChassisColorRGBA[3]}",
            "wheel1_pos": (chassisXSize * self.wheelAxisSpacing, -chassisYsize * self.wheelSpacing, self.chassisHeight * self.wheelMountHeight),
            "wheel2_pos": (chassisXSize * self.wheelAxisSpacing, chassisYsize * self.wheelSpacing, self.chassisHeight * self.wheelMountHeight),
            "wheel3_pos": (-chassisXSize * self.wheelAxisSpacing, -chassisYsize * self.wheelSpacing, self.chassisHeight * self.wheelMountHeight),
            "wheel4_pos": (-chassisXSize * self.wheelAxisSpacing, chassisYsize * self.wheelSpacing, self.chassisHeight * self.wheelMountHeight),
            "car_front_lights_pos": f"{chassisXSize} 0 0",
            "car_front_right_light_pos": f"0 {-chassisYsize * self.lightsSpacing} 0",
            "car_front_left_light_pos": f"0 {chassisYsize * self.lightsSpacing} 0",
            "wheel_control_angle_range": f"{wheelControlRange[0]} {wheelControlRange[1]}",
            "max_sensor_val": f"{self.maxSensorVal}"
        }

    def generateNodes(self) -> Dict[str, ET.Element]:
        """
        Generate Class nodes from Pattern.

        Returns:
            list: List of nodes.
        """
        nodeDict: Dict[str, ET.Element] = super().generateNodes()
        self.addWheels(nodeDict)
        self.addRangeSensors(nodeDict)

        return nodeDict

    def addRangeSensors(self, nodeDict):
        carBody: ET.Element = nodeDict["carBody"]
        carSensors: ET.Element = nodeDict["carSensors"]

        chassisXSize = self.chassisLength / 2
        chassisYsize = self.chassisWidth / 2
        chassisZsize = self.chassisHeight / 2
        sensor_size = (1e-6, 1e-6)
        sensor_pos_scale_factor = 1e-3

        zaxis = np.array([
            [chassisXSize, chassisYsize, 0],
            [chassisXSize, 0, 0],
            [chassisXSize, -chassisYsize, 0],
            [-chassisXSize, chassisYsize, 0],
            [-chassisXSize, 0, 0],
            [-chassisXSize, -chassisYsize, 0],
            [0, chassisYsize, 0],
            [0, -chassisYsize, 0],

        ])

        for index, _zaxis in enumerate(zaxis):
            _zaxis = _zaxis + _zaxis*sensor_pos_scale_factor
            sensor_site_name = f"{self.carName}_sensor_site_{index}"
            sensor_site = ET.fromstring(
                f"""<site name="{sensor_site_name}" type="sphere" size="{sensor_size[0]} {sensor_size[1]}"
                pos="{_zaxis[0]} {_zaxis[1]} {_zaxis[2]}"
                zaxis="{_zaxis[0]} {_zaxis[1]} {_zaxis[2]}"/>""")

            sensor = ET.fromstring(
                f"""<rangefinder name="{self.carName}_sensor_{index}" site="{sensor_site_name}" cutoff="{self.maxSensorVal}"/>""")

            carBody.append(sensor_site)
            carSensors.append(sensor)

    def addWheels(self, nodeDict):
        carChassis: ET.Element = nodeDict["carBody"].find(
            f".//body[@name='{self.carName}_chassis']")

        wheelGen = WheelGenerator(
            wheel_mass=self.props["car_wheel_mass"],
            wheel_size=self.props["car_wheel_size"],
            wheel_friction=self.props["car_wheel_friction"],
            wheel_angle_limit=self.props["car_wheel_angle_limit"]
        )

        carChassis.append(
            wheelGen
            .with_wheelName(f"{self.carName}_wheel1")
            .with_wheelPos(self.props["wheel1_pos"])
            .with_isSteering(True)
            ._calculateProperties()
            .generateNodes()['wheelNode']
        )
        carChassis.append(
            wheelGen
            .with_wheelName(f"{self.carName}_wheel2")
            .with_wheelPos(self.props["wheel2_pos"])
            .with_isSteering(True)
            ._calculateProperties()
            .generateNodes()['wheelNode']
        )
        carChassis.append(
            wheelGen
            .with_wheelName(f"{self.carName}_wheel3")
            .with_wheelPos(self.props["wheel3_pos"])
            .with_isSteering(False)
            ._calculateProperties()
            .generateNodes()['wheelNode']
        )
        carChassis.append(
            wheelGen
            .with_wheelName(f"{self.carName}_wheel4")
            .with_wheelPos(self.props["wheel4_pos"])
            .with_isSteering(False)
            ._calculateProperties()
            .generateNodes()['wheelNode']
        )
        nodeDict['wheelAsset'] = wheelGen.generateNodes()['wheelAsset']

    def attachToMujoco(self, mujocoNode: ET.Element) -> None:
        nodesDict: Dict[str, ET.Element] = self.generateNodes()
        mujocoNode.find("worldbody").append(nodesDict['carBody'])
        mujocoNode.append(nodesDict['backWheelsTendon'])
        mujocoNode.append(nodesDict['carControls'])
        mujocoNode.append(nodesDict['carSensors'])
        mujocoNode.insert(0, nodesDict["asset"])
        mujocoNode.insert(0, nodesDict["wheelAsset"])


if __name__ == "__main__":
    car = CarGenerator("car1",
                       2,
                       1,
                       0.5)
    a, = car.generateNodes()
    ET.dump(a)
