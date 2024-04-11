import math
from typing import Tuple
from dm_control import mjcf
from dm_control import viewer
from pathlib import Path

from networkx import radius

import Globals
import numpy as np

ASSET_DIR = Globals.ASSETS_DIR


def calculateCameraHeight(x, y, fov_y_degrees):
    fov_y_radians = math.radians(fov_y_degrees)
    h = max(x, y) / 2*math.tan(fov_y_radians / 2)
    return h


class Wheel:
    _wheel_mass = 30
    _wheel_friction = (1, 1e-3, 1e-3)
    _wheel_angle_limit = (-45, 45)
    _wheelRadius: float = 0.3
    _wheelThickness: float = 0.2

    def __init__(self, wheelName) -> None:
        self._wheelName = wheelName
        self._is_steering = False,

    @property
    def is_steering(self):
        return self._is_steering

    @is_steering.setter
    def is_steering(self, val):
        self._is_steering = val

    @property
    def wheel_name(self):
        return self._is_steering

    @wheel_name.setter
    def wheel_name(self, val):
        self._wheelName = val

    def construct_tree(self, name=None, is_steering=None):
        if name is not None:
            self._wheelName = name
        if is_steering is not None:
            self._is_steering = is_steering

        mjcf_model = mjcf.RootElement(model=self._wheelName)
        steering_joint = None
        rolling_joint = None

        wheelMaterial = mjcf_model.asset.add("material", name="material", rgba=[0, 0, 0, 1],
                                             reflectance=0, shininess=0, emission=0, specular=0.1)
        wheelBody = mjcf_model.worldbody.add(
            "body", name="body", zaxis=[0, 1, 0])

        wheelBody.add("geom", type="cylinder", size=[
                      self._wheelRadius, self._wheelThickness / 2], mass=self._wheel_mass, friction=self._wheel_friction, material=wheelMaterial)

        if self._is_steering:
            steering_joint = wheelBody.add("joint", name="steering_joint", type="hinge", axis=[
                                           0, 1, 0], limited=True, range=self._wheel_angle_limit)

        rolling_joint = wheelBody.add("joint", name="rolling_joint")

        return mjcf_model, rolling_joint, steering_joint


class Car:
    _wheelFriction: Tuple[float, float, float] = (1, 1e-3, 1e-3)
    _wheelAxisSpacing: float = 0.6
    _wheelSpacing: float = 1
    _wheelMountHeight: float = -0.25
    _carMass: float = 1500
    _wheelMass: float = 30
    _maxSensorVal: int = 5
    _carChassisColorRGBA = (0.8, 0.102, 0.063, 1)

    def __init__(self, carName, carDims) -> None:
        self._carName: str = carName
        self._carDims = carDims
        self.wheelGenerator = Wheel("Wheel")

    def construct_tree(self):
        mjcf_model = mjcf.RootElement(model=self._carName)

        chassisXSize = self._carDims[0] / 2
        chassisYsize = self._carDims[1] / 2
        chassisZsize = self._carDims[2] / 2

        wheelControlRange = (math.radians(
            Wheel._wheel_angle_limit[0]), math.radians(Wheel._wheel_angle_limit[1]))

        # autopep8: off
        material = mjcf_model.asset.add("material", name="chassis_material",
            rgba=self._carChassisColorRGBA,
            reflectance=0,
            shininess=1,
            emission=0.0,
            specular=0.0)

        self.center_site=mjcf_model.worldbody.add("site", name="site_center")
        mjcf_model.worldbody.add("geom", type="box", size=[chassisXSize,chassisYsize,chassisZsize], mass=self._carMass, material=material)
        touch_site = mjcf_model.worldbody.add("site", type="box", size=[chassisXSize,chassisYsize,chassisZsize], rgba=[0, 0, 0, 0])
        
        # ADD WHEELS
        s1=mjcf_model.worldbody.add("site", name="wheel1_attachment_site", pos=[chassisXSize * self._wheelAxisSpacing, -chassisYsize * self._wheelSpacing, self._carDims[2] * self._wheelMountHeight])
        s2=mjcf_model.worldbody.add("site", name="wheel2_attachment_site", pos=[chassisXSize * self._wheelAxisSpacing, chassisYsize * self._wheelSpacing, self._carDims[2] * self._wheelMountHeight])
        s3=mjcf_model.worldbody.add("site", name="wheel3_attachment_site", pos=[-chassisXSize * self._wheelAxisSpacing, -chassisYsize * self._wheelSpacing, self._carDims[2] * self._wheelMountHeight])
        s4=mjcf_model.worldbody.add("site", name="wheel4_attachment_site", pos=[-chassisXSize * self._wheelAxisSpacing, chassisYsize * self._wheelSpacing, self._carDims[2] * self._wheelMountHeight])

        w1MJCF, w1rolling, w1steering, = self.wheelGenerator.construct_tree("wheel1", True)
        w2MJCF, w2rolling, w2steering, = self.wheelGenerator.construct_tree("wheel2", True)
        w3MJCF, w3rolling, w3steering, = self.wheelGenerator.construct_tree("wheel3", False)
        w4MJCF, w4rolling, w4steering, = self.wheelGenerator.construct_tree("wheel4", False)

        s1.attach(w1MJCF)
        s2.attach(w2MJCF)
        s3.attach(w3MJCF)
        s4.attach(w4MJCF)

        # ADD TENDONS
        tendonFixed = mjcf_model.tendon.add("fixed", name="back_wheels_tendon")
        tendonFixed.add("joint", joint=w3rolling, coef=1000)
        tendonFixed.add("joint", joint=w4rolling, coef=1000)

        #ADD ACTUATORS
        mjcf_model.actuator.add("motor", name="engine", tendon=tendonFixed, ctrlrange=[-1, 1])
        mjcf_model.actuator.add("position", name="wheel1_angle", joint=w1steering, kp=1000, ctrlrange=wheelControlRange)
        mjcf_model.actuator.add("position", name="wheel2_angle", joint=w2steering, kp=1000, ctrlrange=wheelControlRange)

        #ADD SENSORS
        ## RANGE
        zaxis = np.array([
            [chassisXSize, chassisYsize, 0],
            [chassisXSize, 0, 0],
            [chassisXSize, -chassisYsize, 0],
            [-chassisXSize, chassisYsize, 0],
            [-chassisXSize, 0, 0],
            [-chassisXSize, -chassisYsize, 0],
            [0, chassisYsize, 0],
            [0, -chassisYsize, 0]

        ])
        sensor_size = (1e-6, 1e-6)
        sensor_pos_scale_factor = 1e-3
        for index, _zaxis in enumerate(zaxis):
            _zaxis = _zaxis + _zaxis*sensor_pos_scale_factor
            sensor_site = mjcf_model.worldbody.add("site", name=f"sensor_site_{index}", type="sphere", size=sensor_size, pos=_zaxis, zaxis=_zaxis)
            sensor = mjcf_model.sensor.add("rangefinder", name=f"range_sensor_{index}", site=sensor_site, cutoff=self._maxSensorVal)
            
        ## OTHERS
        mjcf_model.sensor.add("touch", name=f"touch_sensor", site=touch_site, cutoff=self._maxSensorVal)
        mjcf_model.sensor.add("velocimeter", name=f"speed_sensor", site=self.center_site, cutoff=10)
        mjcf_model.sensor.add("framepos", name=f"pos_global_sensor", objtype="site", objname=self.center_site)
        
        # autopep8: on
        return mjcf_model


class ParkingSpot:
    _lineHeightSize = 0.001
    _friction = [1.0, 0.005, 0.0001]
    _targetPaddings = (1.4, 1.7)
    _lineWidth = 0.2

    def __init__(self,
                 name,
                 carSize,
                 ):

        self._model_name = name
        self._carSize = carSize

    def construct_tree(self):
        mjcf_model = mjcf.RootElement(model=self._model_name)

        targetXSize = self._carSize[0]/2*self._targetPaddings[0]
        targetYSize = self._carSize[1]/2*self._targetPaddings[1]
        lineWidthSize = self._lineWidth/2
        lineHeightSize = self._lineHeightSize

        # autopep8: off
        material = mjcf_model.asset.add("material", name=f"line_material",rgba=[1, 1, 1, 1])
        self.site_center = mjcf_model.worldbody.add("site",name=f"site_center",  type="cylinder",size=[0.1, 0.001],  material=material)

        mjcf_model.worldbody.add("geom", type="box", size=[lineWidthSize, targetYSize, lineHeightSize], pos=[targetXSize - lineWidthSize, 0, 0],       friction=self._friction, material=material)
        mjcf_model.worldbody.add("geom", type="box", size=[targetXSize, lineWidthSize, lineHeightSize], pos=[0, targetYSize - lineWidthSize, 0],       friction=self._friction, material=material)
        mjcf_model.worldbody.add("geom", type="box", size=[targetXSize, lineWidthSize, lineHeightSize], pos=[ 0, -(targetYSize - lineWidthSize), 0],   friction=self._friction, material=material)
        mjcf_model.worldbody.add("geom", type="box", size=[lineWidthSize, targetYSize, lineHeightSize], pos=[-(targetXSize - lineWidthSize), 0, 0],    friction=self._friction, material=material)
        # autopep8: on
        return mjcf_model


class Generator:
    model_name = "MainModel"
    _carName = "mainCar"
    _spotName = "parkingSpot"
    _offheight = 720
    _offwidth = 1280
    

    def __init__(self):
        # MAP PROPS
        self._map_length = [20, 20, 20, 5]
        self._car_dims = (2, 1, 0.25)
        self._car_pos = (0, 0)
        self._parking_pos = (5, 5, 0)
        
        self.carGenerator = Car(self._carName, self._car_dims)
        self.parkingSpotGenerator = ParkingSpot(self._spotName, self._car_dims)

    @property
    def car_pos(self):
        return self._car_pos

    @car_pos.setter
    def car_pos(self, val):
        self._car_pos = [val[0], val[1], Wheel._wheelRadius +
                         self._car_dims[2] * abs(Car._wheelMountHeight)]

    @property
    def map_length(self):
        return self._map_length

    @map_length.setter
    def map_length(self, val):
        self._mapL_mapLength = val

    @property
    def parking_pos(self):
        return self._parking_pos

    @parking_pos.setter
    def parking_pos(self, val):
        self._parking_pos = [val[0], val[1], 0]

    def construct_tree(self):
        self.mjcf_model = mjcf.RootElement(model=self.model_name)

        # autopep8: off

        # RENDERING
        global_settings = getattr(self.mjcf_model.visual, 'global')
        global_settings.offwidth = self._offwidth
        global_settings.offheight = self._offheight
        
        # MAP
        self._generate_map()

        # CAM
        self._generate_camera()

        # CAR
        self.car_attachment_site = self.mjcf_model.worldbody.add("site", name="car_attachment_site", pos=self._car_pos)
        carMJCF = self.carGenerator.construct_tree()
        a = self.car_attachment_site.attach(carMJCF)
        a.add("freejoint")

        # Parking
        self.parking_attachment_site = self.mjcf_model.worldbody.add("site", name="parking_attachment_site", pos=self._parking_pos)
        parkingSpotMJCF = self.parkingSpotGenerator.construct_tree()
        self.parking_attachment_site.attach(parkingSpotMJCF)
        
        # CONNECT CAR AND PARKING WITH POS SENSOR
        self.mjcf_model.sensor.add("framepos", name=f"{self._carName}_to_{self._spotName}_pos", objtype="site", objname=self.carGenerator.center_site,
                                   reftype="site", refname=self.parkingSpotGenerator.site_center)

        # autopep8: on

    def _generate_map(self):

        x, y, h, t = self.map_length[0]/2, self.map_length[1] / \
            2, self.map_length[2]/2, self.map_length[3]/2

        # autopep8: off
        self.mjcf_model.asset.add( "texture", name="sky_texture", type="skybox", file=ASSET_DIR+"/sky1.png")
        wall_material = self.mjcf_model.asset.add(  "material",     name="wall_material",   rgba=[1, 1, 1, 0.000001])
        ground_texture = self.mjcf_model.asset.add( "texture",      name="ground_texture",  type="2d",              file=ASSET_DIR+"/ground.png")
        ground_material = self.mjcf_model.asset.add("material",     name="ground_material", texture=ground_texture, texrepeat=[25, 25])

        self.mjcf_model.worldbody.add("geom",   name="ground",    type="plane",   size=[x, y, t],   friction= [1.0, 0.005, 0.0001], material=ground_material)
        self.mjcf_model.worldbody.add("geom",   name="ceiling",   type="box",     size=[x, y, t],   pos= [0, 0, self.map_length[2]],rgba= [0, 0, 0, 0])
        self.mjcf_model.worldbody.add("geom",   name="wall1",     type="box",     size=[t, y, h],   pos= [-(x+t), 0, h],            material=wall_material)
        self.mjcf_model.worldbody.add("geom",   name="wall2",     type="box",     size=[t, y, h],   pos= [x+t, 0, h],               material=wall_material)
        self.mjcf_model.worldbody.add("geom",   name="wall3",     type="box",     size=[ x, t, h],  pos= [0, y+t, h],               material=wall_material)
        self.mjcf_model.worldbody.add("geom",   name="wall4",     type="box",     size=[x, t, h],   pos= [0, -(y+t), h],            material=wall_material)
        self.mjcf_model.worldbody.add("light",  name="mainLight", dir=[0, 0, -1], pos=[0, 0, 100],  diffuse= [1, 1, 1],             castshadow= True)

    def _generate_camera(self):
        camHeight = calculateCameraHeight(
            self.map_length[0], self.map_length[1], 90)
        self.mjcf_model.worldbody.add("camera", name="topDownCam", pos=[
                                      0, 0, camHeight], euler=[0, 0, -90], fovy=90)

    def parse_xml_model_from_string(self, xml_string):
        self.mjcf_model: mjcf.RootElement = mjcf.from_xml_string(xml_string)

    def parse_xml_model_from_file(self, file_path):
        file_path = Path(file_path).resolve()
        if not file_path.exists() or file_path.suffix != ".xml":
            raise Exception(f"XML model path does't exist: '{file_path}'")
        self.mjcf_model = mjcf.from_file(str(file_path))

    def export_with_assets(self, dir=Globals.MJCF_OUT_DIR, model_name="out.xml"):
        mjcf.export_with_assets(self.mjcf_model, dir, model_name)


if __name__ == "__main__":
    generator = Generator()
    # Here Change properties
    generator.car_pos = 0, 0
    generator.parking_pos = 5,5

    generator.construct_tree()
    generator.export_with_assets()

