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


class Car:
    _wheelRadius: float = 0.3
    _wheelThickness: float = 0.2
    _wheelFriction: Tuple[float, float, float] = (1, 1e-3, 1e-3)
    _wheelAxisSpacing: float = 0.6
    _wheelSpacing: float = 1
    _wheelMountHeight: float = -0.25
    _carMass: float = 1500
    _wheelMass: float = 30
    _maxSensorVal: int = 5
    _wheelAngleLimit: Tuple[float, float, float] = (-45, 45)
    _carChassisColorRGBA = (0.8, 0.102, 0.063, 1)

    def __init__(self, carName, carDims) -> None:
        self._carName: str = carName
        self._carDims = carDims

    def construct_tree(self):
        self.mjcf_model = mjcf.RootElement(model=self._carName)

        chassisXSize = self._carDims[0] / 2
        chassisYsize = self._carDims[1] / 2
        chassisZsize = self._carDims[2] / 2

        wheelControlRange = (math.radians(
            self._wheelAngleLimit[0]), math.radians(self._wheelAngleLimit[1]))

        # autopep8: off
        material = self.mjcf_model.asset.add("material", name="chassis_material",
            rgba=self._carChassisColorRGBA,
            reflectance=0,
            shininess=1,
            emission=0.0,
            specular=0.0)

        body = self.mjcf_model.worldbody.add("body")
        body.add("freejoint")
        body.add("site", name="site_center")
        body.add("geom", type="box", size=[chassisXSize,chassisYsize,chassisZsize], mass=self._carMass, material=material)

        # autopep8: on


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
        self.mjcf_model = mjcf.RootElement(model=self._model_name)

        targetXSize = self._carSize[0]/2*self._targetPaddings[0]
        targetYSize = self._carSize[1]/2*self._targetPaddings[1]
        lineWidthSize = self._lineWidth/2
        lineHeightSize = self._lineHeightSize

        # autopep8: off
        material = self.mjcf_model.asset.add("material", name=f"line_material",rgba=[1, 1, 1, 1])
        self.site_center = self.mjcf_model.worldbody.add("site",name=f"{self._model_name}_site_center",  type="cylinder",size=[0.1, 0.001],  material=material)

        self.mjcf_model.worldbody.add("geom", type="box", size=[lineWidthSize, targetYSize, lineHeightSize], pos=[targetXSize - lineWidthSize, 0, 0],       friction=self._friction, material=material)
        self.mjcf_model.worldbody.add("geom", type="box", size=[targetXSize, lineWidthSize, lineHeightSize], pos=[0, targetYSize - lineWidthSize, 0],       friction=self._friction, material=material)
        self.mjcf_model.worldbody.add("geom", type="box", size=[targetXSize, lineWidthSize, lineHeightSize], pos=[ 0, -(targetYSize - lineWidthSize), 0],   friction=self._friction, material=material)
        self.mjcf_model.worldbody.add("geom", type="box", size=[lineWidthSize, targetYSize, lineHeightSize], pos=[-(targetXSize - lineWidthSize), 0, 0],    friction=self._friction, material=material)
        # autopep8: on


class Generator:
    model_name = "MainModel"

    def __init__(self):
        # MAP PROPS
        self._map_length = [20, 20, 20, 5]
        self._car_dims = (2, 1, 0.25)
        self._car_pos = (0, 0)
        self._parking_pos = (5, 5, 0)

        self.carGenerator = Car("mainCar", self._car_dims)
        self.parkingSpotGenerator = ParkingSpot("parkingSpot", self._car_dims)

    @property
    def car_pos(self):
        return self._car_pos

    @car_pos.setter
    def car_pos(self, val):
        self._car_pos = [val[0], val[1], Car._wheelRadius +
                         self._car_dims[2] * Car._wheelMountHeight]

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

        # MAP
        self._generate_map()

        # CAM
        self._generate_camera()

        # CAR
        self.car_attachment_site = self.mjcf_model.worldbody.add(
            "site", name="car_attachment_site", pos=self._car_pos)
        self.carGenerator.construct_tree()
        self.car_attachment_site.attach(self.carGenerator.mjcf_model)

        # Parking
        self.parking_attachment_site = self.mjcf_model.worldbody.add(
            "site", name="parking_attachment_site", pos=self._parking_pos)
        self.parkingSpotGenerator.construct_tree()
        self.parking_attachment_site.attach(
            self.parkingSpotGenerator.mjcf_model)

    def _generate_map(self):

        x, y, h, t = self.map_length[0]/2, self.map_length[1] / \
            2, self.map_length[2]/2, self.map_length[3]/2

        # autopep8: off
        self.mjcf_model.asset.add( "texture", name="sky_texture", type="skybox", file=ASSET_DIR+"/sky1.png")
        wall_material = self.mjcf_model.asset.add(  "material",     name="wall_material",   rgba=[1, 1, 1, 0.000001])
        ground_texture = self.mjcf_model.asset.add( "texture",      name="ground_texture",  type="2d",              file=ASSET_DIR+"/ground.png")
        ground_material = self.mjcf_model.asset.add("material",     name="ground_material", texture=ground_texture, texrepeat=[25, 25])

        self.mjcf_model.worldbody.add("geom",   name="ground",    type="plane",   size=[x, y, t],   friction= [1.0, 0.005, 0.0001], material=ground_material)
        self.mjcf_model.worldbody.add("geom",   name="ceiling",   type="box",     size=[x, y, t],   pos= [0, 0, h],                 rgba= [0, 0, 0, 0],          material= wall_material)
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

