import math
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
    pass

class ParkingSpot:
    
    def __init__(self,
                 name="parkingSpot",
                 carSize=(2, 1, 0.25),
                 targetPos=(0, 0, 0),
                 targetPaddings=(1.4, 1.7),
                 lineWidth=0.2):

        self.model_name = name
        self.carSize = carSize
        self.targetPaddings = targetPaddings
        self.targetPos = targetPos
        self.lineWidth = lineWidth
        self.lineHeightSize = 0.001
        self.friction = [1.0, 0.005, 0.0001]
        self.construct_tree()

    def construct_tree(self):
        self.mjcf_model = mjcf.RootElement(model=self.model_name)

        targetXSize = self.carSize[0]/2*self.targetPaddings[0]
        targetYSize = self.carSize[1]/2*self.targetPaddings[1]
        lineWidthSize = self.lineWidth/2
        lineHeightSize = self.lineHeightSize

        material = self.mjcf_model.asset.add(
            "material", name=f"line_material", rgba=[1, 1, 1, 1])

        space = self.mjcf_model.worldbody.add(
            "body", name=f"{self.model_name}_space", pos=self.targetPos)
        self.site_center = space.add("site", name=f"{self.model_name}_site_center", type="cylinder", size=[0.1, 0.001], material=material)
        space.add("geom", type="box", size=[lineWidthSize, targetYSize, lineHeightSize], pos=[
                  targetXSize - lineWidthSize, 0, 0], friction=self.friction, material=material)
        space.add("geom", type="box", size=[targetXSize, lineWidthSize, lineHeightSize], pos=[
                  0, targetYSize - lineWidthSize, 0], friction=self.friction, material=material)
        space.add("geom", type="box", size=[targetXSize, lineWidthSize, lineHeightSize], pos=[
                  0, -(targetYSize - lineWidthSize), 0], friction=self.friction, material=material)
        space.add("geom", type="box", size=[lineWidthSize, targetYSize, lineHeightSize], pos=[
                  -(targetXSize - lineWidthSize), 0, 0], friction=self.friction, material=material)


class Generator:
    topDownCamName = "topDownCam"
    
    def __init__(self, name):
        self.model_name = name

        # MAP PROPS
        self.map_length = np.array([20, 20, 20, 5])  # X, Y, H, T
        self.construct_tree()

    def construct_tree(self):
        self.mjcf_model = mjcf.RootElement(model=self.model_name)
        
        # MAP
        self._generate_map()
        
        #CAM
        self._generate_camera()
        
        # Parking
        parkingSpot = ParkingSpot("parkingSpot", targetPos=(5, 5, 0)) 
        self.mjcf_model.attach(parkingSpot.mjcf_model)

    def _generate_map(self):

        x, y, h, t = self.map_length/2

        self.mjcf_model.asset.add("texture", name="sky_texture", type = "skybox", file=ASSET_DIR+"/sky1.png")
        wall_material = self.mjcf_model.asset.add("material", name="wall_material", rgba=[1, 1, 1, 0.000001])
        ground_texture = self.mjcf_model.asset.add("texture", name= "ground_texture", type= "2d", file= ASSET_DIR+"/ground.png")
        ground_material = self.mjcf_model.asset.add("material", name= "ground_material", texture=ground_texture, texrepeat=[25, 25])
        
        listWorldBody = \
            [
                ["worldbody", "geom", {"name": "ground", "type": "plane", "size": [
                    x, y, t], "friction": [1.0, 0.005, 0.0001], "material": ground_material}],
                ["worldbody", "geom", {"name": "ceiling", "type": "box", "size": [
                    x, y, t], "pos": [0, 0, h], "rgba": [0, 0, 0, 0], "material": wall_material}],
                ["worldbody", "geom", {"name": "wall1", "type": "box", "size": [
                    t, y, h], "pos": [-(x+t), 0, h], "material": wall_material}],
                ["worldbody", "geom", {"name": "wall2", "type": "box", "size": [
                    t, y, h], "pos": [x+t, 0, h], "material": wall_material}],
                ["worldbody", "geom", {"name": "wall3", "type": "box", "size": [
                    x, t, h], "pos": [0, y+t, h], "material": wall_material}],
                ["worldbody", "geom", {"name": "wall4", "type": "box", "size": [
                    x, t, h], "pos": [0, -(y+t), h], "material": wall_material}],
                ["worldbody", "light", {"name": "mainLight", "dir": [
                    0, 0, -1], "pos": [0, 0, 100], "diffuse": [1, 1, 1], "castshadow": True}],
            ]
        for attrib, tag, tagAttrib in listWorldBody:
            self.mjcf_model.__getattr__(attrib).add(tag, **tagAttrib)
            
    def _generate_camera(self):
        camHeight = calculateCameraHeight(self.map_length[0], self.map_length[1], 90)
        self.mjcf_model.worldbody.add("camera", name=self.topDownCamName, pos=[0, 0, camHeight], euler=[0, 0, -90], fovy=90)
        

    def parse_xml_model_from_string(self, xml_string):
        self.mjcf_model: mjcf.RootElement = mjcf.from_xml_string(xml_string)

    def parse_xml_model_from_file(self, file_path):
        file_path = Path(file_path).resolve()
        if not file_path.exists() or file_path.suffix != ".xml":
            raise Exception(f"XML model path does't exist: '{file_path}'")
        self.mjcf_model = mjcf.from_file(str(file_path))

    def generate_model_xml_string(self):
        xml_string = self.mjcf_model.to_xml_string()
        return xml_string

    def generate_model_xml_file(self, dir=Globals.MJCF_OUT_DIR, file_name="out.xml") -> str:

        file_path = Path(f"{dir}/{file_name}").resolve()
        if not file_path.parent.is_dir() or file_path.suffix != ".xml":
            print(
                f"Wrong file path: '{file_path}' Using default: '{Globals.MJCF_OUT_DIR}/{file_name}'")
            file_path = Globals.MJCF_OUT_DIR + "out.xml"

        with open(file_path, 'w') as xml_file:
            xml_file.write(self.generate_model_xml_string())
        return str(file_path)

    def export_with_assets(self, dir=Globals.MJCF_OUT_DIR, model_name="out.xml"):
        mjcf.export_with_assets(self.mjcf_model, dir, model_name)


def run():
    a = Generator("Base World")
    a.construct_tree()

    a.export_with_assets()


if __name__ == "__main__":
    run()
