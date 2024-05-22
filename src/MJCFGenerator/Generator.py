# autopep8: off

from logging import config
import math
import sys
from typing import Tuple
from dm_control import mjcf
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__,'..','..').resolve()))

import MJCFGenerator.Config as mjcf_cfg
import PathsConfig as paths_cfg

# autopep8: on

def calculateCameraHeight(x, y, fov_y_degrees):
    fov_y_radians = math.radians(fov_y_degrees)
    h = max(x, y) / 2*math.tan(fov_y_radians / 2)
    return h


class Wheel:
    def __init__(self, wheelName, is_steering = False) -> None:
        self._wheelName = wheelName
        self._is_steering = is_steering
        
        # FROM CONFIG
        self._wheel_friction = mjcf_cfg.WHEEL_FRICTION
        self._wheel_mass = mjcf_cfg.WHEEL_MASS
        self._wheel_angle_limit = mjcf_cfg.WHEEL_ANGLE_LIMIT
        self._wheel_radius = mjcf_cfg.WHEEL_RADIUS
        self._wheel_thickness = mjcf_cfg.WHEEL_THICKNESS

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

        wheelBody.add("geom",
                      type="cylinder",
                      size=[self._wheel_radius, self._wheel_thickness / 2],
                      mass=self._wheel_mass,
                      friction=self._wheel_friction,
                      material=wheelMaterial)

        if self._is_steering:
            steering_joint = wheelBody.add("joint", name="steering_joint", type="hinge",
                                           axis=[0, 1, 0], limited=True, range=self._wheel_angle_limit)

        rolling_joint = wheelBody.add("joint", name="rolling_joint")

        return mjcf_model, rolling_joint, steering_joint

class Trailer: 
    def __init__(self,
                 trailerName,
                 trailerDims) -> None:
        self._trailer_name: str = trailerName
        self._trailerDims = trailerDims
        
        # FROM CONFIG
        self._trailer_mass = mjcf_cfg.TRAILER_MASS
        self._trailer_wheel_axis_spacing = mjcf_cfg.TRAILER_WHEEL_AXIS_SPACING
        self._trailer_wheel_spacing = mjcf_cfg.TRAILER_WHEEL_SPACING
        self._trailer_wheel_mount_height = mjcf_cfg.TRAILER_WHEEL_MOUNT_HEIGHT
        self._trailer_color = mjcf_cfg.TRAILER_COLOR
        self._trailer_hitbox_scale = mjcf_cfg.TRAILER_HITBOX_SCALE
        
        self.wheelGenerator = Wheel("Wheel")
        
    def construct_tree(self):
        mjcf_model:mjcf.RootElement = mjcf.RootElement(model=self._trailer_name)

        chassisXSize = self._trailerDims[0] / 2
        chassisYsize = self._trailerDims[1] / 2
        chassisZsize = self._trailerDims[2] / 2
        trailerHitchXsize = self._trailerDims[3] / 2 

        # autopep8: off
        material = mjcf_model.asset.add("material", name="trailer_material",
            rgba=self._trailer_color,
            reflectance=0,
            shininess=1,
            emission=0.0,
            specular=0.0)

        self.center_site=mjcf_model.worldbody.add("site", name="site_center")
        mjcf_model.worldbody.add("geom",
                                 type="box",
                                 size=[chassisXSize,chassisYsize,chassisZsize],
                                 mass=self._trailer_mass,
                                 material=material)
        mjcf_model.worldbody.add("geom",
                                 type="cylinder",
                                 zaxis = [1, 0 ,0],
                                 pos = [chassisXSize+trailerHitchXsize, 0, 0],
                                 size=[0.05,trailerHitchXsize],
                                 mass=50,
                                 material=material) # HITCH
        
        front_attachment_site = mjcf_model.worldbody.add("site",
                                                         name="front_attachment_site",
                                                         pos=[chassisXSize+self._trailerDims[3], 0, 0])
        
        touch_site = mjcf_model.worldbody.add("site",
                                              type="box",
                                              size=[chassisXSize*self._trailer_hitbox_scale[0], chassisYsize*self._trailer_hitbox_scale[1], chassisZsize*self._trailer_hitbox_scale[2]],
                                              rgba=[0, 0, 0, 0])
        
        # ADD WHEELS
        s3=mjcf_model.worldbody.add("site",
                                    name="wheel3_attachment_site",
                                    pos=[-chassisXSize * self._trailer_wheel_axis_spacing, -chassisYsize * self._trailer_wheel_spacing, self._trailerDims[2] * self._trailer_wheel_mount_height])
        s4=mjcf_model.worldbody.add("site",
                                    name="wheel4_attachment_site",
                                    pos=[-chassisXSize * self._trailer_wheel_axis_spacing, chassisYsize * self._trailer_wheel_spacing, self._trailerDims[2] * self._trailer_wheel_mount_height])

        w3MJCF, w3rolling, _, = self.wheelGenerator.construct_tree("wheel3", False)
        w4MJCF, w4rolling, _, = self.wheelGenerator.construct_tree("wheel4", False)

        s3.attach(w3MJCF)
        s4.attach(w4MJCF)

        # ADD TENDONS
        tendonFixed = mjcf_model.tendon.add("fixed", name="back_wheels_tendon")
        tendonFixed.add("joint", joint=w3rolling, coef=1000)
        tendonFixed.add("joint", joint=w4rolling, coef=1000)


        #ADD SENSORS
        ## RANGE
        zaxis = np.array([
            # [chassisXSize, chassisYsize, 0],
            # [chassisXSize, 0, 0],
            # [chassisXSize, -chassisYsize, 0],
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
            sensor = mjcf_model.sensor.add("rangefinder", name=f"range_sensor_{index}", site=sensor_site, cutoff=mjcf_cfg.SENSORS_MAX_RANGE)
            
        # ## OTHERS
        mjcf_model.sensor.add("touch", name=f"touch_sensor", site=touch_site, cutoff=mjcf_cfg.SENSORS_MAX_RANGE)
        
        # autopep8: on
        return mjcf_model, front_attachment_site

class Car:
    def __init__(self, carName, carDims) -> None:
        self._car_name: str = carName
        self._carDims = carDims
        self.wheelGenerator = Wheel("Wheel")
        
        # FROM CONFIG
        self._car_mass = mjcf_cfg.CAR_MASS
        self._car_wheel_axis_spacing = mjcf_cfg.CAR_WHEEL_AXIS_SPACING
        self._car_wheel_spacing = mjcf_cfg.CAR_WHEEL_SPACING
        self._car_wheel_mount_height = mjcf_cfg.CAR_WHEEL_MOUNT_HEIGHT
        self._car_spawn_height = mjcf_cfg.CAR_SPAWN_HEIGHT
        self._car_color = mjcf_cfg.CAR_COLOR
        self._car_hitbox_scale = mjcf_cfg.CAR_HITBOX_SCALE

    def construct_tree(self):
        mjcf_model = mjcf.RootElement(model=self._car_name)

        chassisXSize = self._carDims[0] / 2
        chassisYsize = self._carDims[1] / 2
        chassisZsize = self._carDims[2] / 2

        wheelControlRange = (math.radians(
            mjcf_cfg.WHEEL_ANGLE_LIMIT[0]), math.radians(mjcf_cfg.WHEEL_ANGLE_LIMIT[1]))

        # autopep8: off
        material = mjcf_model.asset.add("material", name="chassis_material",
            rgba=self._car_color,
            reflectance=0,
            shininess=1,
            emission=0.0,
            specular=0.0)

        self.center_site=mjcf_model.worldbody.add("site", name="site_center")
        mjcf_model.worldbody.add("geom", type="box", size=[chassisXSize,chassisYsize,chassisZsize], mass=self._car_mass, material=material)
        touch_site = mjcf_model.worldbody.add("site",
                                              type="box",
                                              size=[chassisXSize*self._car_hitbox_scale[0], chassisYsize*self._car_hitbox_scale[1],chassisZsize*self._car_hitbox_scale[2]],
                                              rgba=[0, 0, 0, 0])
        
        rear_attachment_site = mjcf_model.worldbody.add("site", name="rear_attachment_site", pos=[-chassisXSize, 0, 0])
        
        # ADD WHEELS
        s1=mjcf_model.worldbody.add("site", name="wheel1_attachment_site", pos=[chassisXSize * self._car_wheel_axis_spacing, -chassisYsize * self._car_wheel_spacing, self._carDims[2] * self._car_wheel_mount_height])
        s2=mjcf_model.worldbody.add("site", name="wheel2_attachment_site", pos=[chassisXSize * self._car_wheel_axis_spacing, chassisYsize * self._car_wheel_spacing, self._carDims[2] * self._car_wheel_mount_height])
        s3=mjcf_model.worldbody.add("site", name="wheel3_attachment_site", pos=[-chassisXSize * self._car_wheel_axis_spacing, -chassisYsize * self._car_wheel_spacing, self._carDims[2] * self._car_wheel_mount_height])
        s4=mjcf_model.worldbody.add("site", name="wheel4_attachment_site", pos=[-chassisXSize * self._car_wheel_axis_spacing, chassisYsize * self._car_wheel_spacing, self._carDims[2] * self._car_wheel_mount_height])

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
            # [-chassisXSize, chassisYsize, 0],
            # [-chassisXSize, 0, 0],
            # [-chassisXSize, -chassisYsize, 0],
            [0, chassisYsize, 0],
            [0, -chassisYsize, 0]

        ])
        sensor_size = (1e-6, 1e-6)
        sensor_pos_scale_factor = 1e-3
        for index, _zaxis in enumerate(zaxis):
            _zaxis = _zaxis + _zaxis*sensor_pos_scale_factor
            sensor_site = mjcf_model.worldbody.add("site", name=f"sensor_site_{index}", type="sphere", size=sensor_size, pos=_zaxis, zaxis=_zaxis)
            sensor = mjcf_model.sensor.add("rangefinder", name=f"range_sensor_{index}", site=sensor_site, cutoff=mjcf_cfg.SENSORS_MAX_RANGE)
            
        ## OTHERS
        mjcf_model.sensor.add("touch", name=f"touch_sensor", site=touch_site, cutoff=mjcf_cfg.SENSORS_MAX_RANGE)
        mjcf_model.sensor.add("velocimeter", name=f"speed_sensor", site=self.center_site, cutoff=10)
        mjcf_model.sensor.add("framepos", name=f"pos_global_sensor", objtype="site", objname=self.center_site)
        
        # autopep8: on
        return mjcf_model, rear_attachment_site


class ParkingSpot:
    def __init__(self,
                 name,
                 carSize,
                 trailerSize,
                 ):

        self._model_name = name
        self._carSize = carSize
        self._trailerSize = trailerSize
        
        # FROM CONFIG
        self._parking_spot_paddings = mjcf_cfg.PARKING_SPOT_PADDINGS
        self._parking_line_width = mjcf_cfg.PARKING_LINE_WIDTH
        self._parking_line_height_size = mjcf_cfg.PARKING_LINE_HEIGHT_SIZE
        self._car_color = mjcf_cfg.CAR_COLOR
        self._car_name = mjcf_cfg.CAR_NAME
        self._trailer_color = mjcf_cfg.TRAILER_COLOR
        self._trailer_name = mjcf_cfg.TRAILER_NAME
        

    def construct_tree(self):
        mjcf_model = mjcf.RootElement(model=self._model_name)

        self.site_center = mjcf_model.worldbody.add("site",
                                                        name=f"site_center",
                                                        type="cylinder",
                                                        pos = [0, 0, 0],
                                                        size=[0.1, 0.001],
                                                        rgba=[1, 1, 1, 1])
        
        self.site_center_car = mjcf_model.worldbody.add("site",
                                                        name=f"site_center_{self._car_name}",
                                                        type="cylinder",
                                                        pos = [self._carSize[0]/2, 0, 0],
                                                        size=[0.1, 0.001],
                                                        rgba=self._car_color)
        self.site_center_trailer = mjcf_model.worldbody.add("site",
                                                        name=f"site_center_{self._trailer_name}",
                                                        type="cylinder",
                                                        pos = [-(self._trailerSize[0]/2+self._trailerSize[3]), 0, 0],
                                                        size=[0.1, 0.001],
                                                        rgba=self._trailer_color)
        
        
        longer_line_length = (self._trailerSize[0] + self._trailerSize[3] + self._carSize[0]) * self._parking_spot_paddings[0]
        shorter_line_length = max(self._carSize[1], self._trailerSize[1]) * self._parking_spot_paddings[1]

        half_longer = longer_line_length / 2
        half_shorter = shorter_line_length / 2
        half_width = self._parking_line_width / 2

        mjcf_model.worldbody.add("site",
                                type="box",
                                pos=[0, -half_shorter, 0],
                                size=[half_longer, half_width, self._parking_line_height_size / 2],
                                rgba=[1, 1, 1, 1])
        mjcf_model.worldbody.add("site",
                                type="box",
                                pos=[0, half_shorter, 0],
                                size=[half_longer, half_width, self._parking_line_height_size / 2],
                                rgba=[1, 1, 1, 1])

        mjcf_model.worldbody.add("site",
                                type="box",
                                pos=[-half_longer, 0, 0],
                                size=[half_width, half_shorter, self._parking_line_height_size / 2],
                                rgba=[1, 1, 1, 1])
        mjcf_model.worldbody.add("site",
                                type="box",
                                pos=[half_longer, 0, 0],
                                size=[half_width, half_shorter, self._parking_line_height_size / 2],
                                rgba=[1, 1, 1, 1])
        
        return mjcf_model


class GeneratorClass:
    def __init__(self):
        
        # FROM CONFIG
        self._map_length = mjcf_cfg.MAP_LENGTH
        self._parking_spot_kwargs = mjcf_cfg.PARKING_SPOT_KWARGS
        self._car_name = mjcf_cfg.CAR_NAME
        self._car_dims = mjcf_cfg.CAR_DIMS
        self._trailer_name = mjcf_cfg.TRAILER_NAME
        self._trailer_dims = mjcf_cfg.TRAILER_DIMS
        self._parking_name = mjcf_cfg.PARKING_NAME
        self._mjcdf_model_name = paths_cfg.MJCF_MODEL_NAME
        self._render_off_width = mjcf_cfg.RENDER_OFF_WIDTH
        self._render_off_height = mjcf_cfg.RENDER_OFF_HEIGHT
        self._spawn_points = mjcf_cfg.CAR_SPAWN_KWARGS
        self._custom_obstacles = mjcf_cfg.CUSTOM_OBSTACLES_KWARGS
        self._car_spawn_height = mjcf_cfg.CAR_SPAWN_HEIGHT
        self._trailer_hitch_angle_limit = mjcf_cfg.TRAILER_HITCH_ANGLE_LIMIT
        # ===================================
        
        self.carGenerator = Car(self._car_name, self._car_dims)
        self.trailerGenerator = Trailer(self._trailer_name, self._trailer_dims)
        self.parkingSpotGenerator = ParkingSpot(self._parking_name, self._car_dims, self._trailer_dims)

    def construct_tree(self):
        self.mjcf_model: mjcf.RootElement = mjcf.RootElement(self._mjcdf_model_name)

        # autopep8: off

        # RENDERING
        global_settings = getattr(self.mjcf_model.visual, 'global')
        global_settings.offwidth = self._render_off_width
        global_settings.offheight = self._render_off_height
        
        # MAP
        self._generate_map()

        # CAM
        self._generate_camera()

        # CAR
        carMJCF, carRearAttachmentSite = self.carGenerator.construct_tree()
        car_attachment_body = self.spawn_points[0].attach(carMJCF)
        car_attachment_body.add("freejoint", name="")
        
        #TRAILER
        trailerMJCF, TrailerFrontAttachmentSite = self.trailerGenerator.construct_tree()
        
        # CONNECT CAR AND TRAILER
        self._connect_bodies_at_sites(carRearAttachmentSite, TrailerFrontAttachmentSite, trailerMJCF)

        # Parking
        self.parking_attachment_site = self.mjcf_model.worldbody.add("site", name="parking_attachment_site", **self._parking_spot_kwargs)
        parkingSpotMJCF = self.parkingSpotGenerator.construct_tree()
        self.parking_attachment_site.attach(parkingSpotMJCF)
        
        # CONNECT CAR AND PARKING WITH POS SENSOR
        self.mjcf_model.sensor.add("framepos", name=f"{self._car_name}_to_{self._parking_name}_pos", objtype="site", objname=self.carGenerator.center_site,
                                   reftype="site", refname=self.parkingSpotGenerator.site_center_car)

        # autopep8: on
    
    def _connect_bodies_at_sites(self, site_a, site_b, body_b):
        site_a_pos = site_a.pos
        site_b_pos = site_b.pos
        new_pos = site_a_pos - site_b_pos
        
        b_attachment_body = site_a.attach(body_b)
        b_attachment_body.pos = new_pos
        b_attachment_body.add("joint",
                              type="hinge",
                              axis=[0, 0, 1],
                              pos=site_b_pos,
                              limited=True,
                              range=self._trailer_hitch_angle_limit
                              )
            
    
    def _generate_map(self):

        x, y, h, t = self._map_length[0]/2, self._map_length[1] / \
            2, self._map_length[2]/2, self._map_length[3]/2

        # autopep8: off
        self.mjcf_model.asset.add( "texture",
                                  name="sky_texture",
                                  type="skybox",
                                  file=paths_cfg.ASSET_DIR+"/sky1.png")
        wall_material = self.mjcf_model.asset.add("material",
                                                  name="wall_material",
                                                  rgba=[1, 1, 1, 0.000001])
        ground_texture = self.mjcf_model.asset.add( "texture",
                                                   name="ground_texture",
                                                   type="2d",
                                                   file=paths_cfg.ASSET_DIR+"/ground.png")
        ground_material = self.mjcf_model.asset.add("material",
                                                    name="ground_material",
                                                    texture=ground_texture,
                                                    texrepeat=[25, 25])
        obstacle_texture = self.mjcf_model.asset.add( "texture",
                                                     name="obstacle_texture",
                                                     type="2d",
                                                     file=paths_cfg.ASSET_DIR+"/rustymetal.png")
        obstacle_material = self.mjcf_model.asset.add("material",
                                                      name="obstacle_material",
                                                      texture=obstacle_texture, )

        self.mjcf_model.worldbody.add("geom",   name="ground",    type="plane",   size=[x, y, t],   friction= [1.0, 0.005, 0.0001], material=ground_material)
        self.mjcf_model.worldbody.add("geom",   name="ceiling",   type="box",     size=[x, y, t],   pos= [0, 0, self._map_length[2]],rgba= [0, 0, 0, 0])
        self.mjcf_model.worldbody.add("geom",   name="wall1",     type="box",     size=[t, y, h],   pos= [-(x+t), 0, h],            material=wall_material)
        self.mjcf_model.worldbody.add("geom",   name="wall2",     type="box",     size=[t, y, h],   pos= [x+t, 0, h],               material=wall_material)
        self.mjcf_model.worldbody.add("geom",   name="wall3",     type="box",     size=[ x, t, h],  pos= [0, y+t, h],               material=wall_material)
        self.mjcf_model.worldbody.add("geom",   name="wall4",     type="box",     size=[x, t, h],   pos= [0, -(y+t), h],            material=wall_material)
        self.mjcf_model.worldbody.add("light",  name="mainLight", dir=[0, 0, -1], pos=[0, 0, 100],  diffuse= [1, 1, 1],             castshadow= True)

        # CUSTOM OBSTACLES
        for obstacle_kwargs in self._custom_obstacles:
            self.mjcf_model.worldbody.add("geom",
                                          type="box",
                                          material=obstacle_material,
                                          **obstacle_kwargs)
        
        # SPAWN POINTS
        self.spawn_points = []
        for i, spawn_point in enumerate(self._spawn_points):
            self.spawn_points.append(
                self.mjcf_model.worldbody.add("site",
                                              name=f"spawn_point_{i}",
                                              type="ellipsoid",
                                              size="0.2 0.1 0.1",
                                              rgba=(0.408, 0.592, 0.941, 1),
                                              **spawn_point
                                              )
            )
            
        
        
    def _generate_camera(self):
        camHeight = calculateCameraHeight(
            self._map_length[0], self._map_length[1], 90)
        self.mjcf_model.worldbody.add("camera", name="topDownCam", pos=[
                                      0, 0, camHeight], euler=[0, 0, 0], fovy=90)

    def parse_xml_model_from_string(self, xml_string):
        self.mjcf_model: mjcf.RootElement = mjcf.from_xml_string(xml_string)

    def parse_xml_model_from_file(self, file_path):
        file_path = Path(file_path).resolve()
        if not file_path.exists() or file_path.suffix != ".xml":
            raise Exception(f"XML model path does't exist: '{file_path}'")
        self.mjcf_model = mjcf.from_file(str(file_path))

    def export_with_assets(self,
                           dir=paths_cfg.MJCF_OUT_DIR,
                           model_name=paths_cfg.MJCF_MODEL_NAME):
        mjcf.export_with_assets(self.mjcf_model, dir, model_name)
        print(f"Generated: {Path(dir, model_name)}")

def generate_MJCF():
    generator = GeneratorClass()
    generator.construct_tree()
    generator.export_with_assets()

if __name__ == "__main__":
    generate_MJCF()

