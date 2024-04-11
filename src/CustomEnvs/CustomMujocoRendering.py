# This file is modified version of

from CustomEnvs.CarParking import CarParkingEnv, ObsIndex
import collections
import os
import time
from typing import Optional

import glfw
import imageio
import mujoco
import numpy as np

from pathlib import Path
import sys
import os

SELF_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SELF_DIR.parent))

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


def _import_egl(width, height):
    from mujoco.egl import GLContext

    return GLContext(width, height)


def _import_glfw(width, height):
    from mujoco.glfw import GLContext

    return GLContext(width, height)


def _import_osmesa(width, height):
    from mujoco.osmesa import GLContext

    return GLContext(width, height)


_ALL_RENDERERS = collections.OrderedDict(
    [
        ("glfw", _import_glfw),
        ("egl", _import_egl),
        ("osmesa", _import_osmesa),
    ]
)


class CustomBaseRender:
    def __init__(
        self, env: CarParkingEnv, width: int, height: int
    ):
        """Render context superclass for offscreen and window rendering."""
        self.env = env
        self.model = env.model
        self.data = env.data

        self._markers = []
        self._overlays = {}

        self.viewport = mujoco.MjrRect(0, 0, width, height)

        # This goes to specific visualizer
        self.scn = mujoco.MjvScene(self.model, 1000)
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        self.make_context_current()

        # Keep in Mujoco Context
        self.con = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._set_mujoco_buffer()

    def _set_mujoco_buffer(self):
        raise NotImplementedError

    def make_context_current(self):
        raise NotImplementedError

    def add_overlay(self, gridpos: int, text1: str, text2: str):
        """Overlays text on the scene."""
        if gridpos not in self._overlays:
            self._overlays[gridpos] = ["", ""]
        self._overlays[gridpos][0] += text1 + "\n"
        self._overlays[gridpos][1] += text2 + "\n"

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker: dict):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError("Ran out of geoms. maxgeom: %d" %
                               self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)
                    )
                )
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self.scn.ngeom += 1

    def close(self):
        """Override close in your rendering subclass to perform any necessary cleanup
        after env.close() is called.
        """
        raise NotImplementedError


class CustomOffScreenViewer(CustomBaseRender):
    """Offscreen rendering class with opengl context."""

    def __init__(self, env):
        width = env.model.vis.global_.offwidth
        height = env.model.vis.global_.offheight

        # We must make GLContext before MjrContext
        self._get_opengl_backend(width, height)

        self.env = env
        super().__init__(self.env, width, height)

        self._init_camera()

    def _init_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        for i in range(3):
            self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
        self.cam.distance = self.model.stat.extent

    def _get_opengl_backend(self, width: int, height: int):
        self.backend = os.environ.get("MUJOCO_GL")
        if self.backend is not None:
            try:
                self.opengl_context = _ALL_RENDERERS[self.backend](
                    width, height)
            except KeyError as e:
                raise RuntimeError(
                    "Environment variable {} must be one of {!r}: got {!r}.".format(
                        "MUJOCO_GL", _ALL_RENDERERS.keys(), self.backend
                    )
                ) from e

        else:
            for name, _ in _ALL_RENDERERS.items():
                try:
                    self.opengl_context = _ALL_RENDERERS[name](width, height)
                    self.backend = name
                    break
                except:  # noqa:E722
                    pass
            if self.backend is None:
                raise RuntimeError(
                    "No OpenGL backend could be imported. Attempting to create a "
                    "rendering context will result in a RuntimeError."
                )

    def _set_mujoco_buffer(self):
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

    def make_context_current(self):
        self.opengl_context.make_current()

    def free(self):
        self.opengl_context.free()

    def __del__(self):
        self.free()

    def render(
        self,
        render_mode: str,
        camera_id: Optional[int] = None,
        segmentation: bool = False,
    ):
        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        for marker_params in self._markers:
            self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(self.viewport, self.scn, self.con)

        for gridpos, (text1, text2) in self._overlays.items():
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_100,
                gridpos,
                self.viewport,
                text1.encode(),
                text2.encode(),
                self.con,
            )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

        rgb_arr = np.zeros(
            3 * self.viewport.width * self.viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            self.viewport.width * self.viewport.height, dtype=np.float32
        )

        mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.con)

        if render_mode == "depth_array":
            depth_img = depth_arr.reshape(
                self.viewport.height, self.viewport.width)
            # original image is upside-down, so flip it
            return depth_img[::-1, :]
        else:
            rgb_img = rgb_arr.reshape(
                self.viewport.height, self.viewport.width, 3)

            if segmentation:
                seg_img = (
                    rgb_img[:, :, 0]
                    + rgb_img[:, :, 1] * (2**8)
                    + rgb_img[:, :, 2] * (2**16)
                )
                seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
                seg_ids = np.full(
                    (self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32
                )

                for i in range(self.scn.ngeom):
                    geom = self.scn.geoms[i]
                    if geom.segid != -1:
                        seg_ids[geom.segid + 1, 0] = geom.objtype
                        seg_ids[geom.segid + 1, 1] = geom.objid
                rgb_img = seg_ids[seg_img]

            # original image is upside-down, so flip i
            return rgb_img[::-1, :, :]

    def close(self):
        self.free()
        glfw.terminate()


class CustomWindowViewer(CustomBaseRender):
    """Class for window rendering in all MuJoCo environments."""

    def __init__(self, env, defaultCamId=None):
        glfw.init()

        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        # self._paused = False
        # self._transparent = False
        # self._contacts = False
        self._render_every_frame = True
        # self._image_idx = 0
        # self._image_path = "/tmp/frame_%07d.png"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        # self._advance_by_one_step = False
        self._hide_menu = False

        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        glfw.window_hint(glfw.VISIBLE, 1)
        self.window = glfw.create_window(
            width // 2, height // 2, "mujoco", None, None)

        self.width, self.height = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = self.width * 1.0 / window_width

        # set callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(
            self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        super().__init__(env, width, height)
        glfw.swap_interval(1)

        if defaultCamId is not None:
            self.cam.fixedcamid = defaultCamId
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def _set_mujoco_buffer(self):
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con)

    def make_context_current(self):
        glfw.make_context_current(self.window)

    def free(self):
        if self.window:
            if glfw.get_current_context() == self.window:
                glfw.make_context_current(None)
        glfw.destroy_window(self.window)
        self.window = None

    def __del__(self):
        """Eliminate all of the OpenGL glfw contexts and windows"""
        self.free()

    def render(self):
        """
        Renders the environment geometries in the OpenGL glfw window:
            1. Create the overlay for the left side panel menu.
            2. Update the geometries used for rendering based on the current state of the model - `mujoco.mjv_updateScene()`.
            3. Add markers to scene, these are additional geometries to include in the model, i.e arrows, https://mujoco.readthedocs.io/en/latest/APIreference.html?highlight=arrow#mjtgeom.
                These markers are added with the `add_marker()` method before rendering.
            4. Render the 3D scene to the window context - `mujoco.mjr_render()`.
            5. Render overlays in the window context - `mujoco.mjr_overlay()`.
            6. Swap front and back buffer, https://www.glfw.org/docs/3.3/quick.html.
            7. Poll events like mouse clicks or keyboard input.
        """

        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.destroy_window(self.window)
                glfw.terminate()

            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window
            )

            # update scene
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                mujoco.MjvPerturb(),
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scn,
            )

            # marker items
            for marker in self._markers:
                self._add_marker_to_scene(marker)

            # render
            mujoco.mjr_render(self.viewport, self.scn, self.con)

            # overlay items
            if not self._hide_menu:
                for gridpos, [t1, t2] in self._overlays.items():
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.con,
                    )

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

        self._loop_count += self.model.opt.timestep / (
            self._time_per_render * self._run_speed
        )
        if self._render_every_frame:
            self._loop_count = 1

        while self._loop_count > 0:
            update()
            self._loop_count -= 1

        # clear overlay
        self._overlays.clear()
        # clear markers
        self._markers.clear()

    def close(self):
        self.free()
        glfw.terminate()

    def _key_callback(self, window, key: int, scancode, action: int, mods):
        if action != glfw.RELEASE:
            return
        # Switch cameras
        elif key == glfw.KEY_TAB:
            self.cam.fixedcamid += 1
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            if self.cam.fixedcamid >= self.model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Pause simulation
        # elif key == glfw.KEY_SPACE and self._paused is not None:
        #     self._paused = not self._paused
        # Advances simulation by one step.
        # elif key == glfw.KEY_RIGHT and self._paused is not None:
        #     self._advance_by_one_step = True
        #     self._paused = True
        # Slows down simulation
        # elif key == glfw.KEY_S:
        #     self._run_speed /= 2.0
        # Speeds up simulation
        # elif key == glfw.KEY_F:
        #     self._run_speed *= 2.0
        # Turn off / turn on rendering every frame.
        # elif key == glfw.KEY_D:
        #     self._render_every_frame = not self._render_every_frame
        # Capture screenshot
        # elif key == glfw.KEY_T:
        #     img = np.zeros(
        #         (
        #             glfw.get_framebuffer_size(self.window)[1],
        #             glfw.get_framebuffer_size(self.window)[0],
        #             3,
        #         ),
        #         dtype=np.uint8,
        #     )
        #     mujoco.mjr_readPixels(img, None, self.viewport, self.con)
        #     imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
        #     self._image_idx += 1
        # Display contact forces
        # elif key == glfw.KEY_C:
        #     self._contacts = not self._contacts
        #     self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
        #     self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
        # Display coordinate frames
        # elif key == glfw.KEY_E:
        #     self.vopt.frame = 1 - self.vopt.frame
        # Hide overlay menu
        elif key == glfw.KEY_H:
            self._hide_menu = not self._hide_menu
        # Make transparent
        # elif key == glfw.KEY_R:
        #     self._transparent = not self._transparent
        #     if self._transparent:
        #         self.model.geom_rgba[:, 3] /= 5.0
        #     else:
        #         self.model.geom_rgba[:, 3] *= 5.0
        # Geom group visibility
        # elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
        #     self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        # Quit
        if key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            glfw.destroy_window(self.window)
            glfw.terminate()

    def _cursor_pos_callback(
        self, window: "glfw.LP__GLFWwindow", xpos: float, ypos: float
    ):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self._button_left_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        mujoco.mjv_moveCamera(
            self.model, action, dx / height, dy / height, self.scn, self.cam
        )

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window: "glfw.LP__GLFWwindow", button, act, mods):
        self._button_left_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._button_right_pressed = (
            glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

    def _scroll_callback(self, window, x_offset, y_offset: float):
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.05 * y_offset,
            self.scn,
            self.cam,
        )

    def _create_overlay(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT

        OVERLAY_LIST = [
            [bottomleft, "FPS", f"{int(1 / self._time_per_render)}"],
            [bottomleft, "Step",
                f"{round(self.data.time / self.model.opt.timestep)}"],
            [bottomleft, "episode", f"{self.env.episode}"],
            [bottomleft, "time", f"{self.data.time}"],

            [topleft, "Env stats", "values"],
            [topleft, "reward", f"{self.env.reward}"],
            [topleft, "speed",
                f"{self.env.observation[ObsIndex.VELOCITY_BEGIN:ObsIndex.VELOCITY_END+1]}"],
            [topleft, "dist",
                f"{self.env.observation[ObsIndex.DISTANCE_BEGIN:ObsIndex.DISTANCE_END+1]}"],
            [topleft, "adiff",
                f"{self.env.observation[ObsIndex.ANGLE_DIFF_BEGIN:ObsIndex.ANGLE_DIFF_END+1]}"],
            [topleft, "contact",
                f"{self.env.observation[ObsIndex.CONTACT_BEGIN:ObsIndex.CONTACT_END+1]}"],
            [topleft, "range",
                f"{self.env.observation[ObsIndex.RANGE_BEGIN:ObsIndex.RANGE_END+1]}"],
            [topleft, "pos",
                f"{self.env.observation[ObsIndex.POS_BEGIN:ObsIndex.POS_END+1]}"],
            [topleft, "eul",
                f"{self.env.observation[ObsIndex.EUL_BEGIN:ObsIndex.EUL_END+1]}"],

            [topright, "Model Action", "values"],
            [topright, "engine", "%.2f" % self.env.action[0]],
            [topright, "wheel", "%.2f" % self.env.action[1]],
        ]

        for overlay in OVERLAY_LIST:
            self.add_overlay(*overlay)


class CustomMujocoRenderer:
    """This is the MuJoCo renderer manager class for every MuJoCo environment.

    The class has two main public methods available:
    - :meth:`render` - Renders the environment in three possible modes: "human", "rgb_array", or "depth_array"
    - :meth:`close` - Closes all contexts initialized with the rendere

    """

    def __init__(
        self,
        model,
        data
    ):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            model: MjModel data structure of the MuJoCo simulation
            data: MjData data structure of the MuJoCo simulation
            default_cam_config: dictionary with attribute values of the viewer's default camera, https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global
        """
        self.model = model
        self.data = data
        self._viewers = {}
        self.viewer = None
        self.default_human_mode_camera = 0

    def render(
        self,
        render_mode: str,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        """Renders a frame of the simulation in a specific format and camera view.

        Args:
            render_mode: The format to render the frame, it can be: "human", "rgb_array", or "depth_array"
            camera_id: The integer camera id from which to render the frame in the MuJoCo simulation
            camera_name: The string name of the camera from which to render the frame in the MuJoCo simulation. This argument should not be passed if using cameara_id instead and vice versa

        Returns:
            If render_mode is "rgb_array" or "depth_arra" it returns a numpy array in the specified format. "human" render mode does not return anything.
        """

        viewer = self._get_viewer(render_mode=render_mode)

        if render_mode in {
            "rgb_array",
            "depth_array",
        }:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

            img = viewer.render(render_mode=render_mode, camera_id=camera_id)
            return img

        elif render_mode == "human":
            return viewer.render()

    def _get_viewer(self, render_mode: str):
        """Initializes and returns a viewer class depending on the render_mode
        - `WindowViewer` class for "human" render mode
        - `OffScreenViewer` class for "rgb_array" or "depth_array" render mode
        """
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            if render_mode == "human":
                self.viewer = CustomWindowViewer(
                    self.env, self.default_human_mode_camera)

            elif render_mode in {"rgb_array", "depth_array"}:
                self.viewer = CustomOffScreenViewer(self.env)
            else:
                raise AttributeError(
                    f"Unexpected mode: {render_mode}, expected modes: human, rgb_array, or depth_array"
                )
            # Add default camera parameters
            self._viewers[render_mode] = self.viewer

        if len(self._viewers.keys()) > 1:
            # Only one context can be current at a time
            self.viewer.make_context_current()

        return self.viewer

    def close(self):
        """Close the OpenGL rendering contexts of all viewer modes"""
        for _, viewer in self._viewers.items():
            viewer.close()
