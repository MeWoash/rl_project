from abc import ABC
import collections
import os
from typing import Optional
import mujoco
import numpy as np
import glfw
import matplotlib.pyplot as plt


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


class BaseRender(ABC):
    def __init__(
        self, model, data, width: int, height: int
    ):
        """Render context superclass for offscreen and window rendering."""
        self._model = model
        self._data = data

        self._markers = []
        self._overlays = {}

        buffer_width = self._model.vis.global_.offwidth
        buffer_height = self._model.vis.global_.offheight
        if width > buffer_width:
            raise ValueError('Image width {} > framebuffer width {}. Either reduce '
                             'the image width or specify a larger offscreen '
                             'framebuffer in the model XML using the clause\n'
                             '<visual>\n'
                             '  <global offwidth="my_width"/>\n'
                             '</visual>'.format(width, buffer_width))
        if height > buffer_height:
            raise ValueError('Image height {} > framebuffer height {}. Either reduce '
                             'the image height or specify a larger offscreen '
                             'framebuffer in the model XML using the clause\n'
                             '<visual>\n'
                             '  <global offheight="my_height"/>\n'
                             '</visual>'.format(height, buffer_height))

        self._height = height
        self._width = width

        self.viewport = mujoco.MjrRect(0, 0, self._width, self._height)

        # This goes to specific visualizer
        self.scn = mujoco.MjvScene(self._model, 1000)
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        self.make_context_current()

        # Keep in Mujoco Context
        self.con = mujoco.MjrContext(
            self._model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._set_mujoco_buffer()


    def _set_mujoco_buffer(self):
        raise NotImplementedError

    def make_context_current(self):
        raise NotImplementedError

    def close(self):
        """Override close in your rendering subclass to perform any necessary cleanup
        after env.close() is called.
        """
        raise NotImplementedError

class OffScreenViewer(BaseRender):
    """Offscreen rendering class with opengl context."""

    def __init__(self, model, data, width, height):

        # We must make GLContext before MjrContext
        self._get_opengl_backend(width, height)
        super().__init__(model, data, width, height)
        self._init_camera()

    def _init_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        self.cam.lookat = [0, 0, 0]

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
        camera_id: Optional[int] = -1,
    ):
        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self._model,
            self._data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        mujoco.mjr_render(self.viewport, self.scn, self.con)

        # OVERLAY HERE

        rgb_arr = np.zeros(
            3 * self.viewport.width * self.viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            self.viewport.width * self.viewport.height, dtype=np.float32
        )

        mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.con)

        rgb_img = rgb_arr.reshape(self.viewport.height,
                                  self.viewport.width, 3)

        # original image is upside-down, so flip it
        return rgb_img[::-1, :, :]

    def close(self):
        self.free()
        glfw.terminate()

class Renderer:
    def __init__(self,
                 model,
                 data,
                 height: int = 720,
                 width: int = 720) -> None:

        self._model = model
        self._data = data
        
        self._viewers = {}
        self.viewer = None

        self._height = height
        self._width = width
        
    def _get_viewer(self, render_mode: str):
        """Initializes and returns a viewer class depending on the render_mode
        - `WindowViewer` class for "human" render mode
        - `OffScreenViewer` class for "rgb_array" or "depth_array" render mode
        """
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            if render_mode == "human":
                # self.viewer = WindowViewer(self.model, self.default_human_mode_camera)
                pass
            elif render_mode in {"rgb_array", "depth_array"}:
                self.viewer = OffScreenViewer(self._model, self._data, self._width, self._height)
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

    def render(self, render_mode = "rgb_array", camera_id = -1, overlays=()):
        self.viewer = self._get_viewer(render_mode)
        im = self.viewer.render(camera_id)
        plt.imshow(im)
        plt.show()
