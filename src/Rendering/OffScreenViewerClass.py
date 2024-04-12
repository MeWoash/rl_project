import os
from typing import Optional
import mujoco
import numpy as np
import glfw
import matplotlib.pyplot as plt

from Rendering.BaseRendererClass import BaseRender
from Rendering.Utils import _ALL_RENDERERS

class OffScreenViewer(BaseRender):
    """Offscreen rendering class with opengl context."""

    def __init__(self,
                 model,
                 data,
                 simulation_frame_skip,
                 capture_frames,
                 capture_fps,
                 frame_size):

        # We must make GLContext before MjrContext
        self._get_opengl_backend(*frame_size)
        super().__init__(model,
                         data,
                         simulation_frame_skip,
                         capture_frames,
                         capture_fps,
                         frame_size)
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
        overlay = None
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
        if overlay is not None:
            overlay.add_to_viewport(self.con, self.viewport)

        

        mujoco.mjr_readPixels(self.rgb_arr, self.depth_arr, self.viewport, self.con)

        rgb_img = self.rgb_arr.reshape(self.viewport.height,
                                  self.viewport.width, 3)[::-1, :, :]

        # original image is upside-down, so flip it
        return rgb_img

    def close(self):
        self.free()
        glfw.terminate()