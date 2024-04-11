from abc import ABC
import cv2
import mujoco
import imageio
import numpy as np
import os
import threading

from Rendering.Utils import MEDIA_DIR

class BaseRender(ABC):
    def __init__(
        self, model, data, simulation_frame_skip , width: int, height: int
    ):
        """Render context superclass for offscreen and window rendering."""
        self._model = model
        self._data = data
        self._simulation_frame_skip = simulation_frame_skip

        self._frames = []
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
        
        self.rgb_arr = np.zeros(
            3 * self.viewport.width * self.viewport.height, dtype=np.uint8
        )
        self.depth_arr = np.zeros(
            self.viewport.width * self.viewport.height, dtype=np.float32
        )

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
        
        # MY VARIABLES
        render_steps_per_sec = 1 / (self._model.opt.timestep * self._simulation_frame_skip)
        self._fps_movie = 30
        frame_skip_ratio = render_steps_per_sec / self._fps_movie
        self._frame_capture = round(frame_skip_ratio)
        self._nth_render_call = 0

    def render_movie(self, filename = "tmp.mp4"):
        # filename = f"out-thread-{threading.get_ident()}.mp4"
        output_file = str(MEDIA_DIR/filename)
        
        frame_size = (self.viewport.width, self.viewport.height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self._fps_movie, frame_size)
        
        for frame in self._frames:
            bgr_buffer = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_buffer)
            
        self._frames.clear()
        out.release()
        self._nth_render_call = 0

    def _set_mujoco_buffer(self):
        raise NotImplementedError

    def make_context_current(self):
        raise NotImplementedError

    def close(self):
        """Override close in your rendering subclass to perform any necessary cleanup
        after env.close() is called.
        """
        raise NotImplementedError