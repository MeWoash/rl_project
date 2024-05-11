# autopep8: off
from abc import ABC
from ast import Raise
from math import ceil
from pathlib import Path
import sys
import cv2
import mujoco
import numpy as np
import sys
sys.path.append(str(Path(__file__,'..','..').resolve()))
from PathsConfig import MEDIA_DIR

# autopep8: on

class BaseRender(ABC):
    def __init__(
        self,
        env,
        frame_size,
    ):
        """Render context superclass for offscreen and window rendering."""
        self.env = env
        self._capture_frames = False
        self._capture_fps = 24
        
        width, height = frame_size

        self._frames = []
        self._overlays = {}
        
        buffer_width = self.env.model.vis.global_.offwidth
        buffer_height = self.env.model.vis.global_.offheight
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
        self.scn = mujoco.MjvScene(self.env.model, 1000)
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        self.make_context_current()

        # Keep in Mujoco Context
        self.con = mujoco.MjrContext(
            self.env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._set_mujoco_buffer()
        
        # MY VARIABLES
        render_steps_per_sec = 1 / (self.env.model.opt.timestep * self.env.simulation_frame_skip)
        frame_skip_ratio = render_steps_per_sec / self._capture_fps
        self._nth_frame_capture = ceil(frame_skip_ratio)
        self._nth_render_call = 0

    def set_capturing_frames(self, value: bool):
        match value:
            case True:
                self._capture_frames = True
            case False:
                self._capture_frames = False
            case _:
                print("Wrong Argument!")
                
    
    def render_movie(self, filename = "tmp.mp4"):
        if self._capture_frames == False:
            print("To save video enable capturing frames [c].")
            return
            
        # filename = f"out-thread-{threading.get_ident()}.mp4"
        output_file = str(Path(MEDIA_DIR,filename))
        
        frame_size = (self.viewport.width, self.viewport.height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self._capture_fps, frame_size)
        
        for frame in self._frames:
            bgr_buffer = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_buffer)
            
        self._frames.clear()
        out.release()
        self._nth_render_call = 0
        print(f"Output file at: {output_file}")
        
    def render_image(self, filename = "tmp.jpg"):
        output_file = str(MEDIA_DIR/filename)
        
        self.rgb_arr = np.zeros(
                        3 * self.viewport.width * self.viewport.height, dtype=np.uint8
                    )
        self.depth_arr = np.zeros(
            self.viewport.width * self.viewport.height, dtype=np.float32
        )
        mujoco.mjr_readPixels(self.rgb_arr, self.depth_arr, self.viewport, self.con)
        rgb_img = np.copy(self.rgb_arr.reshape(self.viewport.height,
                            self.viewport.width, 3)[::-1, :, :])
        bgr_buffer = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_file, bgr_buffer)
        print(f"Output file at: {output_file}")
        return rgb_img

    def _set_mujoco_buffer(self):
        raise NotImplementedError

    def make_context_current(self):
        raise NotImplementedError

    def close(self):
        """Override close in your rendering subclass to perform any necessary cleanup
        after env.close() is called.
        """
        raise NotImplementedError