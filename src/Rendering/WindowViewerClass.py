# autopep8: off
from pathlib import Path
import sys
import time
import mujoco
import numpy as np
import glfw
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__,'..','..').resolve()))
from Rendering.BaseRendererClass import BaseRender

# autopep8: on

class WindowViewer(BaseRender):
    """Class for window rendering in all MuJoCo environments."""

    def __init__(self,
                 model,
                 data,
                 simulation_frame_skip,
                 capture_frames,
                 capture_fps,
                 frame_size):
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

        # width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        glfw.window_hint(glfw.VISIBLE, 1)
        self.window = glfw.create_window(*frame_size,
                                         "mujoco",
                                         None,
                                         None)

        self.width, self.height = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = self.width * 1.0 / window_width

        # set callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(
            self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        super().__init__(model,
                         data,
                         simulation_frame_skip,
                         capture_frames,
                         capture_fps,
                         frame_size)
        glfw.swap_interval(1)

        self.cam.fixedcamid = 0
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
        # self.free()
        pass

    def render(self,
               camera_id,
               overlay = None):
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
                self._model,
                self._data,
                self.vopt,
                mujoco.MjvPerturb(),
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scn,
            )

            # render
            mujoco.mjr_render(self.viewport, self.scn, self.con)

            # OVERLAY HERE
            if overlay is not None and not self._hide_menu:
                overlay.add("FPS", f"{int(1 / self._time_per_render)}", "bottom left")
                overlay.add("Capture Frames", f"{self._capture_frames}: {len(self._frames)}", "bottom left")
                overlay.add_to_viewport(self.con, self.viewport)

            if self._capture_frames:
                if self._nth_render_call % self._nth_frame_capture == 0:
                    self.rgb_arr = np.zeros(
                        3 * self.viewport.width * self.viewport.height, dtype=np.uint8
                    )
                    self.depth_arr = np.zeros(
                        self.viewport.width * self.viewport.height, dtype=np.float32
                    )
                    mujoco.mjr_readPixels(self.rgb_arr, self.depth_arr, self.viewport, self.con)
                    rgb_img = np.copy(self.rgb_arr.reshape(self.viewport.height,
                                        self.viewport.width, 3)[::-1, :, :])
                    self._frames.append(rgb_img)
                self._nth_render_call += 1

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )


        update()


        # clear overlay
        self._overlays.clear()

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
            if self.cam.fixedcamid >= self._model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        elif key == glfw.KEY_H:
            self._hide_menu = not self._hide_menu
        elif key == glfw.KEY_S:
            self.render_movie()
        elif key == glfw.KEY_C:
            new_val = not self._capture_frames
            self._capture_frames = new_val
            self._frames.clear()
        elif key == glfw.KEY_F:
            self.render_image()
        if key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            self.close()

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
            self._model, action, dx / width, dy / height, self.scn, self.cam
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
            self._model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.05 * y_offset,
            self.scn,
            self.cam,
        )