from abc import ABC
import mujoco


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