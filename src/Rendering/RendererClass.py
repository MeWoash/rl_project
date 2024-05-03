import matplotlib.pyplot as plt

from Rendering.OffScreenViewerClass import OffScreenViewer
from Rendering.WindowViewerClass import WindowViewer

class Renderer:
    def __init__(self,
                 model,
                 data,
                 simulation_frame_skip,
                 capture_frames = True,
                 capture_fps = 24,
                 frame_size = (480, 480), #  Width, Height
                 ) -> None:

        self._model = model
        self._data = data
        self._simulation_frame_skip = simulation_frame_skip
        self._capture_frames= capture_frames
        self._capture_fps= capture_fps
        self._frame_size= frame_size
        
        self._viewers = {}
        self.viewer = None

        
    def _get_viewer(self, render_mode: str):
        """Initializes and returns a viewer class depending on the render_mode
        - `WindowViewer` class for "human" render mode
        - `OffScreenViewer` class for "rgb_array" or "depth_array" render mode
        """
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            if render_mode == "human":
                self.viewer = WindowViewer(self._model,
                                           self._data,
                                           self._simulation_frame_skip,
                                           self._capture_frames,
                                           self._capture_fps,
                                           (1080, 720))
                pass
            elif render_mode in {"rgb_array", "depth_array", "none"}:
                self.viewer = OffScreenViewer(self._model,
                                              self._data,
                                              self._simulation_frame_skip,
                                              self._capture_frames,
                                              self._capture_fps,
                                              self._frame_size)
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
        
        return self.viewer.render(camera_id, overlays)
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
