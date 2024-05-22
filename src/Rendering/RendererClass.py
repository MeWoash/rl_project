# autopep8: off
from pathlib import Path
import sys

sys.path.append(str(Path(__file__,'..','..').resolve()))

from Rendering.OffScreenViewerClass import OffScreenViewer
from Rendering.WindowViewerClass import WindowViewer
# autopep8: on

class Renderer:
    def __init__(self, env) -> None:
        
        self.env = env
        self.env.model = env.model
        self.env.data = env.data
        
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
                self.viewer = WindowViewer(self.env,
                                           (1280, 720))
                pass
            elif render_mode in {"rgb_array", "depth_array", "none"}:
                self.viewer = OffScreenViewer(self.env,
                                              (480, 480))
            else:
                raise AttributeError(
                    f"Unexpected mode: {render_mode}, expected modes: human, rgb_array, depth_array or none"
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
