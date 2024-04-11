from abc import ABC
import collections
import os
from pathlib import Path
from typing import Optional
import mujoco
import numpy as np
import glfw
import matplotlib.pyplot as plt

MEDIA_DIR = Path(__file__).parent.joinpath("../../out/media").resolve()

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

_FONT_STYLES = {
    'normal': mujoco.mjtFont.mjFONT_NORMAL,
    'shadow': mujoco.mjtFont.mjFONT_SHADOW,
    'big': mujoco.mjtFont.mjFONT_BIG,
}
_GRID_POSITIONS = {
    'top left': mujoco.mjtGridPos.mjGRID_TOPLEFT,
    'top right': mujoco.mjtGridPos.mjGRID_TOPRIGHT,
    'bottom left': mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
    'bottom right': mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
}

class TextOverlay:
    """A text overlay that can be drawn on top of a camera view."""

    def __init__(self, style="normal"):
        """Initializes a new TextOverlay instance."""
        self._overlays = {}
        self._style = _FONT_STYLES[style]
    
    def add(self, title, body, gridpos):
        
        position = _GRID_POSITIONS[gridpos]
        
        if position not in self._overlays:
            self._overlays[position] = ["", ""]
        self._overlays[position][0] += title + "\n"
        self._overlays[position][1] += body + "\n"

    def add_to_viewport(self, context, rect):
        """Draws the overlay.

        Args:
        context: A `mujoco.MjrContext` pointer.
        rect: A `mujoco.MjrRect`.
        """
        for gridpos, (title, body) in self._overlays.items():
            mujoco.mjr_overlay(self._style,
                                gridpos,
                                rect,
                                title.encode(),
                                body.encode(),
                                context)

