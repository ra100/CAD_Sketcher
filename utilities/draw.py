from collections import deque
from math import sin, cos
from typing import List, Tuple, Union
from enum import Enum

from mathutils import Vector, Matrix

from .. import global_data
from .constants import FULL_TURN

import bpy


# def draw_circle_2d(cx: float, cy: float, r: float, num_segments: int):
#     """NOTE: Not used?"""
#     # circle outline
#     # NOTE: also see gpu_extras.presets.draw_circle_2d
#     theta = FULL_TURN / num_segments

#     # precalculate the sine and cosine
#     c = math.cos(theta)
#     s = math.sin(theta)

#     # start at angle = 0
#     x = r
#     y = 0
#     coords = []
#     for _ in range(num_segments):
#         coords.append((x + cx, y + cy))
#         # apply the rotation matrix
#         t = x
#         x = c * x - s * y
#         y = s * t + c * y
#     coords.append(coords[0])
#     return coords


def draw_rect_2d(cx: float, cy: float, width: float, height: float):
    # NOTE: this currently returns xyz coordinates, might make sense to return 2d coords
    ox = cx - (width / 2)
    oy = cy - (height / 2)
    cz = 0
    return (
        (ox, oy, cz),
        (ox + width, oy, cz),
        (ox + width, oy + height, cz),
        (ox, oy + height, cz),
    )


def draw_rect_3d(origin: Vector, orientation: Vector, width: float) -> List[Vector]:
    mat_rot = global_data.Z_AXIS.rotation_difference(orientation).to_matrix()
    mat = Matrix.Translation(origin) @ mat_rot.to_4x4()
    coords = draw_rect_2d(0, 0, width, width)
    coords = [(mat @ Vector(co))[:] for co in coords]
    return coords


def draw_quad_3d(cx: float, cy: float, cz: float, width: float):
    half_width = width / 2
    coords = (
        (cx - half_width, cy - half_width, cz),
        (cx + half_width, cy - half_width, cz),
        (cx + half_width, cy + half_width, cz),
        (cx - half_width, cy + half_width, cz),
    )
    indices = ((0, 1, 2), (2, 3, 0))
    return coords, indices


def tris_from_quad_ids(*args):
    return (args[0], args[1], args[2]), (args[1], args[2], args[3])


def draw_cube_3d(cx: float, cy: float, cz: float, width: float):
    half_width = width / 2
    coords = []
    for x in (cx - half_width, cx + half_width):
        for y in (cy - half_width, cy + half_width):
            for z in (cz - half_width, cz + half_width):
                coords.append((x, y, z))
    # order: ((-x, -y, -z), (-x, -y, +z), (-x, +y, -z), ...)
    indices = (
        *tris_from_quad_ids(0, 1, 2, 3),
        *tris_from_quad_ids(0, 1, 4, 5),
        *tris_from_quad_ids(1, 3, 5, 7),
        *tris_from_quad_ids(2, 3, 6, 7),
        *tris_from_quad_ids(0, 2, 4, 6),
        *tris_from_quad_ids(4, 5, 6, 7),
    )

    return coords, indices


def coords_circle_2d(x: float, y: float, radius: float, segments: int):
    coords = []
    m = (1.0 / (segments - 1)) * FULL_TURN

    for p in range(segments):
        p1 = x + cos(m * p) * radius
        p2 = y + sin(m * p) * radius
        coords.append((p1, p2))
    return coords


def coords_arc_2d(
    x: float,
    y: float,
    radius: float,
    segments: int,
    angle=FULL_TURN,
    offset: float = 0.0,
    type="LINE_STRIP",
):
    coords = deque()
    segments = max(segments, 1)

    m = (1.0 / segments) * angle

    prev_point = None
    for p in range(segments + 1):
        co_x = x + cos(m * p + offset) * radius
        co_y = y + sin(m * p + offset) * radius
        if type == "LINES":
            if prev_point:
                coords.append(prev_point)
                coords.append((co_x, co_y))
            prev_point = co_x, co_y
        else:
            coords.append((co_x, co_y))
    return coords


# Configuration for thick rendering (Vulkan backend compatibility)
class RenderingConfig:
    """Centralized configuration for thick rendering."""
    THICK_LINE_SCALE = 0.005
    THICK_POINT_SCALE = 0.004
    WORKPLANE_LINE_SCALE = 0.0002
    MIN_SEGMENT_LENGTH = 1e-6
    DEFAULT_DASH_LENGTH = 0.1
    DEFAULT_GAP_LENGTH = 0.05

class RenderMode(Enum):
    """Rendering modes for different entity types."""
    SOLID = "solid"
    DASHED = "dashed"

class ThickRenderer:
    """Unified thick rendering system for Vulkan backend compatibility."""

    @staticmethod
    def _calculate_perpendicular_offset(direction: Vector, width: float, scale_factor: float = RenderingConfig.THICK_LINE_SCALE) -> Vector:
        """Calculate perpendicular offset vector for thick line rendering."""
        if direction.length < RenderingConfig.MIN_SEGMENT_LENGTH:
            return Vector((0, 0, 0))

        # Create perpendicular vector using cross product
        if abs(direction.z) < 0.9:
            perpendicular = direction.cross(Vector((0, 0, 1))).normalized()
        else:
            perpendicular = direction.cross(Vector((1, 0, 0))).normalized()

        return perpendicular * (width * scale_factor)

    @staticmethod
    def _create_quad_geometry(start: Vector, end: Vector, offset: Vector) -> Tuple[List[Tuple], List[Tuple]]:
        """Create quad geometry from start/end points and offset."""
        coords = [
            (start - offset)[:],
            (start + offset)[:],
            (end + offset)[:],
            (end - offset)[:]
        ]
        indices = [(0, 1, 2), (0, 2, 3)]
        return coords, indices

    @classmethod
    def render_point(cls, center: Union[Vector, Tuple], size: float) -> Tuple[List[Tuple], List[Tuple]]:
        """Create thick point geometry using triangles."""
        if not isinstance(center, Vector):
            center = Vector(center)

        half_size = size * RenderingConfig.THICK_POINT_SCALE
        coords = [
            (center.x - half_size, center.y - half_size, center.z),
            (center.x + half_size, center.y - half_size, center.z),
            (center.x + half_size, center.y + half_size, center.z),
            (center.x - half_size, center.y + half_size, center.z)
        ]
        indices = [(0, 1, 2), (0, 2, 3)]
        return coords, indices

    @classmethod
    def render_line(cls, start: Union[Vector, Tuple], end: Union[Vector, Tuple],
                   width: float, mode: RenderMode = RenderMode.SOLID,
                   dash_length: float = RenderingConfig.DEFAULT_DASH_LENGTH,
                   gap_length: float = RenderingConfig.DEFAULT_GAP_LENGTH) -> Tuple[List[Tuple], List[Tuple]]:
        """Unified line rendering for solid and dashed lines."""
        if not isinstance(start, Vector):
            start = Vector(start)
        if not isinstance(end, Vector):
            end = Vector(end)

        direction = end - start
        if direction.length < RenderingConfig.MIN_SEGMENT_LENGTH:
            return [], []

        if mode == RenderMode.SOLID:
            return cls._render_solid_line(start, end, width)
        else:
            return cls._render_dashed_line(start, end, width, dash_length, gap_length)

    @classmethod
    def _render_solid_line(cls, start: Vector, end: Vector, width: float) -> Tuple[List[Tuple], List[Tuple]]:
        """Render a solid thick line."""
        direction_normalized = (end - start).normalized()
        offset = cls._calculate_perpendicular_offset(direction_normalized, width)
        return cls._create_quad_geometry(start, end, offset)

    @classmethod
    def _render_dashed_line(cls, start: Vector, end: Vector, width: float,
                           dash_length: float, gap_length: float) -> Tuple[List[Tuple], List[Tuple]]:
        """Render a dashed thick line."""
        direction = end - start
        total_length = direction.length
        direction_normalized = direction.normalized()
        offset = cls._calculate_perpendicular_offset(direction_normalized, width)

        coords = []
        indices = []
        pattern_length = dash_length + gap_length
        current_pos = 0.0

        while current_pos < total_length:
            dash_end = min(current_pos + dash_length, total_length)
            if dash_end > current_pos:
                start_pos = start + direction_normalized * current_pos
                end_pos = start + direction_normalized * dash_end

                base_idx = len(coords)
                quad_coords, quad_indices = cls._create_quad_geometry(start_pos, end_pos, offset)
                coords.extend(quad_coords)

                adjusted_indices = [(idx[0] + base_idx, idx[1] + base_idx, idx[2] + base_idx)
                                  for idx in quad_indices]
                indices.extend(adjusted_indices)

            current_pos += pattern_length

        return coords, indices

    @classmethod
    def render_line_strip(cls, coords: List[Union[Vector, Tuple]], width: float,
                         mode: RenderMode = RenderMode.SOLID,
                         dash_length: float = RenderingConfig.DEFAULT_DASH_LENGTH,
                         gap_length: float = RenderingConfig.DEFAULT_GAP_LENGTH) -> Tuple[List[Tuple], List[Tuple]]:
        """Unified line strip rendering for solid and dashed line strips."""
        if len(coords) < 2:
            return coords if coords else [], []

        # Convert to Vector objects
        points = [Vector(co) if not isinstance(co, Vector) else co for co in coords]

        all_coords = []
        all_indices = []

        for i in range(len(points) - 1):
            segment_coords, segment_indices = cls.render_line(
                points[i], points[i + 1], width, mode, dash_length, gap_length
            )

            if segment_coords:
                base_idx = len(all_coords)
                all_coords.extend(segment_coords)
                adjusted_indices = [(idx[0] + base_idx, idx[1] + base_idx, idx[2] + base_idx)
                                  for idx in segment_indices]
                all_indices.extend(adjusted_indices)

        return all_coords, all_indices

# Legacy function wrappers for backward compatibility
def draw_thick_point_3d(center, size):
    """Legacy wrapper for thick point rendering."""
    return ThickRenderer.render_point(center, size)

def draw_thick_line_3d(start, end, width):
    """Legacy wrapper for thick line rendering."""
    return ThickRenderer.render_line(start, end, width)

def draw_thick_dashed_line_3d(start, end, width, dash_length=RenderingConfig.DEFAULT_DASH_LENGTH, gap_length=RenderingConfig.DEFAULT_GAP_LENGTH):
    """Legacy wrapper for thick dashed line rendering."""
    return ThickRenderer.render_line(start, end, width, RenderMode.DASHED, dash_length, gap_length)

def draw_thick_line_strip_3d(coords, width):
    """Legacy wrapper for thick line strip rendering."""
    return ThickRenderer.render_line_strip(coords, width)

def draw_thick_dashed_line_strip_3d(coords, width, dash_length=RenderingConfig.DEFAULT_DASH_LENGTH, gap_length=RenderingConfig.DEFAULT_GAP_LENGTH):
    """Legacy wrapper for thick dashed line strip rendering."""
    return ThickRenderer.render_line_strip(coords, width, RenderMode.DASHED, dash_length, gap_length)
