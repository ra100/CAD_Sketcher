from collections import deque
from math import sin, cos
from typing import List

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


# Constants for thick rendering (Vulkan backend compatibility)
THICK_LINE_SCALE = 0.005
THICK_POINT_SCALE = 0.004
WORKPLANE_LINE_SCALE = 0.0002  # Much thinner for workplane edges
MIN_SEGMENT_LENGTH = 1e-6
DEFAULT_DASH_LENGTH = 0.1
DEFAULT_GAP_LENGTH = 0.05


def _calculate_perpendicular_offset(direction, width, scale_factor=THICK_LINE_SCALE):
    """Calculate perpendicular offset vector for thick line rendering.

    Args:
        direction: Normalized direction vector
        width: Line width
        scale_factor: Scaling factor for thickness

    Returns:
        Vector: Perpendicular offset vector
    """
    if direction.length < MIN_SEGMENT_LENGTH:
        return Vector((0, 0, 0))

    # Create perpendicular vector using cross product
    # Choose axis that's most perpendicular to direction
    if abs(direction.z) < 0.9:
        perpendicular = direction.cross(Vector((0, 0, 1))).normalized()
    else:
        perpendicular = direction.cross(Vector((1, 0, 0))).normalized()

    return perpendicular * (width * scale_factor)


def _create_quad_geometry(start, end, offset):
    """Create quad geometry from start/end points and offset.

    Args:
        start: Start point (Vector)
        end: End point (Vector)
        offset: Perpendicular offset vector

    Returns:
        tuple: (coords, indices) for quad rendered as triangles
    """
    coords = [
        start - offset,
        start + offset,
        end + offset,
        end - offset
    ]

    # Convert to tuples for GPU batch
    coords = [co[:] for co in coords]

    # Triangle indices for quad (two triangles)
    indices = [(0, 1, 2), (0, 2, 3)]

    return coords, indices


def draw_thick_line_3d(start, end, width):
    """Create thick line geometry using triangles for Vulkan compatibility.

    Args:
        start: Start point (Vector)
        end: End point (Vector)
        width: Line width

    Returns:
        tuple: (coords, indices) for triangle-based thick line
    """
    if not isinstance(start, Vector):
        start = Vector(start)
    if not isinstance(end, Vector):
        end = Vector(end)

    # Check for zero-length line
    direction = end - start
    if direction.length < MIN_SEGMENT_LENGTH:
        return [], []

    direction_normalized = direction.normalized()
    offset = _calculate_perpendicular_offset(direction_normalized, width)

    return _create_quad_geometry(start, end, offset)


def draw_thick_point_3d(center, size):
    """Create thick point geometry using triangles for Vulkan compatibility.

    Args:
        center: Center point (Vector)
        size: Point size

    Returns:
        tuple: (coords, indices) for triangle-based thick point
    """
    if not isinstance(center, Vector):
        center = Vector(center)

    # Create a small quad centered at the point
    half_size = size * THICK_POINT_SCALE

    # Create quad vertices around center point
    coords = [
        (center.x - half_size, center.y - half_size, center.z),
        (center.x + half_size, center.y - half_size, center.z),
        (center.x + half_size, center.y + half_size, center.z),
        (center.x - half_size, center.y + half_size, center.z)
    ]

    # Triangle indices for quad
    indices = [(0, 1, 2), (0, 2, 3)]

    return coords, indices


def draw_thick_line_strip_3d(coords, width):
    """Create thick line strip geometry using triangles for Vulkan compatibility.

    Args:
        coords: List of points forming the line strip
        width: Line width

    Returns:
        tuple: (coords, indices) for triangle-based thick line strip
    """
    if len(coords) < 2:
        return coords if coords else [], []

    # Convert to Vector objects if needed and validate
    points = []
    for co in coords:
        if isinstance(co, Vector):
            points.append(co)
        else:
            points.append(Vector(co))

    # Create thick line segments
    thick_coords = []
    indices = []

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        # Skip zero-length segments
        direction = end - start
        if direction.length < MIN_SEGMENT_LENGTH:
            continue

        direction_normalized = direction.normalized()
        offset = _calculate_perpendicular_offset(direction_normalized, width)

        # Add quad vertices for this segment
        base_idx = len(thick_coords)
        quad_coords, quad_indices = _create_quad_geometry(start, end, offset)

        thick_coords.extend(quad_coords)

        # Adjust indices for current vertex offset
        adjusted_indices = [(idx[0] + base_idx, idx[1] + base_idx, idx[2] + base_idx)
                          for idx in quad_indices]
        indices.extend(adjusted_indices)

    return thick_coords, indices


def draw_thick_dashed_line_3d(start, end, width, dash_length=DEFAULT_DASH_LENGTH, gap_length=DEFAULT_GAP_LENGTH):
    """Create thick dashed line geometry using triangles for Vulkan compatibility.

    Args:
        start: Start point (Vector)
        end: End point (Vector)
        width: Line width
        dash_length: Length of each dash
        gap_length: Length of each gap

    Returns:
        tuple: (coords, indices) for triangle-based thick dashed line
    """
    if not isinstance(start, Vector):
        start = Vector(start)
    if not isinstance(end, Vector):
        end = Vector(end)

    # Calculate line direction and length
    direction = end - start
    total_length = direction.length

    if total_length < MIN_SEGMENT_LENGTH:
        return [], []

    direction_normalized = direction.normalized()
    offset = _calculate_perpendicular_offset(direction_normalized, width)

    # Create dashes
    coords = []
    indices = []
    pattern_length = dash_length + gap_length
    current_pos = 0.0

    while current_pos < total_length:
        # Create dash segment
        dash_start = current_pos
        dash_end = min(current_pos + dash_length, total_length)

        if dash_end > dash_start:
            # Calculate positions along the line
            start_pos = start + direction_normalized * dash_start
            end_pos = start + direction_normalized * dash_end

            # Add quad geometry for this dash
            base_idx = len(coords)
            quad_coords, quad_indices = _create_quad_geometry(start_pos, end_pos, offset)

            coords.extend(quad_coords)
            adjusted_indices = [(idx[0] + base_idx, idx[1] + base_idx, idx[2] + base_idx)
                              for idx in quad_indices]
            indices.extend(adjusted_indices)

        current_pos += pattern_length

    return coords, indices


def draw_thick_dashed_line_strip_3d(coords, width, dash_length=DEFAULT_DASH_LENGTH, gap_length=DEFAULT_GAP_LENGTH):
    """Create thick dashed line strip geometry using triangles for Vulkan compatibility.

    Args:
        coords: List of points forming the line strip
        width: Line width
        dash_length: Length of each dash
        gap_length: Length of each gap

    Returns:
        tuple: (coords, indices) for triangle-based thick dashed line strip
    """
    if len(coords) < 2:
        return coords if coords else [], []

    # Convert to Vector objects if needed
    points = []
    for co in coords:
        if isinstance(co, Vector):
            points.append(co)
        else:
            points.append(Vector(co))

    # Create thick dashed line segments
    thick_coords = []
    indices = []

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        # Calculate segment length and direction
        direction = end - start
        segment_length = direction.length

        if segment_length < MIN_SEGMENT_LENGTH:
            continue

        direction_normalized = direction.normalized()
        offset = _calculate_perpendicular_offset(direction_normalized, width)

        # Create dashes along this segment
        pattern_length = dash_length + gap_length
        current_pos = 0.0

        while current_pos < segment_length:
            # Create dash segment
            dash_start = current_pos
            dash_end = min(current_pos + dash_length, segment_length)

            if dash_end > dash_start:
                # Calculate positions along the segment
                start_pos = start + direction_normalized * dash_start
                end_pos = start + direction_normalized * dash_end

                # Add quad geometry for this dash
                base_idx = len(thick_coords)
                quad_coords, quad_indices = _create_quad_geometry(start_pos, end_pos, offset)

                thick_coords.extend(quad_coords)
                adjusted_indices = [(idx[0] + base_idx, idx[1] + base_idx, idx[2] + base_idx)
                                  for idx in quad_indices]
                indices.extend(adjusted_indices)

            current_pos += pattern_length

    return thick_coords, indices
