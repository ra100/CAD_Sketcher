import logging
from typing import List

import bpy
import gpu
from mathutils import Vector, Matrix
from bpy.types import PropertyGroup
from gpu_extras.batch import batch_for_shader
from bpy.utils import register_classes_factory

from ..declarations import Operators
from .. import global_data
from ..utilities.draw import draw_rect_2d, draw_thick_line_3d, RenderingConfig
from ..shaders import Shaders
from ..utilities import preferences
from ..solver import Solver
from .base_entity import SlvsGenericEntity
from .utilities import slvs_entity_pointer


logger = logging.getLogger(__name__)


class SlvsWorkplane(SlvsGenericEntity, PropertyGroup):
    """Representation of a plane which is defined by an origin point
    and a normal. Workplanes are used to define the position of 2D entities
    which only store the coordinates on the plane.

    Arguments:
        p1 (SlvsPoint3D): Origin Point of the Plane
        nm (SlvsNormal3D): Normal which defines the orientation
    """

    size = 0.4

    def dependencies(self) -> List[SlvsGenericEntity]:
        return [self.p1, self.nm]

    @property
    def size(self):
        return preferences.get_prefs().workplane_size

    def update(self):
        """Update the workplane's visual representation with thick geometry for Vulkan compatibility."""
        if bpy.app.background:
            return

        # Create thick line geometry for workplane edges (Vulkan compatible)
        rect_coords = draw_rect_2d(0, 0, self.size, self.size)
        rect_coords = [Vector(co) for co in rect_coords]

        # Use a reasonable width for workplane edges
        line_width = 0.2

        # Create border batch for thick lines
        border_coords = []
        border_indices = []
        vertex_offset = 0

        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for start_idx, end_idx in edges:
            start_point = rect_coords[start_idx]
            end_point = rect_coords[end_idx]

            line_coords, line_indices = draw_thick_line_3d(start_point, end_point, line_width)
            border_coords.extend(line_coords)

            # Adjust indices for current vertex offset
            adjusted_indices = [(idx[0] + vertex_offset, idx[1] + vertex_offset, idx[2] + vertex_offset) for idx in line_indices]
            border_indices.extend(adjusted_indices)
            vertex_offset += len(line_coords)

        if border_coords and border_indices:
            self._batch = batch_for_shader(
                self._shader, "TRIS", {"pos": border_coords}, indices=border_indices
            )

        self.is_dirty = False

    def draw(self, context):
        if not self.is_visible(context):
            return

        with gpu.matrix.push_pop():
            scale = context.region_data.view_distance
            gpu.matrix.multiply_matrix(self.matrix_basis)
            gpu.matrix.scale(Vector((scale, scale, scale)))

            col = self.color(context)

            # Let parent class handle borders with its alpha blending
            super().draw(context)

            # Add transparent surface AFTER parent renders borders
            # Use the same alpha settings as parent
            gpu.state.blend_set("ALPHA")

            # Create surface with transparency
            col_surface = col[:-1] + (0.3,) if len(col) >= 4 else (col[0], col[1], col[2], 0.3)

            shader = Shaders.uniform_color_3d()
            shader.bind()
            shader.uniform_float("color", col_surface)

            coords = draw_rect_2d(0, 0, self.size, self.size)
            coords = [Vector(co)[:] for co in coords]
            indices = ((0, 1, 2), (0, 2, 3))
            surface_batch = batch_for_shader(shader, "TRIS", {"pos": coords}, indices=indices)
            surface_batch.draw(shader)

        # Let parent class handle cleanup
        self.restore_opengl_defaults()

    def draw_id(self, context):
        with gpu.matrix.push_pop():
            scale = context.region_data.view_distance
            gpu.matrix.multiply_matrix(self.matrix_basis)
            gpu.matrix.scale(Vector((scale, scale, scale)))

            # Draw borders for selection
            super().draw_id(context)

            # Also draw surface for selection (to allow clicking anywhere on workplane)
            shader = self._id_shader
            shader.bind()
            from ..utilities.index import index_to_rgb
            shader.uniform_float("color", index_to_rgb(self.slvs_index))

            coords = draw_rect_2d(0, 0, self.size, self.size)
            coords = [Vector(co)[:] for co in coords]
            indices = ((0, 1, 2), (0, 2, 3))
            surface_batch = batch_for_shader(shader, "TRIS", {"pos": coords}, indices=indices)
            surface_batch.draw(shader)

    def create_slvs_data(self, solvesys, group=Solver.group_fixed):
        handle = solvesys.addWorkplane(self.p1.py_data, self.nm.py_data, group=group)
        self.py_data = handle

    @property
    def matrix_basis(self):
        mat_rot = self.nm.orientation.to_matrix().to_4x4()
        return Matrix.Translation(self.p1.location) @ mat_rot

    @property
    def normal(self):
        v = global_data.Z_AXIS.copy()
        quat = self.nm.orientation
        v.rotate(quat)
        return v

    def draw_props(self, layout):
        # Display the normals props as they're not drawn in the viewport
        sub = self.nm.draw_props(layout)
        sub.operator(Operators.AlignWorkplaneCursor).index = self.slvs_index
        return sub


slvs_entity_pointer(SlvsWorkplane, "p1")
slvs_entity_pointer(SlvsWorkplane, "nm")

register, unregister = register_classes_factory((SlvsWorkplane,))
