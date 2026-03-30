# render_dataset_polyhedra.py
import bpy
import bmesh
import os
import sys
import json
import math
import random
from mathutils import Vector, Quaternion

#from root run:  /Applications/Blender.app/Contents/MacOS/Blender -b -P stimuli/render_dataset_polyhedra.py -- --config stimuli/config.json

# ----------------------------
# CLI + utils
# ----------------------------
def parse_args(argv):
    if "--" not in argv:
        return {}
    idx = argv.index("--") + 1
    args = argv[idx:]
    out = {"config": None}
    it = iter(args)
    for a in it:
        if a == "--config":
            out["config"] = next(it)
    return out


def resolve_path(config_path: str, p: str) -> str:
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return os.path.abspath(p)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.abspath(os.path.join(config_dir, p))


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    try:
        bpy.ops.outliner.orphans_purge(do_recursive=True)
    except Exception:
        pass


def linspace(a: float, b: float, n: int):
    if n <= 1:
        return [float(a)]
    a = float(a)
    b = float(b)
    step = (b - a) / (n - 1)
    return [a + k * step for k in range(n)]


# ----------------------------
# Render settings
# ----------------------------
def set_render_settings(cfg):
    scene = bpy.context.scene

    res = cfg.get("resolution", [224, 224])
    scene.render.resolution_x = int(res[0])
    scene.render.resolution_y = int(res[1])
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    engine = cfg.get("engine", "CYCLES").upper()
    if engine not in {"CYCLES", "EEVEE"}:
        engine = "CYCLES"
    scene.render.engine = engine

    if engine == "CYCLES":
        scene.cycles.samples = int(cfg.get("samples", 128))
        scene.cycles.use_denoising = bool(cfg.get("use_denoising", True))
        scene.cycles.device = cfg.get("device", "CPU").upper()
        scene.cycles.max_bounces = int(cfg.get("max_bounces", 12))
        scene.cycles.transparent_max_bounces = int(cfg.get("transparent_max_bounces", 8))
        scene.cycles.diffuse_bounces = int(cfg.get("diffuse_bounces", 3))
        scene.cycles.glossy_bounces = int(cfg.get("glossy_bounces", 4))
        scene.cycles.transmission_bounces = int(cfg.get("transmission_bounces", 8))
    else:
        scene.eevee.taa_render_samples = max(16, int(cfg.get("samples", 128)) // 4)
        scene.eevee.use_ssr = True
        scene.eevee.use_ssr_refraction = True


# ----------------------------
# Materials / camera / scene
# ----------------------------
def make_principled_material(name: str):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out.location = (300, 0)
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, bsdf


def point_camera_at(camera_obj, target_loc):
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=tuple(target_loc))
    target = bpy.context.object
    target.name = "CamTarget"

    c = camera_obj.constraints.new(type="TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    return target


def add_camera(cfg):
    bpy.ops.object.camera_add(
        location=tuple(cfg.get("cam_location", [0.0, -8.0, 4.5]))
    )
    cam = bpy.context.object
    cam.data.lens = float(cfg.get("cam_focal_length_mm", 45.0))
    bpy.context.scene.camera = cam

    point_camera_at(cam, cfg.get("cam_target", [0.0, 0.0, 1.8]))
    return cam


def add_room_corner(cfg):
    floor_size = float(cfg.get("floor_size", 10.0))
    wall_height = float(cfg.get("wall_height", 12.0))
    wall_thickness = float(cfg.get("wall_thickness", 0.05))
    wall_extent = float(cfg.get("wall_extent", 30.0))

    # Floor centered at origin
    bpy.ops.mesh.primitive_plane_add(size=floor_size, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "Floor"

    floor_mat, floor_bsdf = make_principled_material("MAT_FloorWood")
    nodes = floor_mat.node_tree.nodes
    links = floor_mat.node_tree.links

    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    noise = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")

    texcoord.location = (-900, 0)
    mapping.location = (-700, 0)
    noise.location = (-500, 0)
    ramp.location = (-250, 0)

    mapping.inputs["Scale"].default_value = (6.0, 1.2, 1.0)
    noise.inputs["Scale"].default_value = 10.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.5

    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0.55, 0.42, 0.28, 1.0)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (0.72, 0.58, 0.40, 1.0)

    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], floor_bsdf.inputs["Base Color"])

    floor_bsdf.inputs["Roughness"].default_value = float(cfg.get("floor_roughness", 0.25))
    floor_bsdf.inputs["Specular"].default_value = float(cfg.get("floor_specular", 0.35))
    floor.data.materials.append(floor_mat)

    # Shared wall material
    wall_mat, wall_bsdf = make_principled_material("MAT_Walls")
    wall_bsdf.inputs["Base Color"].default_value = tuple(
        cfg.get("wall_color_rgba", [0.05, 0.05, 0.05, 1.0])
    )
    wall_bsdf.inputs["Roughness"].default_value = float(cfg.get("wall_roughness", 1.0))

    # Back wall
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(0.0, floor_size / 2.0, wall_height / 2.0 - 3.0)
    )
    wall_back = bpy.context.object
    wall_back.name = "Wall_Back"
    wall_back.scale = (wall_extent / 2.0, wall_thickness / 2.0, wall_height / 2.0)
    wall_back.data.materials.append(wall_mat)

    # Side wall
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(-floor_size / 2.0, 0.0, wall_height / 2.0 - 3.0)
    )
    wall_side = bpy.context.object
    wall_side.name = "Wall_Side"
    wall_side.scale = (wall_thickness / 2.0, wall_extent / 2.0, wall_height / 2.0)
    wall_side.data.materials.append(wall_mat)

    return floor, wall_back, wall_side


def hide_light_from_camera(light_obj):
    for attr in ["visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter"]:
        if hasattr(light_obj, attr):
            try:
                if attr == "visible_camera":
                    setattr(light_obj, attr, False)
            except Exception:
                pass

    if hasattr(light_obj, "cycles_visibility"):
        try:
            light_obj.cycles_visibility.camera = False
        except Exception:
            pass


# def add_lighting(cfg):
#     key_energy = float(cfg.get("key_energy", 2200.0))
#     fill_energy = float(cfg.get("fill_energy", 700.0))
#     back_energy = float(cfg.get("back_energy", 700.0))

#     bpy.ops.object.light_add(type="AREA", location=(4.0, -5.5, 6.0))
#     key = bpy.context.object
#     key.data.energy = key_energy
#     key.data.size = float(cfg.get("key_size", 4.5))
#     key.rotation_euler = (math.radians(58), 0.0, math.radians(28))
#     hide_light_from_camera(key)

#     bpy.ops.object.light_add(type="AREA", location=(-3.5, -5.0, 5.0))
#     fill = bpy.context.object
#     fill.data.energy = fill_energy
#     fill.data.size = float(cfg.get("fill_size", 5.0))
#     fill.rotation_euler = (math.radians(62), 0.0, math.radians(-18))
#     hide_light_from_camera(fill)

#     bpy.ops.object.light_add(type="POINT", location=(0.0, 4.5, 8.0))
#     back = bpy.context.object
#     back.data.energy = back_energy
#     hide_light_from_camera(back)

#     return key, fill, back

def add_lighting(cfg):
    key_energy = float(cfg.get("key_energy", 2200.0))
    fill_energy = float(cfg.get("fill_energy", 900.0))
    back_energy = float(cfg.get("back_energy", 700.0))

    # KEY: much further left/up, aimed inward
    bpy.ops.object.light_add(type="AREA", location=(-6.5, -5.0, 8.5))
    key = bpy.context.object
    key.data.energy = key_energy
    key.data.size = float(cfg.get("key_size", 4.0))
    key.rotation_euler = (math.radians(78), 0.0, math.radians(-42))
    hide_light_from_camera(key)

    # FILL: softer front-right support
    bpy.ops.object.light_add(type="AREA", location=(3.0, -4.5, 5.5))
    fill = bpy.context.object
    fill.data.energy = fill_energy
    fill.data.size = float(cfg.get("fill_size", 5.5))
    fill.rotation_euler = (math.radians(62), 0.0, math.radians(20))
    hide_light_from_camera(fill)

    # BACK / rim
    bpy.ops.object.light_add(type="POINT", location=(0.0, 4.5, 7.0))
    back = bpy.context.object
    back.data.energy = back_energy
    hide_light_from_camera(back)

    # Small gloss kicker to bring back highlights on shiny objects
    bpy.ops.object.light_add(type="AREA", location=(2.0, -2.5, 4.5))
    gloss = bpy.context.object
    gloss.data.energy = float(cfg.get("gloss_energy", 250.0))
    gloss.data.size = float(cfg.get("gloss_size", 1.5))
    gloss.rotation_euler = (math.radians(80), 0.0, math.radians(10))
    hide_light_from_camera(gloss)

    return key, fill, back

def rotate_object_to_rest_on_face(obj, face_index: int, yaw_deg: float = 0.0):
    """
    Rotate object so the chosen face becomes the supporting face on the floor.

    Steps:
    1) Align chosen face normal with global Z
    2) Check whether the chosen face is above the object center
       If so, flip object by 180 degrees around X
    3) Add yaw around global Z
    """
    bpy.context.view_layer.update()

    if obj.type != "MESH" or obj.data is None or len(obj.data.polygons) == 0:
        return

    face = obj.data.polygons[int(face_index) % len(obj.data.polygons)]
    n_local = face.normal.normalized()

    # First make the face horizontal
    target = Vector((0.0, 0.0, 1.0))
    q = n_local.rotation_difference(target)

    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = q
    bpy.context.view_layer.update()

    # Compute whether chosen face lies above or below object center
    face = obj.data.polygons[int(face_index) % len(obj.data.polygons)]
    face_center_world = obj.matrix_world @ face.center
    obj_center_world = obj.matrix_world @ Vector((0.0, 0.0, 0.0))

    # If the face is above the object's center, flip it to the bottom
    if face_center_world.z > obj_center_world.z:
        q_flip = Quaternion((1.0, 0.0, 0.0), math.pi)
        obj.rotation_quaternion = q_flip @ obj.rotation_quaternion
        bpy.context.view_layer.update()

    # Convert to Euler and add yaw
    e = obj.rotation_quaternion.to_euler()
    e.z += math.radians(float(yaw_deg))
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = e
    bpy.context.view_layer.update()


# ----------------------------
# Polyhedra
# ----------------------------
def _icosahedron_data():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        (-1,  phi,  0),
        ( 1,  phi,  0),
        (-1, -phi,  0),
        ( 1, -phi,  0),
        ( 0, -1,  phi),
        ( 0,  1,  phi),
        ( 0, -1, -phi),
        ( 0,  1, -phi),
        ( phi,  0, -1),
        ( phi,  0,  1),
        (-phi,  0, -1),
        (-phi,  0,  1),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    return verts, faces


def _dodecahedron_data():
    """
    Construct dodecahedron as dual of icosahedron:
    - one dodeca vertex per icosa face centroid
    - one dodeca pentagon per icosa vertex
    """
    ico_verts_raw, ico_faces = _icosahedron_data()
    ico_verts = [Vector(v) for v in ico_verts_raw]

    # Dodeca vertices = face centroids of icosa
    dodeca_verts = []
    for f in ico_faces:
        c = (ico_verts[f[0]] + ico_verts[f[1]] + ico_verts[f[2]]) / 3.0
        dodeca_verts.append(c)

    # For each icosa vertex, find adjacent icosa faces; these form one pentagonal dodeca face
    dodeca_faces = []
    for vi, v in enumerate(ico_verts):
        adjacent_face_indices = [fi for fi, f in enumerate(ico_faces) if vi in f]
        n = v.normalized()

        # build tangent basis at this vertex
        ref = Vector((0.0, 0.0, 1.0))
        if abs(n.dot(ref)) > 0.9:
            ref = Vector((0.0, 1.0, 0.0))
        u = n.cross(ref).normalized()
        w = n.cross(u).normalized()

        face_angles = []
        for fi in adjacent_face_indices:
            c = dodeca_verts[fi]
            t = c - n * c.dot(n)
            ang = math.atan2(t.dot(w), t.dot(u))
            face_angles.append((ang, fi))

        face_angles.sort()
        ordered = [fi for _, fi in face_angles]
        dodeca_faces.append(tuple(ordered))

    return [tuple(v) for v in dodeca_verts], dodeca_faces


def polyhedron_data(shape_name: str):
    shape_name = shape_name.lower()

    if shape_name == "tetrahedron":
        verts = [
            ( 1,  1,  1),
            (-1, -1,  1),
            (-1,  1, -1),
            ( 1, -1, -1),
        ]
        faces = [
            (0, 1, 2),
            (0, 3, 1),
            (0, 2, 3),
            (1, 3, 2),
        ]
        return verts, faces

    if shape_name == "cube":
        verts = [
            (-1, -1, -1),
            ( 1, -1, -1),
            ( 1,  1, -1),
            (-1,  1, -1),
            (-1, -1,  1),
            ( 1, -1,  1),
            ( 1,  1,  1),
            (-1,  1,  1),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ]
        return verts, faces

    if shape_name == "octahedron":
        verts = [
            ( 1,  0,  0),
            (-1,  0,  0),
            ( 0,  1,  0),
            ( 0, -1,  0),
            ( 0,  0,  1),
            ( 0,  0, -1),
        ]
        faces = [
            (0, 2, 4),
            (2, 1, 4),
            (1, 3, 4),
            (3, 0, 4),
            (2, 0, 5),
            (1, 2, 5),
            (3, 1, 5),
            (0, 3, 5),
        ]
        return verts, faces

    if shape_name == "icosahedron":
        return _icosahedron_data()

    if shape_name == "dodecahedron":
        return _dodecahedron_data()

    raise ValueError(f"Unknown shape_name: {shape_name}")


def create_polyhedron(shape_name: str):
    verts, faces = polyhedron_data(shape_name)

    mesh = bpy.data.meshes.new(f"{shape_name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(f"{shape_name}_obj", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Flat shading is usually nicer for polyhedra
    for p in obj.data.polygons:
        p.use_smooth = False

    return obj


# ----------------------------
# Object geometry / material
# ----------------------------
def normalize_object_size(obj, target_max_dim: float):
    """
    Normalize object size by volume^(1/3) rather than max bounding-box dimension.
    This equalises the visual 'mass' of different polyhedra types:
    a tetrahedron, cube, and icosahedron with the same scale parameter
    will appear the same size, not just have the same bounding box.
    target_max_dim controls volume^(1/3) of the normalised mesh.
    """
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    volume = abs(bm.calc_volume())
    bm.free()

    if volume > 0:
        vol_cbrt = volume ** (1.0 / 3.0)
        s = float(target_max_dim) / vol_cbrt
        obj.scale = (obj.scale[0] * s, obj.scale[1] * s, obj.scale[2] * s)
        bpy.context.view_layer.update()


def place_object_on_floor(obj, pos_xy, floor_z=0.0, clearance=0.001):
    """
    Place object so its lowest actual mesh vertex sits on the floor.
    """
    x, y = float(pos_xy[0]), float(pos_xy[1])
    obj.location = (x, y, 0.0)
    bpy.context.view_layer.update()

    if obj.type != "MESH" or obj.data is None or len(obj.data.vertices) == 0:
        return

    minz = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
    obj.location.z += (floor_z - minz + clearance)
    bpy.context.view_layer.update()


def apply_object_pose(obj, shape_name: str, yaw_deg: float, scale_mul: float, face_index: int = 0):
    """
    First scale, then align a chosen face to the floor, then add yaw.
    """
    s = float(scale_mul)
    obj.scale = (obj.scale[0] * s, obj.scale[1] * s, obj.scale[2] * s)
    bpy.context.view_layer.update()

    rotate_object_to_rest_on_face(obj, face_index=face_index, yaw_deg=yaw_deg)
    bpy.context.view_layer.update()

def apply_material_latents(obj, glossiness: float, transparency: float, rgb, engine: str):
    gloss = float(glossiness)
    trans = float(transparency)
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])

    mat = bpy.data.materials.new(name="MAT_Object")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (500, 0)

    mix = nodes.new("ShaderNodeMixShader")
    mix.location = (300, 0)

    # latent -> renderer mapping
    r_min = 0.05
    rough_mapped = r_min ** gloss          # log-linear: [1.0, 0.47, 0.22, 0.11, 0.05]
    trans_mapped = 0.95 * (trans ** 0.3) 

    opaque = nodes.new("ShaderNodeBsdfPrincipled")
    opaque.location = (0, 120)
    opaque.inputs["Base Color"].default_value = (r, g, b, 1.0)
    opaque.inputs["Roughness"].default_value = 0.0
    opaque.inputs["Specular"].default_value = 0.9
    opaque.inputs["Transmission"].default_value = 0.0
    opaque.inputs["IOR"].default_value = 1.45

    glass = nodes.new("ShaderNodeBsdfPrincipled")
    glass.location = (0, -120)
    glass.inputs["Base Color"].default_value = (r, g, b, 1.0)
    glass.inputs["Roughness"].default_value = rough_mapped
    glass.inputs["Specular"].default_value = 0.9
    glass.inputs["Transmission"].default_value = 1.0
    glass.inputs["IOR"].default_value = 1.45
    glass.inputs["Transmission Roughness"].default_value = 0.0

    mix.inputs["Fac"].default_value = trans_mapped
    links.new(opaque.outputs["BSDF"], mix.inputs[1])
    links.new(glass.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])

    if engine.upper() == "EEVEE":
        mat.blend_method = "BLEND"
        mat.shadow_method = "HASHED"
        mat.use_screen_refraction = True
        try:
            mat.refraction_depth = 0.4
        except Exception:
            pass

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

# def apply_material_latents(obj, glossiness: float, transparency: float, rgb, engine: str):
#     gloss = float(glossiness)
#     trans = float(transparency)
#     r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])

#     mat = bpy.data.materials.new(name="MAT_Object")
#     mat.use_nodes = True
#     nodes = mat.node_tree.nodes
#     links = mat.node_tree.links
#     for n in list(nodes):
#         nodes.remove(n)

#     out = nodes.new("ShaderNodeOutputMaterial")
#     out.location = (300, 0)

#     bsdf = nodes.new("ShaderNodeBsdfPrincipled")
#     bsdf.location = (0, 0)

#     r_min = 0.05
#     rough_mapped = r_min ** gloss          # [1.0, 0.47, 0.22, 0.11, 0.05]
#     alpha_mapped = 1.0 - 0.80 * (trans ** 0.5)  # [1.0, 0.64, 0.47, 0.36, 0.20]

#     bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
#     bsdf.inputs["Roughness"].default_value = rough_mapped
#     bsdf.inputs["Specular"].default_value = 0.9
#     bsdf.inputs["Alpha"].default_value = alpha_mapped
#     bsdf.inputs["IOR"].default_value = 1.45

#     links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

#     mat.blend_method = "BLEND"
#     mat.shadow_method = "HASHED"

#     if obj.data.materials:
#         obj.data.materials[0] = mat
#     else:
#         obj.data.materials.append(mat)

def lab_to_xyz(L, a, b):
    """
    CIELAB -> XYZ
    Reference white: D65
    """
    # D65 reference white
    Xn, Yn, Zn = 95.047, 100.000, 108.883

    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    def f_inv(t):
        delta = 6.0 / 29.0
        if t > delta:
            return t ** 3
        return 3 * (delta ** 2) * (t - 4.0 / 29.0)

    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)
    return X, Y, Z


def xyz_to_srgb(X, Y, Z):
    """
    XYZ -> sRGB in [0,1]
    Assumes D65 white point.
    """
    # scale XYZ from 0..100 to 0..1
    X /= 100.0
    Y /= 100.0
    Z /= 100.0

    # linear RGB
    r_lin =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin =  0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gamma_encode(c):
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055

    rgb = [gamma_encode(c) for c in (r_lin, g_lin, b_lin)]
    return rgb


def lab_to_rgb(L, a, b):
    """
    Convert Lab to sRGB.
    Returns:
      rgb: list of floats (can be outside [0,1])
      in_gamut: bool
    """
    X, Y, Z = lab_to_xyz(L, a, b)
    rgb = xyz_to_srgb(X, Y, Z)
    in_gamut = all(0.0 <= c <= 1.0 for c in rgb)
    return rgb, in_gamut

def max_chroma_for_hue(L: float, hue_rad: float, max_tries: int = 50) -> float:
    """Binary search for max in-gamut chroma at this L and hue angle."""
    lo, hi = 0.0, 120.0
    for _ in range(max_tries):
        mid = (lo + hi) / 2.0
        a = mid * math.cos(hue_rad)
        b = mid * math.sin(hue_rad)
        _, in_gamut = lab_to_rgb(L, a, b)
        if in_gamut:
            lo = mid
        else:
            hi = mid
    return lo


def sample_lab_color(cfg, rng, max_tries: int = 200):
    """
    Sample uniformly in polar Lab (hue angle, chroma) then convert.
    Guarantees vivid, hue-diverse, in-gamut colors.
    """
    L_range = cfg.get("L_range", [55.0, 75.0])
    sample_L = bool(cfg.get("sample_L", False))
    fixed_L = float(cfg.get("fixed_L", 65.0))
    min_chroma = float(cfg.get("min_chroma", 30.0))  # avoid muddy colors

    for _ in range(max_tries):
        L = rng.uniform(float(L_range[0]), float(L_range[1])) if sample_L else fixed_L
        hue = rng.uniform(0.0, 2.0 * math.pi)
        c_max = max_chroma_for_hue(L, hue)

        if c_max < min_chroma:
            continue  # this hue is too constrained at this L, resample

        chroma = rng.uniform(min_chroma, c_max)
        a = chroma * math.cos(hue)
        b = chroma * math.sin(hue)
        rgb, in_gamut = lab_to_rgb(L, a, b)

        if in_gamut:
            return {"lab": [L, a, b], "rgb": rgb}

    # fallback — should rarely trigger
    L = fixed_L
    hue = rng.uniform(0.0, 2.0 * math.pi)
    c_max = max_chroma_for_hue(L, hue)
    chroma = min(min_chroma, c_max * 0.8)
    a = chroma * math.cos(hue)
    b = chroma * math.sin(hue)
    rgb, _ = lab_to_rgb(L, a, b)
    rgb = [min(1.0, max(0.0, c)) for c in rgb]
    return {"lab": [L, a, b], "rgb": rgb}

# ----------------------------
# Latents
# ----------------------------
def sample_random_latents(cfg, rng: random.Random):
    def u(key, lohi):
        a, b = cfg.get(key, lohi)
        return rng.uniform(float(a), float(b))

    shape_names = cfg.get(
        "shape_names",
        ["tetrahedron", "cube", "octahedron", "dodecahedron", "icosahedron"],
    )
    shape_name = rng.choice(shape_names)

    face_counts = {
        "tetrahedron": 4,
        "cube": 6,
        "octahedron": 8,
        "dodecahedron": 12,
        "icosahedron": 20,
    }

    color_sample = sample_lab_color(cfg, rng)

    return {
        "shape_name": shape_name,
        "face_index": rng.randrange(face_counts[shape_name]),
        "glossiness": u("glossiness_range", [0.0, 1.0]),
        "transparency": u("transparency_range", [0.0, 1.0]),
        "lab": color_sample["lab"],
        "rgb": color_sample["rgb"],
        "scale": u("obj_scale_range", [0.3, 0.7]),
        "pos_xy": [
            rng.uniform(*cfg.get("obj_x_range", [-2.5, 2.5])),
            rng.uniform(*cfg.get("obj_y_range", [-1.5, 2.5])),
        ],
        "yaw_deg": u("yaw_deg_range", [0.0, 360.0]),
    }

def build_filename(uid: str, z: dict):
    shape_codes = {
        "tetrahedron": "tet",
        "cube": "cub",
        "octahedron": "oct",
        "dodecahedron": "dod",
        "icosahedron": "ico",
    }

    shape = shape_codes.get(z["shape_name"], z["shape_name"][:3])

    face = int(z.get("face_index", 0))
    yaw = int(round(z["yaw_deg"]))
    size = int(round(z["scale"] * 100))

    L = int(round(z["lab"][0]))
    a = int(round(z["lab"][1]))
    b = int(round(z["lab"][2]))

    trans = int(round(z["transparency"] * 100))
    gloss = int(round(z["glossiness"] * 100))

    x = int(round(z["pos_xy"][0] * 100))
    y = int(round(z["pos_xy"][1] * 100))

    fname = (
        f"{uid}"
        f"_sh{shape}"
        f"_f{face:02d}"
        f"_yaw{yaw:03d}"
        f"_s{size:03d}"
        f"_lab{L:03d}-{a:+04d}-{b:+04d}"
        f"_t{trans:03d}"
        f"_gl{gloss:03d}"
        f"_xy{x:+05d}-{y:+05d}"
        ".png"
    )
    return fname

# ----------------------------
# Rendering
# ----------------------------
def render_one(cfg, engine, img_dir, meta_path, uid, z, tag=None, target_max_dim=None):
    clear_scene()
    set_render_settings(cfg)
    add_room_corner(cfg)
    add_camera(cfg)
    add_lighting(cfg)

    shape_name = z["shape_name"]
    obj = create_polyhedron(shape_name)

    if target_max_dim is None:
        target_max_dim = float(cfg.get("target_max_dim", 2.8))
    normalize_object_size(obj, float(target_max_dim))

    apply_object_pose(
    obj,
    shape_name=z["shape_name"],
    yaw_deg=z["yaw_deg"],
    scale_mul=z["scale"],
    face_index=z.get("face_index", 0),
)
    place_object_on_floor(obj, z["pos_xy"], floor_z=0.0)

    apply_material_latents(
        obj=obj,
        glossiness=z["glossiness"],
        transparency=z["transparency"],
        rgb=z["rgb"],
        engine=engine,
    )

    fname = build_filename(uid, z)

    if tag is not None:
        fname = fname.replace(".png", f"_{tag}.png")
        
    out_path = os.path.join(img_dir, fname)
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

    record = {
        "id": uid,
        "tag": tag,
        "shape_name": shape_name,
        "image_file": os.path.join("images", fname),
        "z": z,
        "engine": engine,
    }
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    try:
        bpy.data.objects.remove(obj, do_unlink=True)
    except Exception:
        pass


def main():
    args = parse_args(sys.argv)
    config_path = args.get("config")
    if not config_path:
        raise SystemExit("Blender script missing --config path.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out_dir = resolve_path(config_path, cfg["out_dir"])

    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "metadata.jsonl")

    engine = cfg.get("engine", "CYCLES").upper()
    seed = int(cfg.get("seed", 0))
    rng = random.Random(seed)

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("")

    # ----------------------------
    # Random dataset
    # ----------------------------
    random_n = int(cfg.get("random_n", 100))
    for i in range(random_n):
        uid = f"{i:06d}"
        z = sample_random_latents(cfg, rng)
        render_one(cfg, engine, img_dir, meta_path, uid, z, tag="random")

    # ----------------------------
    # Grid dataset
    # same shape / same position / same color / same size
    # vary only roughness x transparency
    # ----------------------------
    grid_n = int(cfg.get("grid_n", 5))
    g0, g1 = cfg.get("glossiness_range", [0.0, 1.0])
    t0, t1 = cfg.get("transparency_range", [0.0, 1.0])
    gloss_vals = linspace(g0, g1, grid_n)
    trans_vals = linspace(t0, t1, grid_n)

    grid_shape = cfg.get("grid_shape", "icosahedron")
    grid_pos_xy = cfg.get("grid_pos_xy", [0.0, 0.5])
    grid_yaw_deg = float(cfg.get("grid_yaw_deg", 0.0))
    grid_scale = float(cfg.get("grid_scale", 1.0))
    grid_target_max_dim = float(
        cfg.get("grid_target_max_dim", float(cfg.get("target_max_dim", 2.8)) * 1.25)
    )

    # Grid color in Lab
    grid_sample_L = bool(cfg.get("grid_sample_L", False))
    if grid_sample_L:
        grid_color_sample = sample_lab_color(cfg, rng)
        grid_lab = grid_color_sample["lab"]
        grid_rgb = grid_color_sample["rgb"]
    else:
        grid_L = float(cfg.get("grid_fixed_L", cfg.get("fixed_L", 65.0)))
        grid_a = float(cfg.get("grid_a", 0.0))
        grid_b = float(cfg.get("grid_b", 0.0))
        grid_rgb, in_gamut = lab_to_rgb(grid_L, grid_a, grid_b)
        if not in_gamut:
            grid_rgb = [min(1.0, max(0.0, c)) for c in grid_rgb]
        grid_lab = [grid_L, grid_a, grid_b]

    base_uid = random_n
    idx = 0
    for i_g, gloss in enumerate(gloss_vals):
        for j_t, trans in enumerate(trans_vals):
            uid = f"{(base_uid + idx):06d}"
            z = {
                "shape_name": grid_shape,
                "face_index": 0,
                "glossiness": float(gloss),
                "transparency": float(trans),
                "lab": [float(grid_lab[0]), float(grid_lab[1]), float(grid_lab[2])],
                "rgb": [float(grid_rgb[0]), float(grid_rgb[1]), float(grid_rgb[2])],
                "scale": float(grid_scale),
                "pos_xy": [float(grid_pos_xy[0]), float(grid_pos_xy[1])],
                "yaw_deg": float(grid_yaw_deg),
            }
            tag = f"grid_r{i_g:02d}_t{j_t:02d}"
            render_one(
                cfg,
                engine,
                img_dir,
                meta_path,
                uid,
                z,
                tag=tag,
                target_max_dim=grid_target_max_dim,
            )
            idx += 1

    print(f"Done. Random={random_n}, Grid={grid_n}x{grid_n}. Images in {img_dir}")
if __name__ == "__main__":
    main()