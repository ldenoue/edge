"""Generate paired RGB/mask images from a rigged MakeHuman body via MPFB.

This script intentionally keeps MPFB outside the repository. Install the
official MPFB extension in Blender 4.2 or newer, then run:

    blender --background --python scripts/generate_realistic_people.py -- \
      --output data/realistic_people_preview --backgrounds /path/to/photos

The generated person is a detailed skinned mesh.  Garment colors are assigned
procedurally for the RGB preview; the paired mask is rendered from the exact
same camera and pose.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/realistic_people_preview"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--validation-count",
        type=int,
        default=0,
        help="Reserve the final N frames and a held-out subset of backgrounds for validation.",
    )
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=2405)
    parser.add_argument(
        "--backgrounds",
        type=Path,
        help="Folder of JPG/PNG/WebP/TIFF photos. A random crop is used for each frame.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if args.validation_count < 0 or args.validation_count >= args.count:
        parser.error("--validation-count must be non-negative and smaller than --count")
    return args


def load_mpfb():
    root = next((name for name in sys.modules if name.endswith(".mpfb")), None)
    if root is None:
        raise RuntimeError(
            "MPFB is not enabled. Install it from Blender Preferences > Extensions, "
            "then rerun this script."
        )
    human_module = importlib.import_module(f"{root}.services.humanservice")
    target_module = importlib.import_module(f"{root}.services.targetservice")
    props_module = importlib.import_module(f"{root}.entities.objectproperties")
    return human_module.HumanService, target_module.TargetService, props_module.HumanObjectProperties


def make_material(name: str, color: tuple[float, float, float, float], roughness: float = 0.7):
    value = bpy.data.materials.new(name)
    value.diffuse_color = color
    value.use_nodes = True
    shader = value.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    return value


def make_mask_material():
    value = bpy.data.materials.new("person-mask")
    value.use_nodes = True
    nodes = value.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1, 1, 1, 1)
    output = nodes.new("ShaderNodeOutputMaterial")
    value.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return value


def look_at(obj, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def find_backgrounds(folder: Path | None) -> list[Path]:
    if folder is None:
        return []
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Background folder does not exist: {folder}")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    images = sorted(path for path in folder.rglob("*") if path.suffix.lower() in extensions)
    if not images:
        raise FileNotFoundError(f"No supported background photos below {folder}")
    return images


def make_photo_background(path: Path, camera, center: Vector, person_height: float, rng: random.Random):
    """Place a camera-facing, center-cropped emissive photo behind the person."""
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    if not width or not height:
        raise ValueError(f"Could not read background dimensions: {path}")

    material = bpy.data.materials.new(f"background-{path.stem}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    coordinates = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Linear"
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = rng.uniform(0.72, 1.08)
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(coordinates.outputs["UV"], mapping.inputs["Vector"])
    material.node_tree.links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    material.node_tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    # Preserve the source aspect ratio by sampling a square crop. Extra random
    # zoom and pan prevent the same photo from becoming the same background.
    if width >= height:
        crop_x, crop_y = height / width, 1.0
    else:
        crop_x, crop_y = 1.0, width / height
    zoom = rng.uniform(0.72, 1.0)
    crop_x *= zoom
    crop_y *= zoom
    mapping.inputs["Scale"].default_value = (crop_x, crop_y, 1.0)
    mapping.inputs["Location"].default_value = (
        rng.uniform(0.0, 1.0 - crop_x),
        rng.uniform(0.0, 1.0 - crop_y),
        0.0,
    )

    away_from_camera = (center - camera.location).normalized()
    plane_center = center + away_from_camera * max(3.0, person_height * 1.8)
    bpy.ops.mesh.primitive_plane_add(size=2, location=plane_center)
    plane = bpy.context.object
    plane.name = "photo-background"
    plane.rotation_mode = "QUATERNION"
    plane.rotation_quaternion = (camera.location - plane.location).to_track_quat("Z", "Y")
    distance = (plane.location - camera.location).length
    half_size = distance * math.tan(camera.data.angle * 0.5) * 1.08
    plane.scale = (half_size, half_size, 1)
    plane.data.materials.append(material)
    return plane


def random_skin(rng: random.Random) -> tuple[float, float, float, float]:
    palettes = [
        ((0.82, 0.58, 0.43), (1.08, 0.98, 0.92)),
        ((0.52, 0.29, 0.18), (1.10, 1.00, 0.91)),
        ((0.27, 0.13, 0.075), (1.13, 1.02, 0.88)),
        ((0.70, 0.43, 0.28), (1.10, 0.98, 0.89)),
    ]
    base, tint = rng.choice(palettes)
    jitter = rng.uniform(0.90, 1.08)
    return tuple(min(1.0, base[i] * tint[i] * jitter) for i in range(3)) + (1.0,)


def set_character_shape(human, rng, TargetService, HumanObjectProperties) -> dict:
    # Values stay away from the extremes where the generic rig can deform poorly.
    values = {
        "gender": rng.uniform(0.10, 0.90),
        "age": rng.uniform(0.28, 0.68),
        "weight": rng.uniform(0.25, 0.78),
        "muscle": rng.uniform(0.22, 0.75),
        "height": rng.uniform(0.28, 0.74),
    }
    ancestry = [rng.uniform(0.05, 1.0) for _ in range(3)]
    total = sum(ancestry)
    values.update(dict(zip(("african", "asian", "caucasian"), (v / total for v in ancestry))))
    for name, value in values.items():
        HumanObjectProperties.set_value(name, value, entity_reference=human)
    TargetService.reapply_macro_details(human)
    return {name: round(value, 4) for name, value in values.items()}


def color_body(human, rng: random.Random) -> None:
    skin = make_material(f"skin-{human.name}", random_skin(rng), 0.82)
    shirt = make_material(
        f"shirt-{human.name}",
        (rng.uniform(0.03, 0.9), rng.uniform(0.03, 0.9), rng.uniform(0.03, 0.9), 1),
    )
    pants = make_material(
        f"pants-{human.name}",
        (rng.uniform(0.015, 0.32), rng.uniform(0.015, 0.32), rng.uniform(0.02, 0.42), 1),
    )
    inherited_materials = list(human.data.materials)
    human.data.materials.clear()
    for material in inherited_materials:
        if material.users == 0:
            bpy.data.materials.remove(material)
    for value in (skin, shirt, pants):
        human.data.materials.append(value)

    zs = [vertex.co.z for vertex in human.data.vertices]
    low, high = min(zs), max(zs)
    span = high - low
    hip = low + span * 0.49
    shoulder = low + span * 0.79
    # Polygon centers let one detailed mesh read as clothed without changing its
    # geometry. This is only an RGB-domain cue; every face remains person mask.
    for polygon in human.data.polygons:
        center_z = sum(human.data.vertices[i].co.z for i in polygon.vertices) / len(polygon.vertices)
        if hip <= center_z <= shoulder:
            polygon.material_index = 1
        elif low + span * 0.13 <= center_z < hip:
            polygon.material_index = 2
        else:
            polygon.material_index = 0


def apply_pose(rig, rng: random.Random) -> dict:
    # Rotations are deliberately bounded. The MPFB game rig uses FK bones and
    # skinning weights, making these deformations smooth and anatomically useful.
    pose_ranges = {
        "pelvis": ((-12, 12), (-18, 18), (-14, 14)),
        "spine_01": ((-10, 14), (-12, 12), (-10, 10)),
        "spine_02": ((-10, 16), (-15, 15), (-12, 12)),
        "spine_03": ((-8, 14), (-14, 14), (-12, 12)),
        "neck_01": ((-15, 20), (-22, 22), (-15, 15)),
        "head": ((-15, 18), (-28, 28), (-18, 18)),
        "upperarm_l": ((-70, 80), (-55, 55), (-105, 75)),
        "upperarm_r": ((-70, 80), (-55, 55), (-75, 105)),
        "lowerarm_l": ((-8, 115), (-18, 18), (-12, 85)),
        "lowerarm_r": ((-8, 115), (-18, 18), (-85, 12)),
        "hand_l": ((-28, 28), (-25, 25), (-30, 30)),
        "hand_r": ((-28, 28), (-25, 25), (-30, 30)),
        "thigh_l": ((-62, 28), (-26, 38), (-30, 32)),
        "thigh_r": ((-62, 28), (-38, 26), (-32, 30)),
        "calf_l": ((0, 105), (-9, 9), (-8, 8)),
        "calf_r": ((0, 105), (-9, 9), (-8, 8)),
        "foot_l": ((-28, 30), (-12, 12), (-18, 18)),
        "foot_r": ((-28, 30), (-12, 12), (-18, 18)),
    }
    result = {}
    for name, ranges in pose_ranges.items():
        bone = rig.pose.bones.get(name)
        if bone is None:
            continue
        angles = [math.radians(rng.uniform(*limits)) for limits in ranges]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = angles
        result[name] = [round(math.degrees(value), 2) for value in angles]
    bpy.context.view_layer.update()
    return result


def evaluated_bounds(obj) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def setup_scene(size: int):
    scene = bpy.context.scene
    # Blender 5.x renamed the engine identifier back to BLENDER_EEVEE.
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 35
    scene.view_settings.view_transform = "Standard"
    scene.render.film_transparent = False

    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    world.use_nodes = True
    scene.world = world

    bpy.ops.object.camera_add(location=(0, -4, 0))
    camera = bpy.context.object
    scene.camera = camera
    camera.data.lens = 58

    bpy.ops.mesh.primitive_plane_add(size=30)
    floor = bpy.context.object

    bpy.ops.object.light_add(type="AREA", location=(-3.5, -4, 5))
    key = bpy.context.object
    key.data.shape = "DISK"
    key.data.size = 4

    bpy.ops.object.light_add(type="AREA", location=(3, 0, 3))
    fill = bpy.context.object
    fill.data.size = 3
    return scene, camera, floor, key, fill


def remove_photo_background(plane) -> None:
    """Remove a temporary photo plane and all of its per-frame data blocks."""
    mesh = plane.data
    materials = list(mesh.materials)
    images = []
    for material in materials:
        if material.use_nodes:
            images.extend(
                node.image
                for node in material.node_tree.nodes
                if node.type == "TEX_IMAGE" and node.image is not None
            )
    bpy.data.objects.remove(plane, do_unlink=True)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    for material in materials:
        if material.users == 0:
            bpy.data.materials.remove(material)
    for image in images:
        if image.users == 0:
            bpy.data.images.remove(image)


def write_training_manifests(output: Path, metadata: list[dict], validation_count: int) -> None:
    rows = []
    for record in metadata:
        frame = record["frame"]
        rows.append(
            {
                "id": f"synthetic-{frame:04d}",
                "image": str((output / "rgb" / f"{frame:04d}.png").resolve()),
                "human_ids": str((output / "masks" / f"{frame:04d}.png").resolve()),
                "source": "blender-mpfb-pixabay-rooms",
                "background": record["background"],
            }
        )
    split = len(rows) - validation_count if validation_count else len(rows)
    partitions = {
        "training.jsonl": rows[:split],
        "validation.jsonl": rows[split:],
    }
    for filename, partition in partitions.items():
        with (output / filename).open("w", encoding="utf-8") as handle:
            for row in partition:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rgb").mkdir(exist_ok=True)
    (args.output / "masks").mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    background_photos = find_backgrounds(args.backgrounds)
    training_backgrounds = background_photos
    validation_backgrounds: list[Path] = []
    if background_photos and args.validation_count:
        shuffled = background_photos.copy()
        random.Random(args.seed + 1).shuffle(shuffled)
        held_out_count = max(1, round(len(shuffled) * 0.15))
        validation_backgrounds = shuffled[:held_out_count]
        training_backgrounds = shuffled[held_out_count:]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    HumanService, TargetService, HumanObjectProperties = load_mpfb()
    scene, camera, floor, key, fill = setup_scene(args.size)
    mask_material = make_mask_material()
    metadata = []
    photo_plane = None

    for index in range(1, args.count + 1):
        human = HumanService.create_human()
        human.name = f"person-{index:04d}"
        human["person"] = True
        shape = set_character_shape(human, rng, TargetService, HumanObjectProperties)
        color_body(human, rng)
        rig = HumanService.add_builtin_rig(human, "game_engine")
        rig.name = f"person-rig-{index:04d}"
        pose = apply_pose(rig, rng)

        lower, upper = evaluated_bounds(human)
        floor.location.z = lower.z - 0.015
        center = (lower + upper) * 0.5
        width = upper.x - lower.x
        height = upper.z - lower.z
        camera.location = Vector(
            (
                center.x + rng.uniform(-0.06, 0.06) * height,
                center.y - max(2.8, max(width, height) * rng.uniform(2.05, 2.35)),
                center.z + rng.uniform(-0.035, 0.035) * height,
            )
        )
        camera.data.lens = rng.uniform(50, 62)
        look_at(camera, center)

        background_path = None
        if background_photos:
            is_validation = args.validation_count and index > args.count - args.validation_count
            background_pool = validation_backgrounds if is_validation else training_backgrounds
            background_path = rng.choice(background_pool)
            floor.hide_render = True
            photo_plane = make_photo_background(background_path, camera, center, height, rng)
            camera.data.dof.use_dof = True
            camera.data.dof.focus_distance = (center - camera.location).length
            camera.data.dof.aperture_fstop = rng.uniform(5.6, 11.0)
        else:
            floor.hide_render = False
            camera.data.dof.use_dof = False

        previous_floor_materials = list(floor.data.materials)
        floor.data.materials.clear()
        for material in previous_floor_materials:
            if material.users == 0:
                bpy.data.materials.remove(material)
        floor.data.materials.append(
            make_material(
                f"floor-{index}",
                (rng.uniform(0.10, 0.62), rng.uniform(0.10, 0.62), rng.uniform(0.10, 0.62), 1),
                0.9,
            )
        )
        background = scene.world.node_tree.nodes["Background"]
        background.inputs["Color"].default_value = (
            rng.uniform(0.025, 0.42), rng.uniform(0.025, 0.42), rng.uniform(0.025, 0.42), 1
        )
        background.inputs["Strength"].default_value = rng.uniform(0.25, 0.65)
        key.data.energy = rng.uniform(500, 1050)
        fill.data.energy = rng.uniform(180, 450)
        look_at(key, center)
        look_at(fill, center)

        scene.render.filepath = str((args.output / "rgb" / f"{index:04d}.png").resolve())
        bpy.ops.render.render(write_still=True)

        original_materials = list(human.data.materials)
        human.data.materials.clear()
        human.data.materials.append(mask_material)
        for polygon in human.data.polygons:
            polygon.material_index = 0
        floor.hide_render = True
        if photo_plane is not None:
            photo_plane.hide_render = True
        old_color = tuple(background.inputs["Color"].default_value)
        old_strength = background.inputs["Strength"].default_value
        background.inputs["Color"].default_value = (0, 0, 0, 1)
        background.inputs["Strength"].default_value = 0
        scene.render.filepath = str((args.output / "masks" / f"{index:04d}.png").resolve())
        bpy.ops.render.render(write_still=True)
        floor.hide_render = bool(background_photos)
        if photo_plane is not None:
            photo_plane.hide_render = False
        human.data.materials.clear()
        for value in original_materials:
            human.data.materials.append(value)
        background.inputs["Color"].default_value = old_color
        background.inputs["Strength"].default_value = old_strength

        metadata.append({
            "frame": index,
            "seed": args.seed,
            "split": "validation" if args.validation_count and index > args.count - args.validation_count else "training",
            "shape": shape,
            "pose_degrees_xyz": pose,
            "bounds": {"min": list(lower), "max": list(upper)},
            "background": str(background_path) if background_path else None,
        })
        print(f"rendered {index}/{args.count}", flush=True)

        if photo_plane is not None:
            remove_photo_background(photo_plane)
            photo_plane = None
        rig_data = rig.data
        human_mesh = human.data
        generated_materials = list(human_mesh.materials)
        bpy.data.objects.remove(rig, do_unlink=True)
        bpy.data.objects.remove(human, do_unlink=True)
        if rig_data.users == 0:
            bpy.data.armatures.remove(rig_data)
        if human_mesh.users == 0:
            bpy.data.meshes.remove(human_mesh)
        for material in generated_materials:
            if material.users == 0 and material != mask_material:
                bpy.data.materials.remove(material)

    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_training_manifests(args.output, metadata, args.validation_count)
    print(f"wrote {args.count} realistic RGB/mask pairs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
