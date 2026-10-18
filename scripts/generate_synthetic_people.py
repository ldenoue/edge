"""Generate paired RGB/person-mask images with a procedural articulated human.

Run with Blender, not the system Python:

    blender --background --python scripts/generate_synthetic_people.py -- \
      --output data/synthetic_people_preview --count 16 --size 320
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_people_preview"))
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=1021)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def material(name: str, color: tuple[float, float, float, float], roughness: float = 0.65):
    value = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    value.diffuse_color = color
    value.use_nodes = True
    shader = value.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    return value


def mask_material():
    value = bpy.data.materials.new("person mask")
    value.use_nodes = True
    nodes = value.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1, 1, 1, 1)
    emission.inputs["Strength"].default_value = 1
    output = nodes.new("ShaderNodeOutputMaterial")
    value.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return value


def assign_person(obj, mat) -> None:
    obj.data.materials.append(mat)
    obj.pass_index = 1
    obj["person"] = True


def sphere(name: str, location: Vector, scale: tuple[float, float, float], mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_person(obj, mat)
    return obj


def segment(name: str, start: Vector, end: Vector, radius: float, mat):
    direction = end - start
    midpoint = (start + end) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=direction.length, location=midpoint)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    bevel = obj.modifiers.new("soft silhouette", "BEVEL")
    bevel.width = radius * 0.35
    bevel.segments = 2
    assign_person(obj, mat)
    return obj


def direction(angle: float, length: float, depth: float = 0.0) -> Vector:
    """Vector in the frontal X/Z plane, with angle zero pointing right."""
    return Vector((math.cos(angle) * length, depth, math.sin(angle) * length))


def look_at(obj, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def clear_person() -> None:
    for obj in list(bpy.data.objects):
        if obj.get("person"):
            bpy.data.objects.remove(obj, do_unlink=True)


def create_pose(rng: random.Random, skin, shirt, pants, shoes) -> dict:
    pelvis = Vector((0, rng.uniform(-0.12, 0.12), 2.45))
    torso_lean = math.radians(rng.uniform(-22, 22))
    torso_depth = rng.uniform(-0.14, 0.14)
    shoulders = pelvis + Vector((math.sin(torso_lean) * 1.35, torso_depth, math.cos(torso_lean) * 1.35))
    across = Vector((math.cos(torso_lean), 0, -math.sin(torso_lean)))
    left_shoulder = shoulders - across * 0.61
    right_shoulder = shoulders + across * 0.61
    left_hip = pelvis - across * 0.34
    right_hip = pelvis + across * 0.34

    joints: dict[str, Vector] = {
        "pelvis": pelvis,
        "shoulders": shoulders,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_hip": left_hip,
        "right_hip": right_hip,
    }

    # Arms can point down, sideways, overhead, forward, or cross the torso.
    for side_name, side, shoulder in (("left", -1, left_shoulder), ("right", 1, right_shoulder)):
        base = math.pi if side < 0 else 0.0
        upper_angle = base + math.radians(rng.uniform(-145, 145))
        elbow = shoulder + direction(upper_angle, 0.92, rng.uniform(-0.34, 0.34))
        bend = math.radians(rng.uniform(-125, 125))
        wrist = elbow + direction(upper_angle + bend, 0.84, rng.uniform(-0.30, 0.30))
        joints[f"{side_name}_elbow"] = elbow
        joints[f"{side_name}_wrist"] = wrist

    # Legs favor downward poses but include kicks, lunges, crossed legs, and crouches.
    for side_name, side, hip in (("left", -1, left_hip), ("right", 1, right_hip)):
        spread = math.radians(rng.uniform(-18, 62)) * side
        upper_angle = -math.pi / 2 + spread
        knee = hip + direction(upper_angle, 1.22, rng.uniform(-0.28, 0.28))
        bend = math.radians(rng.uniform(-75, 48)) * side
        ankle = knee + direction(upper_angle + bend, 1.18, rng.uniform(-0.22, 0.22))
        joints[f"{side_name}_knee"] = knee
        joints[f"{side_name}_ankle"] = ankle

    # Put the lowest foot on the floor while preserving the articulated pose.
    lowest = min(joints["left_ankle"].z, joints["right_ankle"].z)
    lift = 0.20 - lowest
    for point in joints.values():
        point.z += lift

    pelvis = joints["pelvis"]
    shoulders = joints["shoulders"]
    torso_mid = (pelvis + shoulders) * 0.5
    torso = sphere("torso", torso_mid, (0.61, 0.34, 0.85), shirt)
    torso.rotation_euler[1] = torso_lean
    sphere("hips", pelvis, (0.48, 0.32, 0.36), pants)

    neck = shoulders + Vector((math.sin(torso_lean) * 0.22, 0, math.cos(torso_lean) * 0.22))
    head = neck + Vector((math.sin(torso_lean) * 0.38, 0, math.cos(torso_lean) * 0.38))
    segment("neck", shoulders, neck, 0.16, skin)
    sphere("head", head, (0.34, 0.31, 0.43), skin)

    for side_name in ("left", "right"):
        shoulder = joints[f"{side_name}_shoulder"]
        elbow = joints[f"{side_name}_elbow"]
        wrist = joints[f"{side_name}_wrist"]
        segment(f"{side_name}_upper_arm", shoulder, elbow, 0.17, shirt)
        segment(f"{side_name}_forearm", elbow, wrist, 0.135, skin)
        sphere(f"{side_name}_elbow_joint", elbow, (0.18, 0.18, 0.18), skin)
        sphere(f"{side_name}_hand", wrist, (0.16, 0.12, 0.22), skin)

        hip = joints[f"{side_name}_hip"]
        knee = joints[f"{side_name}_knee"]
        ankle = joints[f"{side_name}_ankle"]
        segment(f"{side_name}_thigh", hip, knee, 0.23, pants)
        segment(f"{side_name}_shin", knee, ankle, 0.17, skin)
        sphere(f"{side_name}_knee_joint", knee, (0.20, 0.20, 0.20), skin)
        foot_end = ankle + Vector((0, -0.28, 0.02))
        segment(f"{side_name}_foot", ankle, foot_end, 0.19, shoes)

    all_points = list(joints.values()) + [head]
    return {
        "bounds": {
            "min_x": min(point.x for point in all_points) - 0.35,
            "max_x": max(point.x for point in all_points) + 0.35,
            "min_z": min(point.z for point in all_points) - 0.25,
            "max_z": max(point.z for point in all_points) + 0.45,
        },
        "joints": {name: [round(v, 4) for v in point] for name, point in joints.items()},
    }


def setup_scene(args: argparse.Namespace):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version >= (4, 2, 0) else "BLENDER_EEVEE"
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 35
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True

    bpy.ops.object.camera_add(location=(0, -10, 3.4))
    camera = bpy.context.object
    camera.data.lens = 52
    scene.camera = camera

    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "background_floor"
    floor.pass_index = 0

    bpy.ops.object.light_add(type="AREA", location=(-4, -4, 7))
    key = bpy.context.object
    key.data.energy = 850
    key.data.shape = "DISK"
    key.data.size = 5

    bpy.ops.object.light_add(type="AREA", location=(4, 1, 5))
    fill = bpy.context.object
    fill.data.energy = 500
    fill.data.size = 4
    look_at(fill, Vector((0, 0, 2.5)))

    return scene, camera, floor, key, fill, mask_material()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rgb").mkdir(exist_ok=True)
    (args.output / "masks").mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    scene, camera, floor, key, fill, person_mask_material = setup_scene(args)
    metadata = []
    for index in range(1, args.count + 1):
        clear_person()
        skin = material(
            f"skin-{index}",
            (rng.uniform(0.30, 0.92), rng.uniform(0.16, 0.70), rng.uniform(0.10, 0.55), 1),
        )
        shirt = material(
            f"shirt-{index}",
            (rng.uniform(0.04, 0.95), rng.uniform(0.04, 0.95), rng.uniform(0.04, 0.95), 1),
        )
        pants = material(
            f"pants-{index}",
            (rng.uniform(0.02, 0.45), rng.uniform(0.02, 0.45), rng.uniform(0.02, 0.55), 1),
        )
        shoes = material(f"shoes-{index}", (rng.uniform(0.02, 0.20),) * 3 + (1,))
        pose = create_pose(rng, skin, shirt, pants, shoes)

        floor.data.materials.clear()
        floor.data.materials.append(
            material(
                f"floor-{index}",
                (rng.uniform(0.08, 0.65), rng.uniform(0.08, 0.65), rng.uniform(0.08, 0.65), 1),
                0.85,
            )
        )
        scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (
            rng.uniform(0.02, 0.50), rng.uniform(0.02, 0.50), rng.uniform(0.02, 0.50), 1
        )
        scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = rng.uniform(0.25, 0.8)
        key.data.energy = rng.uniform(550, 1200)
        key.location = (rng.uniform(-5, 1), rng.uniform(-5, -1), rng.uniform(5, 9))
        look_at(key, Vector((0, 0, 2.5)))
        fill.data.energy = rng.uniform(180, 650)

        bounds = pose["bounds"]
        center_x = (bounds["min_x"] + bounds["max_x"]) * 0.5
        center_z = (bounds["min_z"] + bounds["max_z"]) * 0.5
        span = max(bounds["max_x"] - bounds["min_x"], bounds["max_z"] - bounds["min_z"])
        camera.location = (
            center_x + rng.uniform(-0.25, 0.25),
            -max(8.2, span * rng.uniform(2.25, 2.75)),
            center_z + rng.uniform(-0.20, 0.20),
        )
        camera.data.lens = rng.uniform(48, 62)
        look_at(camera, Vector((center_x, 0, center_z)))

        scene.frame_set(index)
        scene.render.filepath = str((args.output / "rgb" / f"{index:04d}.png").resolve())
        bpy.ops.render.render(write_still=True)

        # A second unlit render produces the exact aligned person mask. The
        # material override affects every rendered object, so hide the floor.
        background = scene.world.node_tree.nodes["Background"]
        previous_color = tuple(background.inputs["Color"].default_value)
        previous_strength = background.inputs["Strength"].default_value
        floor.hide_render = True
        original_materials = {}
        for obj in bpy.data.objects:
            if obj.get("person"):
                original_materials[obj.name] = list(obj.data.materials)
                obj.data.materials.clear()
                obj.data.materials.append(person_mask_material)
        background.inputs["Color"].default_value = (0, 0, 0, 1)
        background.inputs["Strength"].default_value = 0
        scene.render.filepath = str((args.output / "masks" / f"{index:04d}.png").resolve())
        bpy.ops.render.render(write_still=True)
        floor.hide_render = False
        for name, materials in original_materials.items():
            obj = bpy.data.objects[name]
            obj.data.materials.clear()
            for original in materials:
                obj.data.materials.append(original)
        background.inputs["Color"].default_value = previous_color
        background.inputs["Strength"].default_value = previous_strength
        metadata.append({"frame": index, "seed": args.seed, **pose})
        print(f"rendered {index}/{args.count}", flush=True)

    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.count} RGB/mask pairs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
