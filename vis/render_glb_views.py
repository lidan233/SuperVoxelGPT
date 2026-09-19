"""Render a textured GLB from several viewpoints, through Cycles."""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector


def _parse(argv):
    ap = argparse.ArgumentParser(description="Render a textured GLB through Cycles.")
    ap.add_argument("mesh")
    ap.add_argument("out_dir")
    ap.add_argument("--num-views", type=int, default=4)
    ap.add_argument("--size", type=int, default=518,
                    help="square render size; DINOv2 conditioning wants 518")
    ap.add_argument("--elev", type=float, default=20.0, help="elevation in degrees")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--radius", type=float, default=2.6,
                    help="camera distance after the asset is normalised into the unit sphere")
    return ap.parse_args(argv[argv.index("--") + 1:] if "--" in argv else argv[1:])


def setup_engine(samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    prefs = bpy.context.preferences
    if "cycles" in prefs.addons:
        cp = prefs.addons["cycles"].preferences
        for t in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                cp.compute_device_type = t
                break
            except Exception:
                continue
        try:
            cp.get_devices()
            for d in cp.devices:
                d.use = (d.type != "CPU")
        except Exception:
            pass
        sc.cycles.device = "GPU"
    return sc


def setup_lighting():
    """The corpus rig: a key point light, a large top area light, a bottom fill."""
    for o in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(o, do_unlink=True)

    key = bpy.data.objects.new("Key", bpy.data.lights.new("Key", type="POINT"))
    key.data.energy = 1000
    key.location = (4, 1, 6)
    bpy.context.collection.objects.link(key)

    top = bpy.data.objects.new("Top", bpy.data.lights.new("Top", type="AREA"))
    top.data.energy = 10000
    top.location = (0, 0, 10)
    top.scale = (100, 100, 100)
    bpy.context.collection.objects.link(top)

    bottom = bpy.data.objects.new("Bottom", bpy.data.lights.new("Bottom", type="AREA"))
    bottom.data.energy = 1000
    bottom.location = (0, 0, -10)
    bpy.context.collection.objects.link(bottom)


def normalize_scene():
    """Centre the imported asset on the origin and scale it into the unit cube.

    The transform is applied to the *roots*, not to the meshes. A glTF import nests every mesh
    under empties carrying the node transforms, so moving the meshes themselves moves nothing in
    world space -- the parent transform is reapplied on top and the asset stays where it was.
    """
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("nothing imported: the file carried no mesh")
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for o in meshes:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector((min(lo[i], w[i]) for i in range(3)))
            hi = Vector((max(hi[i], w[i]) for i in range(3)))
    centre = (lo + hi) / 2.0
    extent = max((hi - lo)[i] for i in range(3))
    scale = 1.0 / max(extent, 1e-9)

    roots = [o for o in bpy.context.scene.objects
             if o.parent is None and o.type in ("MESH", "EMPTY")]
    for o in roots:
        o.location = (o.location - centre) * scale
        o.scale = tuple(s * scale for s in o.scale)
    bpy.context.view_layer.update()

    lo2 = Vector((1e9, 1e9, 1e9))
    hi2 = Vector((-1e9, -1e9, -1e9))
    for o in meshes:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo2 = Vector((min(lo2[i], w[i]) for i in range(3)))
            hi2 = Vector((max(hi2[i], w[i]) for i in range(3)))
    print("[glb] normalized bounds %s -> %s"
          % (tuple(round(v, 3) for v in lo2), tuple(round(v, 3) for v in hi2)), flush=True)
    return centre, scale


def main():
    a = _parse(sys.argv)
    os.makedirs(a.out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = setup_engine(a.samples)

    ext = os.path.splitext(a.mesh)[1].lower()
    if ext in (".glb", ".gltf"):
        # Materials and their baseColor textures come in with this; nothing below removes them.
        bpy.ops.import_scene.gltf(filepath=a.mesh)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=a.mesh)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=a.mesh)
    else:
        raise SystemExit(f"unsupported input: {ext}")

    n_mat = len([m for m in bpy.data.materials])
    n_img = len([i for i in bpy.data.images if i.size[0] > 0])
    print(f"[glb] imported: {n_mat} materials, {n_img} images", flush=True)

    normalize_scene()
    setup_lighting()

    # A white background rather than a transparent one: the encoder is handed RGB, and compositing
    # an alpha render onto black would darken every silhouette edge.
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = 1.0
    nt.links.new(bg.outputs["Background"],
                 nt.nodes.new("ShaderNodeOutputWorld").inputs["Surface"])
    sc.render.film_transparent = False

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_height = cam_data.sensor_width = 32
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    sc.camera = cam

    sc.render.resolution_x = a.size
    sc.render.resolution_y = a.size
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"

    elev = math.radians(a.elev)
    for i in range(a.num_views):
        az = 2.0 * math.pi * i / a.num_views
        pos = Vector((a.radius * math.cos(elev) * math.cos(az),
                      a.radius * math.cos(elev) * math.sin(az),
                      a.radius * math.sin(elev)))
        cam.location = pos
        cam.rotation_euler = (Vector((0, 0, 0)) - pos).to_track_quat("-Z", "Y").to_euler()
        out = os.path.join(a.out_dir, "view%02d.png" % i)
        sc.render.filepath = out
        bpy.ops.render.render(write_still=True)
        print(f"[glb] view {i} -> {out}", flush=True)

    print(f"[glb] {a.num_views} views at {a.size}px -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
