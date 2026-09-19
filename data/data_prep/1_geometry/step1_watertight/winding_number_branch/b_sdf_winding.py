"""Decide inside from outside by an integral over the surface rather than by connectivity.

Purpose
    This is the step the branch exists for. Flood labelling answers "can I walk here from outside",
    which is a statement about the volume's connectivity and is therefore only as good as the
    shell's seal: one hole and an entire cavity flips. The generalized winding number asks a
    different question — how many times does the surface wrap around this point — and answers it
    analytically, with no assumption that the surface is closed. On the inputs that break the flood
    (open boundaries, self-intersections, interpenetrating parts, inverted normals) it still lands
    the surface where the geometry says it should.

Input
    The band points from the previous step, and the same source meshes those points were derived
    from.

Output
    Per object: a signed distance at every band voxel, carried alongside the voxel indices so the
    values can be scattered back into a volume.

    Only the band is covered. The interior and exterior away from it are still unassigned; the
    extraction step fills them with constants, since outside the band only the sign matters.

Key idea
    `igl.signed_distance` in winding-number mode. The cost is an integral over every face for every
    query, which is exactly why the previous step narrowed the query set first — the two are a pair,
    and running this one on a full grid is not affordable.

    Normalization is recomputed here and must reproduce the previous step's bit for bit, otherwise
    the band points no longer sit on the mesh they were measured against and the field is quietly
    wrong rather than obviously broken. The all-zeros guard exists because an interrupted libigl
    call returns a zero array instead of raising, which would otherwise be saved as a valid result;
    and each mesh runs under a timeout in its own subprocess, because the integral's cost has no
    useful upper bound on a pathological mesh and one object must not stall a shard.
"""

import numpy as np
import trimesh
import os
import sys
import gc
from tqdm import tqdm
import argparse
import glob
import igl
from multiprocessing import Process, set_start_method

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def normalize_mesh_np(mesh, scale=0.8):
    """Normalize the mesh. Must reproduce a_floodfill.py exactly — the band points
    were measured against the mesh that step normalized."""
    vertices = mesh.vertices.copy()
    min_coords, max_coords = vertices.min(axis=0), vertices.max(axis=0)
    dxyz = max_coords - min_coords
    dist = max(dxyz)
    mesh_scale = 1.0 * scale / dist
    mesh_offset = -(min_coords + max_coords) / 2
    vertices = (vertices + mesh_offset) * mesh_scale
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


def compute_vol_sdf_worker(npz_path, mesh_dir, output_dir):
    """Standalone worker: load OBJ mesh, compute SDF, save."""
    try:
        # Load floodfill data
        data = np.load(npz_path)
        middle_points = data['middle_points']
        middle_voxelxyz = data['middle_voxelxyz']
        bbox_min = float(data['bbox_min'])
        bbox_max = float(data['bbox_max'])
        resolution = int(data['resolution'])

        # Get mesh stem: <stem>_floodfill.npz -> <stem>.obj
        npz_name = os.path.basename(npz_path)
        stem = npz_name.replace('_floodfill.npz', '')
        mesh_path = os.path.join(mesh_dir, f"{stem}.obj")

        if not os.path.isfile(mesh_path):
            print(f"OBJ not found: {mesh_path}")
            return False

        origin_mesh = trimesh.load(mesh_path, process=False)
        if origin_mesh is None or len(origin_mesh.faces) == 0:
            print(f"Empty mesh: {mesh_path}")
            return False

        normalized_mesh = normalize_mesh_np(origin_mesh, scale=0.8)

        # Compute signed distance
        sign_type = igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
        mesh_vertices = np.array(normalized_mesh.vertices).astype(np.float64)
        mesh_faces = np.array(normalized_mesh.faces).astype(np.int64)
        middle_points_np = middle_points.astype(np.float64)

        vol_sdf, _, _, _ = igl.signed_distance(
            middle_points_np, mesh_vertices, mesh_faces, sign_type=sign_type
        )

        # Validate: SDF must not be all zeros (happens when SIGTERM interrupts C++ code)
        if np.all(vol_sdf == 0):
            print(f"ERROR: vol_sdf all zeros for {stem} — likely signal-interrupted, not saving")
            return False

        # Save result
        save_path = os.path.join(output_dir, f"{stem}_sdf.npz")
        np.savez_compressed(
            save_path,
            middle_voxelxyz=middle_voxelxyz,
            vol_sdf=vol_sdf.astype(np.float32),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution=resolution
        )
        print(f"Saved SDF data to {save_path}")
        return True

    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def _subprocess_entry(npz_path, mesh_dir, output_dir):
    """Turn the worker's verdict into an exit code.

    A Process target's return value never reaches the parent — only its exit status does. That
    matters most for the all-zeros guard above, whose entire purpose is to refuse to write a file:
    without this, a refusal would be reported as a success and the missing output would only
    surface at the next stage.
    """
    sys.exit(0 if compute_vol_sdf_worker(npz_path, mesh_dir, output_dir) else 1)


def main():
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        required=True,
                        help="Directory containing floodfill npz files")
    parser.add_argument('--mesh_dir', type=str,
                        required=True,
                        help="Directory of source meshes — the same one a_floodfill.py read")
    parser.add_argument('--output_dir', type=str,
                        required=True)
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--timeout', type=int, default=600,
                        help="Timeout in seconds per mesh")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get npz files to process
    npz_files = sorted(glob.glob(os.path.join(args.input_dir, "*_floodfill.npz")))
    start_idx = args.task_id * args.batch_size
    end_idx = min(start_idx + args.batch_size, len(npz_files))
    npz_files = npz_files[start_idx:end_idx]

    print(f"Processing {len(npz_files)} npz files [{start_idx}, {end_idx})")

    timeout_count = 0
    success_count = 0
    skip_count = 0
    error_count = 0

    for npz_path in tqdm(npz_files, desc="Computing SDF"):
        npz_name = os.path.basename(npz_path)
        stem = npz_name.replace('_floodfill.npz', '')
        save_path = os.path.join(args.output_dir, f"{stem}_sdf.npz")

        if os.path.exists(save_path):
            print(f"Skipping {npz_name} - already processed")
            skip_count += 1
            continue

        # Run in subprocess with timeout
        process = Process(target=_subprocess_entry,
                          args=(npz_path, args.mesh_dir, args.output_dir))
        process.start()
        process.join(timeout=args.timeout)

        if process.is_alive():
            print(f"TIMEOUT: {npz_name} exceeded {args.timeout}s, skipping...")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            timeout_count += 1
        else:
            if process.exitcode == 0:
                success_count += 1
            else:
                error_count += 1

        process.close()
        gc.collect()

    print(f"\nCompleted: {success_count} success, {skip_count} skipped, "
          f"{timeout_count} timeout, {error_count} error")


# ---------------------------------------------------------------------------
# Invocation (stage b of four):
#
#   python b_sdf_winding.py --input_dir <floodfill dir> --mesh_dir <obj dir> \
#       --output_dir <sdf dir> --task_id $i --batch_size 50
#
# --mesh_dir must be the same directory stage a read, and the normalization must
# stay at its default: the band points were measured against that exact mesh.
# Defaults kept: timeout=600s per mesh. The batch is an order of magnitude
# smaller than stage a's because this is the expensive half.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
