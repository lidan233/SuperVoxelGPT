"""Narrow the volume down to the thin band where the sign is actually in question.

Purpose
    The winding number is what makes this branch accurate, and it is an integral over every face
    of the mesh for every query point. Asking it about a full high-resolution grid is not affordable
    on any realistic budget. Almost all of that grid is not in doubt, though: a cell deep inside the
    object and a cell far outside it are separated from the surface by enough empty space that plain
    connectivity settles them. This step spends the cheap method on the easy majority so that the
    expensive one only ever sees the cells near the surface.

Input
    A directory of triangle meshes, one object per file. No assumptions about topology, orientation
    or manifoldness — that is the point of the branch.

Output
    Per object: the flood mask over the volume (which cells connectivity assigned to the interior),
    together with the band points in both world coordinates and voxel indices.

    Signs in the mask are the flood's opinion, not the final answer. Wherever the shell has a hole
    the flood has already leaked and this mask is wrong there; the next step is what overrules it,
    which is why the band is carried forward rather than only the mask.

Key idea
    Coarse-to-fine, so the full resolution is only ever evaluated inside the band. Flood at 64³,
    upsample the mask, refresh the band from the unsigned distance at the next level, flood again.
    Two details keep the upsampling honest: cells whose distance says they cannot be inside are
    erased from the upsampled mask before it is re-flooded, which stops a coarse mask from bleeding
    through a thin wall it could not resolve; and the exterior is identified by the label the flood
    gave the corner cell, which is only correct because normalization leaves margin between the
    object and the box — the two settings are a pair and neither can be changed alone.

    Normalization to a fixed scale is a contract, not a convenience — the next step re-derives it
    from the same source mesh and the band points would otherwise no longer sit on the surface they
    came from. Each object runs in its own subprocess, because a single high-resolution pass
    fragments GPU memory badly enough to cost the object after it.
"""

import torch
import numpy as np
import trimesh
import os
import sys
import gc
from tqdm import tqdm
import argparse
import glob
from multiprocessing import Process, set_start_method

import cubvh
import math
import torch.nn.functional as F

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.set_per_process_memory_fraction(0.95)


# ── Floodfill algorithm ─────────────────────────────────────────────────────

def generate_dense_grid_points_chunked(bbox_min=-1.05, bbox_max=1.05, resolution=512):
    x = torch.linspace(bbox_min, bbox_max, resolution + 1, dtype=torch.float32)
    y = torch.linspace(bbox_min, bbox_max, resolution + 1, dtype=torch.float32)
    z = torch.linspace(bbox_min, bbox_max, resolution + 1, dtype=torch.float32)
    x = x.cuda()
    y = y.cuda()
    z = z.cuda()
    xs, ys, zs = torch.meshgrid(x, y, z, indexing='ij')
    xyz = torch.stack((xs, ys, zs), dim=-1)
    del xs, ys, zs, x, y, z
    torch.cuda.empty_cache()
    xyz = xyz.reshape(-1, 3)
    grid_size = [resolution + 1, resolution + 1, resolution + 1]
    return xyz, grid_size


def process_grid_in_chunks(bvh, grid_xyz, chunk_size=1000000):
    total_points = grid_xyz.shape[0]
    udf_results = []
    for i in tqdm(range(0, total_points, chunk_size), desc="Processing grid chunks"):
        end_idx = min(i + chunk_size, total_points)
        chunk = grid_xyz[i:end_idx]
        with torch.no_grad():
            udf_chunk, _, _ = bvh.unsigned_distance(chunk, return_uvw=False)
            udf_results.append(udf_chunk.cpu())
        torch.cuda.empty_cache()
    udf = torch.cat(udf_results, dim=0).cuda()
    del udf_results
    torch.cuda.empty_cache()
    return udf


def get_boundary_points(mask, bvh, box_min, box_max, range_size=3, mask_true=False):
    inside_middle_points = torch.nonzero(mask).to(torch.int16)
    points_neighbor = inside_middle_points + torch.tensor([[1, 0, 0]]).to(mask.device)
    label = mask[points_neighbor[:, 0], points_neighbor[:, 1], points_neighbor[:, 2]].to(torch.uint8)
    points_neighbor = inside_middle_points + torch.tensor([[0, 1, 0]]).to(mask.device)
    label = label + mask[points_neighbor[:, 0], points_neighbor[:, 1], points_neighbor[:, 2]]
    points_neighbor = inside_middle_points + torch.tensor([[0, 0, 1]]).to(mask.device)
    label = label + mask[points_neighbor[:, 0], points_neighbor[:, 1], points_neighbor[:, 2]]
    points_neighbor = inside_middle_points + torch.tensor([[-1, 0, 0]]).to(mask.device)
    label = label + mask[points_neighbor[:, 0], points_neighbor[:, 1], points_neighbor[:, 2]]
    points_neighbor = inside_middle_points + torch.tensor([[0, -1, 0]]).to(mask.device)
    label = label + mask[points_neighbor[:, 0], points_neighbor[:, 1], points_neighbor[:, 2]]
    points_neighbor = inside_middle_points + torch.tensor([[0, 0, -1]]).to(mask.device)
    label = label + mask[points_neighbor[:, 0], points_neighbor[:, 1], points_neighbor[:, 2]]
    boundary_points = inside_middle_points[label != 6]
    del points_neighbor, label

    if range_size <= 0:
        band = torch.zeros(mask.shape[0], mask.shape[1], mask.shape[2], dtype=torch.bool, device=mask.device)
        band[boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2]] = True
    else:
        band = torch.zeros(mask.shape[0], mask.shape[1], mask.shape[2], dtype=torch.bool, device=mask.device)
        for i in range(-range_size, range_size + 1):
            for j in range(-range_size, range_size + 1):
                for k_offset in range(-range_size, range_size + 1):
                    shifted_x = (boundary_points[:, 0] + i).clamp(0, mask.shape[0] - 1)
                    shifted_y = (boundary_points[:, 1] + j).clamp(0, mask.shape[1] - 1)
                    shifted_z = (boundary_points[:, 2] + k_offset).clamp(0, mask.shape[2] - 1)
                    band[shifted_x.int(), shifted_y.int(), shifted_z.int()] = True

    final_points_xyz = torch.nonzero(band).to(torch.int16)
    del band
    final_points = final_points_xyz / mask.shape[0] * (box_max - box_min) - (box_max - box_min) / 2
    signed_distance, _, _ = bvh.unsigned_distance(final_points, return_uvw=False)
    valid_mask = signed_distance > range_size / mask.shape[0]
    return final_points_xyz[valid_mask]


def multi_scale_floodfill(bvh, target_resolution=1024, tolerance=3,
                          bbox_min=-0.6, bbox_max=0.6):
    print(f"Generating SDF for resolution {target_resolution}")

    steps = 1 + math.ceil(math.log(target_resolution / 64, 2))
    resolutions = [64 * 2 ** i for i in range(steps - 1)] + [target_resolution]
    print(f"Using hierarchical resolutions: {resolutions}")

    current_dense_points = None
    current_dense_grid_points = None
    current_floodfill_mask = None

    for resolution_i, resolution in enumerate(resolutions[:-1]):
        print(f"Processing intermediate resolution: {resolution}")

        if current_dense_grid_points is None:
            grid_xyz, grid_size = generate_dense_grid_points_chunked(
                bbox_min=bbox_min, bbox_max=bbox_max, resolution=resolution
            )
            udf = process_grid_in_chunks(bvh, grid_xyz, chunk_size=500000)
            udf = udf.view(grid_size[0], grid_size[1], grid_size[2]).contiguous()
            eps = tolerance / resolution
            valid_mask = (udf.reshape(-1) < eps)
            get_coordinate = grid_xyz[valid_mask]
            get_coordinate_udf = udf[torch.where(udf < eps)]
            udf_3d = udf.reshape(resolution + 1, resolution + 1, resolution + 1)
            valid_3d = (udf_3d < eps)
            get_coordinate_voxelxyz = torch.stack(torch.where(valid_3d), dim=1)
            del grid_xyz, udf, udf_3d, valid_3d, valid_mask
            torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                udf, _, _ = bvh.unsigned_distance(current_dense_points, return_uvw=False)
                eps = tolerance / resolution
                valid_mask = (udf < eps)
                get_coordinate = current_dense_points[valid_mask]
                get_coordinate_udf = udf[valid_mask]
                get_coordinate_voxelxyz = current_dense_grid_points[valid_mask]
            del current_dense_points, current_dense_grid_points, udf, valid_mask
            torch.cuda.empty_cache()

        next_resolution = resolutions[resolution_i + 1]
        scale_factor = next_resolution // resolution

        offset_x, offset_y, offset_z = torch.meshgrid(
            torch.arange(scale_factor, device='cuda'),
            torch.arange(scale_factor, device='cuda'),
            torch.arange(scale_factor, device='cuda'),
            indexing='ij'
        )
        offsets = torch.stack([offset_x, offset_y, offset_z], dim=-1).view(-1, 3)
        del offset_x, offset_y, offset_z
        torch.cuda.empty_cache()

        if current_floodfill_mask is None:
            current_floodfill_mask = torch.zeros((resolution, resolution, resolution), dtype=torch.bool, device='cuda')
            current_floodfill_mask[get_coordinate_voxelxyz[:, 0], get_coordinate_voxelxyz[:, 1], get_coordinate_voxelxyz[:, 2]] = True
            current_floodfill_mask_out = cubvh.floodfill(current_floodfill_mask)
            empty_label = current_floodfill_mask_out[0, 0, 0].item()
            empty_mask = (current_floodfill_mask_out == empty_label)
            current_flag_mask = ~empty_mask
            del empty_mask, current_floodfill_mask_out
            current_floodfill_mask = current_flag_mask
            del current_flag_mask
            torch.cuda.empty_cache()

        chunk_size = min(500000, get_coordinate_voxelxyz.shape[0])
        high_res_coords_list = []
        high_res_grid_coords_list = []

        for i in range(0, get_coordinate_voxelxyz.shape[0], chunk_size):
            chunk = get_coordinate_voxelxyz[i:i + chunk_size]
            scaled_coords = chunk[:, None] * scale_factor
            chunk_grid_coords = (scaled_coords + offsets[None, :, :]).reshape(-1, 3)
            chunk_grid_coords = torch.clamp(chunk_grid_coords, 0, next_resolution - 1)
            high_res_grid_coords_list.append(chunk_grid_coords)
            del scaled_coords, chunk, chunk_grid_coords
            torch.cuda.empty_cache()

        high_res_grid_coordinates = torch.cat(high_res_grid_coords_list, dim=0)
        del high_res_grid_coords_list, offsets
        torch.cuda.empty_cache()

        mapping = torch.linspace(bbox_min, bbox_max, next_resolution + 1, dtype=torch.float32, device='cuda')
        high_res_coordinates = mapping[high_res_grid_coordinates]
        del mapping
        torch.cuda.empty_cache()

        current_dense_points = high_res_coordinates
        current_dense_grid_points = high_res_grid_coordinates
        print(next_resolution, next_resolution, current_dense_points.shape)

        current_floodfill_mask_up = F.interpolate(
            current_floodfill_mask[None, None, ...].to(torch.uint8),
            size=(next_resolution, next_resolution, next_resolution),
            mode='nearest'
        )[0, 0].bool()
        del current_floodfill_mask
        torch.cuda.empty_cache()

        high_keep_erase_points = get_boundary_points(current_floodfill_mask_up,
            bvh, bbox_min, bbox_max, mask_true=True, range_size=tolerance)
        current_floodfill_mask_up[high_keep_erase_points[:, 0].int(),
                                  high_keep_erase_points[:, 1].int(),
                                  high_keep_erase_points[:, 2].int()] = False
        del high_keep_erase_points
        torch.cuda.empty_cache()

        udf_chunk_size = 500000
        valid_indices_list = []
        for i in range(0, high_res_coordinates.shape[0], udf_chunk_size):
            chunk_coords = high_res_coordinates[i:i + udf_chunk_size]
            chunk_udf, _, _ = bvh.unsigned_distance(chunk_coords, return_uvw=False)
            eps = tolerance // 2 / next_resolution
            chunk_valid = torch.where(chunk_udf < eps)[0] + i
            valid_indices_list.append(chunk_valid)
            del chunk_coords, chunk_udf, chunk_valid
            torch.cuda.empty_cache()

        valid_indices = torch.cat(valid_indices_list, dim=0)
        del valid_indices_list
        torch.cuda.empty_cache()

        get_coordinate_voxelxyz = high_res_grid_coordinates[valid_indices]
        del high_res_coordinates, high_res_grid_coordinates, valid_indices
        torch.cuda.empty_cache()

        current_floodfill_mask_up[get_coordinate_voxelxyz[:, 0], get_coordinate_voxelxyz[:, 1], get_coordinate_voxelxyz[:, 2]] = True
        current_floodfill_mask_out = cubvh.floodfill(current_floodfill_mask_up)
        empty_label = current_floodfill_mask_out[0, 0, 0].item()
        empty_mask = (current_floodfill_mask_out == empty_label)
        current_floodfill_mask = ~empty_mask
        del current_floodfill_mask_up, current_floodfill_mask_out, empty_mask
        torch.cuda.empty_cache()

        del get_coordinate, get_coordinate_udf, get_coordinate_voxelxyz
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Final processing for resolution {target_resolution}")
    with torch.no_grad():
        udf_chunk_size = 500000
        final_valid_indices_list = []
        udf_tolerance = (bbox_max - bbox_min) / target_resolution * tolerance * 6

        for i in range(0, current_dense_points.shape[0], udf_chunk_size):
            chunk_coords = current_dense_points[i:i + udf_chunk_size]
            chunk_udf, _, _ = bvh.unsigned_distance(chunk_coords, return_uvw=False)
            chunk_valid = torch.where(chunk_udf < udf_tolerance)[0] + i
            final_valid_indices_list.append(chunk_valid)
            del chunk_coords, chunk_udf, chunk_valid
            torch.cuda.empty_cache()

        valid_indices = torch.cat(final_valid_indices_list, dim=0)
        del final_valid_indices_list
        torch.cuda.empty_cache()

        if valid_indices.shape[0] > 80000000:
            del valid_indices, current_dense_points, current_dense_grid_points
            torch.cuda.empty_cache()
            return None, None, None

        final_point_use = current_dense_points[valid_indices]
        final_point_voxelxyz = current_dense_grid_points[valid_indices]

    del current_dense_points, current_dense_grid_points, valid_indices
    torch.cuda.empty_cache()
    return current_floodfill_mask, final_point_use, final_point_voxelxyz


# ── Normalize + process ─────────────────────────────────────────────────────

def normalize_mesh_np(mesh, scale=0.8):
    vertices = mesh.vertices
    min_coords, max_coords = vertices.min(axis=0), vertices.max(axis=0)
    dxyz = max_coords - min_coords
    dist = max(dxyz)
    mesh_scale = 1.0 * scale / dist
    mesh_offset = -(min_coords + max_coords) / 2
    vertices = (vertices + mesh_offset) * mesh_scale
    mesh.vertices = vertices
    return mesh


def process_single_mesh(stem, normalized_mesh, output_dir, args):
    """Process single mesh and save npz (runs in subprocess)."""
    try:
        print(f"Building BVH for {stem}...")
        bvh = cubvh.cuBVH(
            torch.as_tensor(normalized_mesh.vertices, dtype=torch.float32, device='cuda'),
            torch.as_tensor(normalized_mesh.faces, dtype=torch.float32, device='cuda')
        )

        print(f"Running multi_scale_floodfill...")
        flood_mask, middle_points, middle_voxelxyz = multi_scale_floodfill(
            bvh,
            target_resolution=args.resolution,
            tolerance=args.tolerance,
            bbox_min=args.bbox_min,
            bbox_max=args.bbox_max
        )

        if flood_mask is None:
            print(f"Skipping {stem} - too many points")
            return False

        save_path = os.path.join(output_dir, f"{stem}_floodfill.npz")
        np.savez_compressed(
            save_path,
            flood_mask=flood_mask.cpu().numpy(),
            middle_points=middle_points.cpu().numpy(),
            middle_voxelxyz=middle_voxelxyz.cpu().numpy(),
            bbox_min=args.bbox_min,
            bbox_max=args.bbox_max,
            resolution=args.resolution
        )
        print(f"Saved floodfill data to {save_path}")

        del flood_mask, middle_points, middle_voxelxyz, bvh
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"Error processing {stem}: {e}")
        return False


def _subprocess_entry(stem, normalized_mesh, output_dir, args):
    """Turn the worker's verdict into an exit code.

    A Process target's return value never reaches the parent — only its exit status does — so a
    worker that returns False on a refusal (missing input, guard tripped, caught exception) would
    otherwise be indistinguishable from one that succeeded, and the tally below would count it.
    """
    sys.exit(0 if process_single_mesh(stem, normalized_mesh, output_dir, args) else 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', type=str,
                        required=True)
    parser.add_argument('--output_dir', type=str,
                        required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--tolerance', type=float, default=3)
    parser.add_argument('--bbox_min', type=float, default=-0.5)
    parser.add_argument('--bbox_max', type=float, default=0.5)
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--max_faces', type=int, default=0,
                        help="Skip meshes with more faces than this (0=no limit)")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    os.makedirs(args.output_dir, exist_ok=True)

    # Build file list: scan mesh_dir for *.obj
    all_files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))

    # Shard
    start_idx = args.task_id * args.batch_size
    end_idx = min(start_idx + args.batch_size, len(all_files))
    shard_files = all_files[start_idx:end_idx]

    print(f"Total OBJ files: {len(all_files)}")
    print(f"Shard [{start_idx}, {end_idx}): {len(shard_files)} files")

    success = 0
    skipped = 0
    errors = 0

    for filepath in tqdm(shard_files, desc="Processing"):
        filename = os.path.basename(filepath)
        stem = os.path.splitext(filename)[0]
        save_path = os.path.join(args.output_dir, f"{stem}_floodfill.npz")

        if os.path.exists(save_path):
            print(f"Skipping {filename} - already processed")
            skipped += 1
            continue

        try:
            origin_mesh = trimesh.load(filepath, process=False)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            errors += 1
            continue

        if origin_mesh is None or len(origin_mesh.faces) == 0:
            print(f"Empty mesh: {filename}")
            errors += 1
            continue

        if args.max_faces > 0 and origin_mesh.faces.shape[0] >= args.max_faces:
            print(f"Skipping {filename} - too many faces ({origin_mesh.faces.shape[0]})")
            skipped += 1
            continue

        normalized_mesh = normalize_mesh_np(origin_mesh, scale=0.8)

        process = Process(target=_subprocess_entry,
                          args=(stem, normalized_mesh, args.output_dir, args))
        process.start()
        process.join()

        if process.exitcode == 0:
            success += 1
        else:
            errors += 1

        process.close()
        gc.collect()

    print(f"\nDone: {success} success, {skipped} skipped, {errors} errors")


# ---------------------------------------------------------------------------
# Invocation (stage a of four; the released dataset used the fast path one level
# up, so unlike the scripts there these commands are the intended recipe rather
# than a record of what ran):
#
#   python a_floodfill.py --mesh_dir <obj dir> --output_dir <floodfill dir> \
#       --resolution 1024 --task_id $i --batch_size 200
#
# Shards are contiguous slices, task_id * batch_size, so a shard count is chosen
# by picking batch_size — there is no separate shard-count flag. Defaults kept:
# tolerance=3 (band half-width in cells at every level), bbox +-0.5 against a
# mesh normalized to 0.8, max_faces=0 (no cap).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
