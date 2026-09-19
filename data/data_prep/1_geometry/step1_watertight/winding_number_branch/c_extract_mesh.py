"""Reassemble the volume from its two halves and pull a surface out of it.

Purpose
    The two previous steps deliberately produced a partial answer: a coarse inside/outside labelling
    everywhere, and an accurate signed distance only near the surface. Neither is a field an
    isosurface extractor can consume on its own. This step puts them back together and extracts.

Input
    The band signed distances, and the flood mask covering the same volume.

Output
    A watertight triangle mesh, at the resolution recorded in the inputs.

    Watertight, but not yet clean — band-limited extraction leaves detached shells wherever the band
    was noisy. Removing them is the next step's job, and it is a separate step because dropping
    components is a decision about the object, not about the field.

Key idea
    Away from the surface the exact distance is irrelevant; only its sign is. So the volume is
    filled with a constant of each sign according to the flood mask and the band is then overwritten
    with the values that were actually computed. That is what makes the expensive query affordable:
    the accurate field is only ever paid for where it changes the outcome.

    Extraction runs at a small positive isovalue rather than at zero. That inflates the surface by a
    fraction of a voxel, which is what keeps a wall thinner than the grid from opening into a hole —
    the same tolerance-for-thin-features trade the fast path makes with its occupancy epsilon, and
    it thickens the result by the same fraction. Large bands are scattered in chunks because the
    index tensors, not the volume, are what exhaust memory here.
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

from diso import DiffDMC

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.set_per_process_memory_fraction(0.95)


def extract_mesh_from_sdf(npz_path, floodfill_dir, output_dir, isovalue_cells):
    """Load SDF npz + floodfill mask, extract watertight mesh via DiffDMC."""
    try:
        # Load SDF data
        data = np.load(npz_path)
        middle_voxelxyz = torch.from_numpy(data['middle_voxelxyz']).cuda()
        vol_sdf = torch.from_numpy(data['vol_sdf']).cuda()
        bbox_min = float(data['bbox_min'])
        bbox_max = float(data['bbox_max'])
        resolution = int(data['resolution'])

        # Load flood_mask from floodfill npz
        npz_name = os.path.basename(npz_path)
        floodfill_path = os.path.join(
            floodfill_dir, npz_name.replace('_sdf.npz', '_floodfill.npz')
        )
        floodfill_data = np.load(floodfill_path)
        flood_mask = torch.from_numpy(floodfill_data['flood_mask']).cuda()

        # Build full SDF volume: inside=-1, outside=+1, band=computed SDF
        final_sdf = torch.full(
            (resolution, resolution, resolution),
            -1.0,
            dtype=torch.float32,
            device='cuda'
        )

        # Set outside voxels to positive
        final_sdf[~flood_mask] = 1.0

        # Fill in computed SDF values for band voxels
        if middle_voxelxyz.shape[0] > 1000000:
            chunk_size = 500000
            for i in range(0, middle_voxelxyz.shape[0], chunk_size):
                end_idx = min(i + chunk_size, middle_voxelxyz.shape[0])
                chunk_voxel = middle_voxelxyz[i:end_idx]
                chunk_sdf = vol_sdf[i:end_idx]
                final_sdf[chunk_voxel[:, 0].long(),
                          chunk_voxel[:, 1].long(),
                          chunk_voxel[:, 2].long()] = chunk_sdf
                del chunk_voxel, chunk_sdf
                torch.cuda.empty_cache()
        else:
            final_sdf[middle_voxelxyz[:, 0].long(),
                      middle_voxelxyz[:, 1].long(),
                      middle_voxelxyz[:, 2].long()] = vol_sdf

        del flood_mask, middle_voxelxyz, vol_sdf
        torch.cuda.empty_cache()

        # Extract mesh using DiffDMC (differentiable marching cubes). The isovalue is expressed in
        # cells and converted here, so it stays meaningful when the resolution changes: the field is
        # in world units on a mesh normalized to the unit box, so one cell is 1/resolution.
        diffdmc = DiffDMC(dtype=torch.float32).cuda()
        vertices, faces = diffdmc(final_sdf, isovalue=isovalue_cells / resolution, normalize=False)
        del final_sdf, diffdmc
        torch.cuda.empty_cache()

        # Scale vertices to world coordinates
        vertices = vertices / resolution * (bbox_max - bbox_min) - (bbox_max - bbox_min) / 2

        # Move to CPU and create mesh
        vertices_cpu = vertices.cpu().numpy()
        faces_cpu = faces.cpu().numpy()
        del vertices, faces
        torch.cuda.empty_cache()

        final_mesh = trimesh.Trimesh(vertices_cpu, faces_cpu, process=False)
        del vertices_cpu, faces_cpu

        # Save as PLY
        mesh_name = npz_name.replace('_sdf.npz', f'_{resolution}_watertight.ply')
        save_path = os.path.join(output_dir, mesh_name)
        final_mesh.export(save_path)
        print(f"Saved mesh to {save_path}")

        del final_mesh
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def _subprocess_entry(npz_path, floodfill_dir, output_dir, isovalue_cells):
    """Turn the worker's verdict into an exit code.

    A Process target's return value never reaches the parent — only its exit status does — so
    without this a caught exception would still be counted as a success below.
    """
    sys.exit(0 if extract_mesh_from_sdf(npz_path, floodfill_dir, output_dir, isovalue_cells) else 1)


def main():
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Extract watertight mesh from SDF volume using DiffDMC"
    )
    parser.add_argument('--input_dir', type=str,
                        required=True,
                        help="Directory of band signed distances written by b_sdf_winding.py")
    parser.add_argument('--floodfill_dir', type=str,
                        required=True,
                        help="Directory of flood masks written by a_floodfill.py")
    parser.add_argument('--output_dir', type=str,
                        required=True)
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--isovalue_cells', type=float, default=2.0,
                        help="Extract at this many cells outside the zero level set. Lowering it "
                             "towards zero reopens walls thinner than the grid; raising it thickens "
                             "everything by the same amount")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    os.makedirs(args.output_dir, exist_ok=True)

    # Get SDF npz files to process
    npz_files = sorted(glob.glob(os.path.join(args.input_dir, "*_sdf.npz")))
    start_idx = args.task_id * args.batch_size
    end_idx = min(start_idx + args.batch_size, len(npz_files))
    npz_files = npz_files[start_idx:end_idx]

    print(f"Processing {len(npz_files)} npz files [{start_idx}, {end_idx})")

    success_count = 0
    skip_count = 0
    error_count = 0

    for npz_path in tqdm(npz_files, desc="Extracting meshes"):
        npz_name = os.path.basename(npz_path)
        # Load resolution from npz to construct output name
        data = np.load(npz_path)
        resolution = int(data['resolution'])
        data.close()

        mesh_name = npz_name.replace('_sdf.npz', f'_{resolution}_watertight.ply')
        save_path = os.path.join(args.output_dir, mesh_name)

        if os.path.exists(save_path):
            print(f"Skipping {npz_name} - already processed")
            skip_count += 1
            continue

        # Run in subprocess to isolate GPU memory
        process = Process(target=_subprocess_entry,
                          args=(npz_path, args.floodfill_dir, args.output_dir,
                                args.isovalue_cells))
        process.start()
        process.join()

        if process.exitcode == 0:
            success_count += 1
        else:
            error_count += 1

        process.close()
        gc.collect()

    print(f"\nCompleted: {success_count} success, {skip_count} skipped, {error_count} errors")


# ---------------------------------------------------------------------------
# Invocation (stage c of four):
#
#   python c_extract_mesh.py --input_dir <sdf dir> --floodfill_dir <floodfill dir> \
#       --output_dir <mesh dir> --task_id $i --batch_size 100
#
# Both directories are needed because the sign field and the band values were
# written by different stages. Resolution is read from the inputs, not passed —
# it is whatever stage a was run at. --isovalue_cells defaults to 2, which is the
# same thin-wall trade the fast path makes with its occupancy epsilon.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
