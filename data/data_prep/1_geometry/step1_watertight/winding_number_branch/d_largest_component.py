"""Drop the debris that band-limited extraction leaves behind.

Purpose
    Extraction is driven by a field that is only accurate near the surface, so wherever the band was
    noisy it produces small shells that are not part of the object. Everything downstream — the
    sharpening step, the voxelization, the saliency field — assumes it is looking at one surface, so
    the fragments have to go before the mesh leaves this stage.

Input
    The extracted meshes.

Output
    The same meshes carrying only their largest face-connected component.

    This is a filter, not a repair. It removes pieces that were never attached; it cannot seal a
    hole, and a mesh that arrives here broken leaves here broken and smaller.

Key idea
    Components are found through face adjacency rather than vertex proximity. The distinction
    matters: merging by vertex position welds two shells that merely touch at a point into one
    component, and the debris this step is meant to remove frequently does touch the object.
    Adjacency asks whether two faces share an edge, which is the question that actually corresponds
    to "same surface".

    Largest is by face count, which is the right measure here rather than volume or area — a
    degenerate shell can enclose a large volume while consisting of very few faces. Each mesh runs
    under a timeout in a subprocess, since the adjacency graph on a pathological mesh can be far
    larger than its face count suggests.
"""
import numpy as np
import trimesh
import os
import sys
import glob
from tqdm import tqdm
import argparse
import gc
from multiprocessing import Process, set_start_method


def extract_largest_component(mesh_path, output_dir):
    """Load mesh and extract the largest connected component."""
    try:
        mesh = trimesh.load(mesh_path, process=False)

        # Compute connected components based on face adjacency
        from scipy import sparse
        adjacency = mesh.face_adjacency
        adjacency_sparse = sparse.coo_matrix(
            (np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])),
            shape=(len(mesh.faces), len(mesh.faces))
        )
        n_components, labels = sparse.csgraph.connected_components(
            adjacency_sparse, directed=False
        )

        # Group faces by component
        components = [np.where(labels == i)[0] for i in range(n_components)]

        print(f"{os.path.basename(mesh_path)}: found {n_components} components")

        # Find the largest component
        keep_index = np.argmax([len(c) for c in components])
        keep_faces = components[keep_index]

        # Extract submesh with only the largest component
        largest_mesh = mesh.submesh([keep_faces], append=True)

        # Save
        mesh_name = os.path.basename(mesh_path)
        save_path = os.path.join(output_dir, mesh_name)
        largest_mesh.export(save_path)
        print(f"Saved to {save_path}")

        del mesh, largest_mesh
        gc.collect()
        return True

    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
        return False


def _subprocess_entry(mesh_path, output_dir):
    """Turn the worker's verdict into an exit code.

    A Process target's return value never reaches the parent — only its exit status does — so
    without this a mesh that failed to load would still be counted as a success below.
    """
    sys.exit(0 if extract_largest_component(mesh_path, output_dir) else 1)


def main():
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Extract largest connected component from watertight meshes"
    )
    parser.add_argument('--input_dir', type=str,
                        required=True,
                        help="Directory containing watertight PLY files")
    parser.add_argument('--output_dir', type=str,
                        required=True,
                        help="Output directory for cleaned meshes")
    parser.add_argument('--task_id', type=int, default=0,
                        help="SLURM array task ID for batch splitting")
    parser.add_argument('--batch_size', type=int, default=200,
                        help="Number of files per task")
    parser.add_argument('--timeout', type=int, default=120,
                        help="Timeout in seconds per mesh (default: 120)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get mesh files to process
    mesh_files = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")))
    start_idx = args.task_id * args.batch_size
    end_idx = min(start_idx + args.batch_size, len(mesh_files))
    mesh_files = mesh_files[start_idx:end_idx]

    print(f"Processing {len(mesh_files)} mesh files [{start_idx}, {end_idx})")

    timeout_count = 0
    success_count = 0
    skip_count = 0
    error_count = 0

    for mesh_path in tqdm(mesh_files, desc="Extracting largest component"):
        mesh_name = os.path.basename(mesh_path)
        save_path = os.path.join(args.output_dir, mesh_name)

        if os.path.exists(save_path):
            print(f"Skipping {mesh_name} - already processed")
            skip_count += 1
            continue

        # Run in subprocess with timeout
        process = Process(target=_subprocess_entry, args=(mesh_path, args.output_dir))
        process.start()
        process.join(timeout=args.timeout)

        if process.is_alive():
            print(f"TIMEOUT: {mesh_name} exceeded {args.timeout}s, skipping...")
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
# Invocation (stage d of four, the last):
#
#   python d_largest_component.py --input_dir <mesh dir> --output_dir <watertight dir> \
#       --task_id $i --batch_size 200 --timeout 120
#
# Output filenames are unchanged from the input, so the two directories must
# differ. The delivered mesh from this branch is what step2_sharpen consumes,
# in place of the fast path's output.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
