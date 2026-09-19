"""Compute the saliency field that decides where the sampling budget goes.

Purpose
    Adaptive tokenization needs a defensible answer to "which parts of this surface matter". Without
    such a field, "adaptive" has no definition. This produces a per-vertex measure of how much a
    region stands out from its surroundings across scales.

Input
    A watertight mesh.

Output
    A per-vertex saliency energy, stored per object alongside the mesh it was computed on.

    Several stages of the computation are stored, and **only the raw log-compressed field is the one
    downstream consumes**. The file also holds a display variant that has been stretched against
    this object's own percentiles; that variant exists for rendering and must never be fed forward.
    Per-shape stretching maps every object's most salient region to the top of the range, so a plain
    bowl and a filigree lattice come out equally "salient" — and the budget downstream, which reads
    absolute values, then collapses toward a constant. Nothing about the result looks wrong until
    reconstruction quality is measured.

    Normalization proper happens downstream and is a **global** percentile map over the whole
    corpus, which is what keeps values comparable across objects and lets a single budget formula
    be valid for a whole dataset.

Key idea
    Multi-scale spectral saliency: curvature seeds diffused by the heat kernel on the mesh's
    cotangent Laplacian, differenced across scales (a difference-of-Gaussians in the spectral
    domain). Working on the Laplacian rather than in screen or voxel space makes the measure
    intrinsic — invariant to how the object is posed or tessellated — and the multi-scale difference
    is what separates genuine structure from both noise and gentle global curvature.

    Two downstream contracts originate here: values stay per-vertex and unsmoothed, and the
    normalization applied later is global rather than per-shape, so saliency remains comparable
    across objects and one budget formula stays valid for a whole dataset.
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import trimesh
import os
import glob
import gc
import random
import time
import igl
import pymeshlab
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from multiprocessing import Process, set_start_method


# ---------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------
@dataclass
class Mesh:
    V: np.ndarray  # (N,3)
    F: np.ndarray  # (M,3), int


# ---------------------------------------------------------------
# PyTorch GPU Cotangent Laplacian & mass matrix
# ---------------------------------------------------------------
def cotangent_laplacian_and_mass_torch(V_gpu, F_gpu, device='cuda'):
    """
    Compute cotangent Laplacian and lumped mass matrix on GPU.

    Returns:
        L: sparse COO Laplacian (N, N)
        M_diag: (N,) diagonal mass vector
    """
    N = V_gpu.shape[0]
    i0, i1, i2 = F_gpu[:, 0], F_gpu[:, 1], F_gpu[:, 2]
    v0, v1, v2 = V_gpu[i0], V_gpu[i1], V_gpu[i2]

    e0 = v2 - v1  # opposite v0
    e1 = v0 - v2  # opposite v1
    e2 = v1 - v0  # opposite v2

    # 2 * area
    cross012 = torch.cross(v1 - v0, v2 - v0, dim=1)
    dblA = torch.norm(cross012, dim=1)
    dblA = torch.clamp(dblA, min=1e-20)

    def safe_cot_torch(a, b):
        num = (a * b).sum(dim=1)
        den = torch.norm(torch.cross(a, b, dim=1), dim=1)
        den = torch.clamp(den, min=1e-20)
        cot = num / den
        return torch.clamp(cot, -1e6, 1e6)

    cot0 = safe_cot_torch(e1, -e2)
    cot1 = safe_cot_torch(e2, -e0)
    cot2 = safe_cot_torch(e0, -e1)

    # Build sparse weight matrix (symmetric)
    I = torch.cat([i1, i2, i2, i0, i0, i1])
    J = torch.cat([i2, i1, i0, i2, i1, i0])
    W = 0.5 * torch.cat([cot0, cot0, cot1, cot1, cot2, cot2])

    indices = torch.stack([I, J])
    Wmat_coo = torch.sparse_coo_tensor(indices, W, (N, N), device=device).coalesce()

    # Remove diagonal entries
    mask = Wmat_coo.indices()[0] != Wmat_coo.indices()[1]
    Wmat = torch.sparse_coo_tensor(
        Wmat_coo.indices()[:, mask],
        Wmat_coo.values()[mask],
        (N, N),
        device=device
    ).coalesce()

    # Symmetrize
    Wmat_T = torch.sparse_coo_tensor(
        torch.stack([Wmat.indices()[1], Wmat.indices()[0]]),
        Wmat.values(),
        (N, N),
        device=device
    )
    Wmat = (Wmat + Wmat_T) * 0.5
    Wmat = Wmat.coalesce()

    # Row sums for diagonal
    d = torch.sparse.sum(Wmat, dim=1).to_dense()

    # L = D - W
    L_indices = Wmat.indices()
    L_values = -Wmat.values()

    diag_indices = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
    L_indices = torch.cat([L_indices, diag_indices], dim=1)
    L_values = torch.cat([L_values, d])

    L = torch.sparse_coo_tensor(L_indices, L_values, (N, N), device=device).coalesce()

    # Lumped mass matrix
    A = 0.5 * dblA
    M_diag = torch.zeros(N, dtype=torch.float64, device=device)
    M_diag.scatter_add_(0, i0, A / 3.0)
    M_diag.scatter_add_(0, i1, A / 3.0)
    M_diag.scatter_add_(0, i2, A / 3.0)
    M_diag = torch.where(M_diag > 1e-18, M_diag,
                         torch.tensor(1e-18, device=device, dtype=torch.float64))

    return L, M_diag


# ---------------------------------------------------------------
# Seed scalar field: mean curvature via igl
# ---------------------------------------------------------------
def compute_seed_scalar_field(mesh, area_tol=1e-10, alpha=1e-8):
    """Compute mean curvature using libigl with proper error handling."""
    V = mesh.vertices
    F_arr = mesh.faces

    # Remove degenerate faces
    face_areas = igl.doublearea(V, F_arr) * 0.5
    mask = face_areas > area_tol
    F_clean = F_arr[mask]

    if len(F_clean) < 3:
        print("Warning: Too few valid faces after degenerate removal, using original faces")
        F_clean = F_arr

    try:
        L = igl.cotmatrix(V, F_clean)
        M = igl.massmatrix(V, F_clean, igl.MASSMATRIX_TYPE_VORONOI)
    except Exception as e:
        print(f"Warning: igl computation failed ({e}), falling back to zero curvature")
        H_signed = np.zeros(len(V))
        return H_signed

    diag = M.diagonal().copy()
    valid = diag > area_tol

    if not np.any(valid):
        diag_reg = diag + alpha
    else:
        diag_reg = np.where(valid, diag + alpha, diag.mean() + alpha)

    M_reg = sp.diags(diag_reg).tocsc()

    Hn = np.zeros_like(V)
    try:
        for dim in range(3):
            rhs = -0.5 * L.dot(V[:, dim])
            Hn[:, dim] = spsolve(M_reg, rhs)
    except Exception as e:
        print(f"Warning: Linear solve failed ({e}), using zero curvature")
        Hn = np.zeros_like(V)

    H = np.linalg.norm(Hn, axis=1)

    try:
        normals = igl.per_vertex_normals(V, F_clean)
    except Exception:
        normals = np.zeros_like(V)
        normals[:, 2] = 1.0

    finite = np.isfinite(normals).all(axis=1)
    norms = np.linalg.norm(normals, axis=1)
    valid_normals = finite & (norms > 1e-8)
    normals[~valid_normals] = np.array([0.0, 0.0, 1.0])
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / (norms[:, None] + 1e-8)

    signs = np.sign(np.einsum('ij,ij->i', Hn, normals))
    H_signed = H * signs

    nan_mask = np.isnan(H_signed)
    if np.any(nan_mask):
        print(f"Warning: {np.sum(nan_mask)} NaN values in curvature, setting to zero")
        H_signed[nan_mask] = 0.0

    # Clip outliers (5th-95th percentile range)
    t_curvature = H_signed[H_signed.argsort()]
    curvature_range = t_curvature[H_signed.shape[0] // 20: H_signed.shape[0] // 20 * 19]
    min_clip = curvature_range.min() - curvature_range.mean() + curvature_range.min()
    max_clip = curvature_range.max() - curvature_range.mean() + curvature_range.max()
    H_signed = np.clip(H_signed, min_clip, max_clip)
    return H_signed


# ---------------------------------------------------------------
# Sparse helper utilities
# ---------------------------------------------------------------
def sparse_add_diag(sparse_mat, diag_vec):
    """Add diagonal matrix to sparse matrix: A + diag(d)"""
    N = sparse_mat.shape[0]
    device = sparse_mat.device
    diag_indices = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
    all_indices = torch.cat([sparse_mat.indices(), diag_indices], dim=1)
    all_values = torch.cat([sparse_mat.values(), diag_vec])
    result = torch.sparse_coo_tensor(all_indices, all_values, sparse_mat.shape, device=device)
    return result.coalesce()


# ---------------------------------------------------------------
# Conjugate gradient solver
# ---------------------------------------------------------------
def conjugate_gradient_sparse(A_sparse, b, M_diag, maxiter=500, rtol=1e-8):
    """
    CG solver for sparse SPD system A x = b with diagonal preconditioner.
    """
    x = torch.zeros_like(b)
    M_inv = 1.0 / (M_diag + 1e-12) if M_diag is not None else torch.ones_like(b)

    r = b - torch.sparse.mm(A_sparse, x.unsqueeze(1)).squeeze(1)
    z = M_inv * r
    p = z.clone()
    rz_old = torch.dot(r, z)

    for i in range(maxiter):
        Ap = torch.sparse.mm(A_sparse, p.unsqueeze(1)).squeeze(1)
        pAp = torch.dot(p, Ap)
        alpha = rz_old / (pAp + 1e-20)
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.norm(r)
        if r_norm < rtol * torch.norm(b):
            break

        z = M_inv * r
        rz_new = torch.dot(r, z)
        beta = rz_new / (rz_old + 1e-20)
        p = z + beta * p
        rz_old = rz_new

    return x


# ---------------------------------------------------------------
# Heat diffusion smoothing on GPU
# ---------------------------------------------------------------
def heat_smooth_scalar_torch(M_diag, L_sparse, U_gpu, t, device='cuda'):
    """
    Solve (M + t*L) u = M * U  via CG.
    """
    N = M_diag.shape[0]

    M_sparse = torch.sparse_coo_tensor(
        torch.arange(N, device=device).unsqueeze(0).repeat(2, 1),
        M_diag,
        (N, N),
        device=device
    )

    A = M_sparse + t * L_sparse
    A = A.coalesce()

    reg = 1e-10 * M_diag.max()
    A = sparse_add_diag(A, torch.full((N,), reg, device=device, dtype=torch.float64))

    b = M_diag * U_gpu
    u = conjugate_gradient_sparse(A, b, M_diag, maxiter=500, rtol=1e-8)
    return u


# ---------------------------------------------------------------
# Laplacian smoothing
# ---------------------------------------------------------------
def smooth_scalar_laplacian_torch(V_gpu, F_gpu, S_gpu, tau=1e-4, device='cuda'):
    """Laplacian smoothing of scalar field on mesh."""
    L_sparse, M_diag = cotangent_laplacian_and_mass_torch(V_gpu, F_gpu, device)

    N = M_diag.shape[0]
    M_sparse = torch.sparse_coo_tensor(
        torch.arange(N, device=device).unsqueeze(0).repeat(2, 1),
        M_diag,
        (N, N),
        device=device
    )

    A = M_sparse + tau * L_sparse
    A = A.coalesce()

    b = M_diag * S_gpu
    x = conjugate_gradient_sparse(A, b, M_diag, maxiter=500, rtol=1e-8)
    return x


# ---------------------------------------------------------------
# Multi-scale spectral saliency
# ---------------------------------------------------------------
@dataclass
class SaliencyConfig:
    scales: int = 5
    delta_scale: float = 0.003   # delta = delta_scale * bbox_diag
    k_factor: float = 2.0       # scale separation factor
    tau_smooth: float = 1e-4     # final smoothing parameter


def mesh_saliency_multiscale_torch(mesh, cfg=None, device='cuda'):
    """
    Multi-scale spectral mesh saliency via heat-diffusion DoG
    on the cotangent Laplacian.

    Returns dict with saliency arrays including S_log_gpu.
    """
    if cfg is None:
        cfg = SaliencyConfig()

    V_cpu, F_cpu = mesh.V, mesh.F

    print(f"Transferring data to {device}...")
    V_gpu = torch.tensor(V_cpu, dtype=torch.float64, device=device)
    F_gpu = torch.tensor(F_cpu, dtype=torch.long, device=device)

    print("Computing Laplacian and mass matrix on GPU...")
    L_sparse, M_diag = cotangent_laplacian_and_mass_torch(V_gpu, F_gpu, device)

    print("Computing seed field (curvature) on CPU...")
    U_cpu = compute_seed_scalar_field(
        trimesh.Trimesh(vertices=V_cpu, faces=F_cpu, process=False)
    )
    U_gpu = torch.tensor(U_cpu, dtype=torch.float64, device=device)

    # Multi-scale parameters
    bbox_diag = float(torch.norm(V_gpu.max(0)[0] - V_gpu.min(0)[0]))
    delta = cfg.delta_scale * bbox_diag
    t0 = delta ** 2
    k_scale = 1.6
    ts = [t0 * (k_scale ** (2 * i)) for i in range(cfg.scales)]

    print(f"Time scales (ts): {ts}")

    # Multi-scale DoG accumulation
    print("Computing multi-scale saliency on GPU...")
    S_acc_gpu = torch.zeros(V_gpu.shape[0], dtype=torch.float64, device=device)

    for i, t in enumerate(ts):
        print(f"  Scale {i+1}/{cfg.scales}: t={t:.6e}")
        u_t_gpu = heat_smooth_scalar_torch(M_diag, L_sparse, U_gpu, t, device)
        u_kt_gpu = heat_smooth_scalar_torch(M_diag, L_sparse, U_gpu, cfg.k_factor * t, device)
        dog_gpu = torch.abs(u_kt_gpu - u_t_gpu)
        S_acc_gpu += dog_gpu

    # Final smoothing
    print("Applying final smoothing on GPU...")
    S_sm_gpu = smooth_scalar_laplacian_torch(V_gpu, F_gpu, S_acc_gpu,
                                              tau=cfg.tau_smooth, device=device)

    # Post-processing: log enhancement + percentile stretching
    print("Post-processing (log enhancement + percentile stretching)...")
    S_sm_gpu1 = torch.clamp(S_sm_gpu, min=0.0)
    S_log_gpu = torch.log1p(S_sm_gpu1)

    S_final_gpu_norm = (S_log_gpu - S_log_gpu.min()) / (S_log_gpu.max() - S_log_gpu.min() + 1e-12)

    p_low = torch.quantile(S_final_gpu_norm, 0.05)
    p_high = torch.quantile(S_final_gpu_norm, 0.95)
    S_final_gpu = torch.clamp((S_final_gpu_norm - p_low) / (p_high - p_low + 1e-12), 0.0, 1.0)

    print("Transferring result back to CPU...")
    S_final_cpu = S_final_gpu.cpu().numpy()
    print(f"Saliency computation complete. Range: [{S_final_cpu.min():.4f}, {S_final_cpu.max():.4f}]")

    return {
        "original_saliency": S_acc_gpu.cpu().numpy(),
        "smoooth_saliency": S_sm_gpu.cpu().numpy(),
        "S_sm_gpu1": S_sm_gpu1.cpu().numpy(),
        "S_log_gpu": S_log_gpu.cpu().numpy(),
        "S_final_gpu_norm": S_final_gpu_norm.cpu().numpy(),
        "S_final_gpu": S_final_gpu.cpu().numpy(),
        "V_gpu": V_gpu.cpu().numpy(),
        "F_gpu": F_gpu.cpu().numpy(),
    }


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def normalize_to_unit_box(V):
    """Translate to centroid and scale to unit bounding box."""
    center = V.mean(axis=0)
    Vc = V - center
    diag = np.linalg.norm(Vc.max(0) - Vc.min(0))
    if diag < 1e-12:
        return V
    return Vc / diag


def taubin_smooth_trimesh(mesh, lambda_=0.5, mu=-0.53, n_iterations=30,
                          selected=False):
    """Apply Taubin smoothing via pymeshlab."""
    ms = pymeshlab.MeshSet()
    v = mesh.vertices.astype(np.float64)
    f = mesh.faces.astype(np.int32)
    ms.add_mesh(pymeshlab.Mesh(v, f))

    ms.apply_coord_taubin_smoothing(
        lambda_=lambda_,
        mu=mu,
        stepsmoothnum=n_iterations,
        selected=selected
    )

    m_out = ms.current_mesh()
    v_out = m_out.vertex_matrix()
    f_out = m_out.face_matrix()
    return trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)


# ---------------------------------------------------------------
# Process a single PLY file
# ---------------------------------------------------------------
def process_ply_file(ply_file_path, output_dir, target_resolution, cuda_idx=0):
    """
    Process one watertight PLY → spectral saliency NPZ.

    Returns 0 on success, 1 on failure.
    """
    device = f'cuda:{cuda_idx}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_idx)

    mesh_name = os.path.basename(ply_file_path).replace(
        f"_{target_resolution}_watertight.ply", "").replace("_rfa.ply","").replace("_final.ply","").replace(".ply","")
    mesh_saliency_path = os.path.join(
        output_dir, f"{mesh_name}_mesh_saliency_energy_{target_resolution}.npz")

    if os.path.exists(mesh_saliency_path):
        print(f"Skipping {mesh_name} - already processed")
        return 0

    try:
        print(f"Loading mesh from {ply_file_path}...")
        raw_mesh = trimesh.load(ply_file_path, process=False)
        raw_mesh = raw_mesh.split(only_watertight=False)[0]

        print("Smoothing mesh...")
        # input is already the smoothed final mesh (refine, or taubin(wt)x30 fallback); no second smoothing here
        smoothed_mesh = raw_mesh

        V = np.asarray(smoothed_mesh.vertices, dtype=np.float64)
        F_arr = np.asarray(smoothed_mesh.faces, dtype=np.int32)
        print(f"Mesh loaded: {V.shape[0]} vertices, {F_arr.shape[0]} faces")

        V_normalized = normalize_to_unit_box(V)
        mesh = Mesh(V_normalized, F_arr)

        cfg = SaliencyConfig(
            scales=5,
            delta_scale=0.003,
            k_factor=2.0,
            tau_smooth=1e-4
        )

        print(f"Computing mesh saliency for {mesh_name} on {device}...")
        result = mesh_saliency_multiscale_torch(mesh, cfg, device=device)

        print(f"Saving saliency data to {mesh_saliency_path}")
        np.savez_compressed(
            mesh_saliency_path,
            original_saliency=result["original_saliency"].astype(np.float32),
            smooth_saliency=result["smoooth_saliency"].astype(np.float32),
            S_sm_gpu1=result["S_sm_gpu1"].astype(np.float32),
            S_log_gpu=result["S_log_gpu"].astype(np.float32),
            S_final_gpu_norm=result["S_final_gpu_norm"].astype(np.float32),
            S_final_gpu=result["S_final_gpu"].astype(np.float32),
            V_gpu=result["V_gpu"].astype(np.float32),
            F_gpu=result["F_gpu"].astype(np.int32),
            vertices=smoothed_mesh.vertices.astype(np.float32),
            faces=smoothed_mesh.faces.astype(np.int32),
            volume_shape=np.array(
                [target_resolution, target_resolution, target_resolution],
                dtype=np.int16),
        )

        torch.cuda.empty_cache()
        gc.collect()

        print(f"[SUCCESS] Completed {mesh_name}")
        return 0

    except Exception as e:
        print(f"[ERROR] Failed to process {mesh_name}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------
# Main: CLI + SLURM array dispatch
# ---------------------------------------------------------------


def main():
    import argparse

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Compute spectral mesh saliency from watertight PLY meshes")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing *_<res>_watertight.ply files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save *_mesh_saliency_energy_<res>.npz')
    parser.add_argument('--target_resolution', type=int, default=1024,
                        help='Target resolution (for filename matching)')
    parser.add_argument('--cuda_idx', type=int, default=0,
                        help='CUDA device index')
    parser.add_argument('--task_id', type=int, default=0,
                        help='SLURM array task ID for distributed processing')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Per-mesh timeout in seconds')
    parser.add_argument('--files_per_task', type=int, default=200,
                        help='Size of the contiguous slice each task claims')
    parser.add_argument('--shuffle_seed', type=int, default=0,
                        help='Seed for the shuffle that spreads heavy meshes across tasks. Every '
                             'task must use the same value: the slice each one takes is an offset '
                             'into the shuffled list, so tasks that shuffle differently do not '
                             'partition the corpus — they overlap and leave gaps, while each still '
                             'reports having processed its share')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    slurm_task_id = args.task_id
    print(f"SLURM_ARRAY_TASK_ID: {slurm_task_id}")

    ply_pattern = os.path.join(
        args.input_dir, f"*_{args.target_resolution}_watertight.ply")
    all_ply_files = sorted(glob.glob(ply_pattern))
    random.Random(args.shuffle_seed).shuffle(all_ply_files)
    print(f"Found {len(all_ply_files)} total PLY files")

    # Partition for this task. The shuffle above is seeded, so every task derives the same ordering
    # and the slices below tile the corpus exactly once.
    files_per_task = args.files_per_task
    start_idx = slurm_task_id * files_per_task
    end_idx = start_idx + files_per_task

    ply_files_to_process = all_ply_files[start_idx:end_idx]
    print(f"Processing files {start_idx} to {end_idx} "
          f"({len(ply_files_to_process)} files)")

    if len(ply_files_to_process) == 0:
        print("No files to process in this range")
        exit(0)

    for ply_file_path in tqdm(ply_files_to_process, desc="Processing PLY files"):
        mesh_name = os.path.basename(ply_file_path).replace(
            f"_{args.target_resolution}_watertight.ply", "")

        try:
            print(f"\nProcessing {mesh_name}...")

            process = Process(
                target=process_ply_file,
                args=(
                    ply_file_path,
                    args.output_dir,
                    args.target_resolution,
                    args.cuda_idx
                )
            )

            process.start()
            process.join(timeout=args.timeout)

            if process.is_alive():
                print(f"Timeout ({args.timeout}s) reached for {mesh_name}, "
                      "terminating process...")
                process.terminate()
                process.join()
                print(f"Process terminated for {mesh_name}")
            else:
                if process.exitcode == 0:
                    print(f"Successfully completed {mesh_name}")
                else:
                    print(f"Process failed for {mesh_name} "
                          f"with exit code {process.exitcode}")

            process.close()

        except Exception as e:
            print(f"Error processing {mesh_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nProcessing complete! Processed {len(ply_files_to_process)} files")


# ---------------------------------------------------------------------------
# Production invocation (per task, tasks tile the corpus):
#
#   python a_spectral_saliency.py --input_dir <sharp mesh dir> \
#       --output_dir <saliency dir> --target_resolution 512 --task_id $i
#
# The input is the sharpened mesh, before the Laplacian smoothing that produces
# the occupancy — saliency answers "what is detailed" and has to see the detail,
# while occupancy answers "where is the object" and is easier to predict smoothed.
# The two are different meshes on purpose.
#
# Defaults kept: 5 scales, delta_scale=0.003 of the bbox diagonal, k_factor=2.0,
# tau_smooth=1e-4. --shuffle_seed must be identical across tasks.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
