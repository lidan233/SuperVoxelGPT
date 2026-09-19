#!/usr/bin/env python3
"""Turn a caption into a mesh — the whole chain, in one file.
"""
import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
import trimesh
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "train", "vendor", "runtime", "pycvt"))
_RUNTIME = os.environ.get("T2S_RUNTIME")
if _RUNTIME:
    sys.path.insert(0, _RUNTIME)

sys.path.insert(0, os.path.dirname(_REPO))
_PKG = os.path.basename(_REPO)
_train = __import__(f"{_PKG}.train", fromlist=["dataset"])
from importlib import import_module

from pycvt import pycvt_from_field

sal_vae = import_module(f"{_PKG}.train.stage1Text2Saliency.vae")
shape_vae = import_module(f"{_PKG}.train.stage2TextSaliency2Shape.vae")
Text3DModelMaskGITSingleCached = import_module(
    f"{_PKG}.train.stage1Text2Saliency.model").Text3DModelMaskGITSingleCached
Text3DQwenAR = import_module(f"{_PKG}.train.stage2TextSaliency2Shape.model").Text3DQwenAR
sp = import_module(f"{_PKG}.train.vendor.trellis2.modules.sparse")


def _warn_checkpoint_mismatch(result, label: str) -> None:
    if result.missing_keys or result.unexpected_keys:
        warnings.warn(
            f"{label}: checkpoint loaded with strict=False; "
            f"{len(result.missing_keys)} missing keys {result.missing_keys[:5]}, "
            f"{len(result.unexpected_keys)} unexpected keys {result.unexpected_keys[:5]}. "
            "Those parameters may remain randomly initialized or unused.",
            RuntimeWarning, stacklevel=2)

def apply_tf32(force: Optional[bool] = None) -> bool:
    """Turn off every source of run-to-run drift this chain is known to have."""
    on = force if force is not None else (os.environ.get("SVOX_DET", "1") == "1")
    if not on:
        return False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for fn in ("enable_flash_sdp", "enable_mem_efficient_sdp"):
        try:
            getattr(torch.backends.cuda, fn)(False)
        except Exception:
            pass
    try:
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    return True


_DET = apply_tf32()


_SAL_FILE = {"encoder": "stage1SaliencyVaeEncoder.pt",
             "mask_decoder": "stage1SaliencyVaeOccupancyDecoder.pt",
             "saliency_decoder": "stage1SaliencyVaeSaliencyDecoder.pt"}
_SAL_LEGACY_STEP = "0242000"
_BACKBONE_RENAME = {
    "shape_enc_next_dc_f16c32_fp16.safetensors": "trellisEncoderOriginal.safetensors",
    "shape_dec_next_dc_f16c32_fp16.safetensors": "trellisDecoderOriginal.safetensors",
}


def _pick(ckpts: str, flat: str, *legacy: str) -> str:
    """Flat release name if it is there, otherwise the nested one."""
    p = os.path.join(ckpts, flat)
    return p if os.path.exists(p) else os.path.join(ckpts, *legacy)


def rebase_ckpt_paths(cfg, ckpts_dir: str):
    """Point the backbone weights at this machine."""
    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str) and "base_safetensors" in o:
            name = os.path.basename(o)
            new = _BACKBONE_RENAME.get(name)
            if new is not None:
                flat = os.path.join(ckpts_dir, new)
                if os.path.exists(flat):
                    return flat
            return os.path.join(ckpts_dir, "base_safetensors", name)
        return o
    return walk(cfg)


def clip_path(ckpts: str) -> str:
    """Local CLIP copy first; reproducibility must not depend on the network."""
    p = os.environ.get("T2S_CLIP")
    if p:
        return p
    local = os.path.join(ckpts, "clipText")
    if os.path.isdir(local) and os.path.exists(os.path.join(local, "config.json")):
        return local
    return "openai/clip-vit-large-patch14"


CVT_RES = 256          # density grid the CVT relaxation runs on
CVT_T = 0.05           # saliency floor below which a cell gets the coarsest spacing
CVT_K = 3              # spacing ratio between the coarsest and finest cells
GRID = 64              # occupancy resolution the saliency stage emits
MESH_RES = 1024        # dual-grid resolution the shape decoder extracts at


def size_field(x, t=CVT_T, k=CVT_K):
    xc = np.clip(np.asarray(x, np.float64), t, 1.0)
    f = (xc - t) / (1.0 - t) * (1.0 - k) + k
    return 1.0 / np.power(f, 5)


def token_budget(x, t=CVT_T, k=CVT_K) -> float:
    """N = sum 1/f^3 — how many supervoxels the saliency field asks for."""
    xc = np.clip(x, t, 1.0)
    f = ((xc - t) / (1.0 - t)) * (1.0 - k) + k
    return float(np.sum(1.0 / np.power(f, 3)))


def upsample_density(vol64: np.ndarray, res: int = CVT_RES) -> np.ndarray:
    """Nearest-neighbour lift of the 64^3 field onto the density grid."""
    t = torch.from_numpy(vol64).float().cuda()
    nz = torch.nonzero(t > 0, as_tuple=False)
    if len(nz) == 0:
        return np.zeros((res,) * 3, np.float32)
    vals = t[t > 0].cpu().numpy()
    low = (nz.float().cpu().numpy() + 0.5) / GRID
    mask = torch.nn.functional.interpolate((t > 0).float()[None, None], size=(res,) * 3,
                                           mode="nearest")[0, 0].bool()
    hi = torch.nonzero(mask, as_tuple=False).cpu().numpy()
    _, nn = cKDTree(low).query((hi + 0.5) / res, k=1, workers=-1)
    out = np.zeros((res,) * 3, np.float32)
    out[hi[:, 0], hi[:, 1], hi[:, 2]] = vals[nn]
    return out


def pca_anchor(coords: np.ndarray) -> int:
    """Deterministic seed for farthest-point sampling: the extreme along the principal axis, with"""
    c = coords.astype(np.float64)
    w = np.full(len(c), 1.0 / len(c))
    mu = (c * w[:, None]).sum(0)
    cc = c - mu
    _, vec = np.linalg.eigh((cc * w[:, None]).T @ cc)
    p1 = cc @ vec[:, -1]
    if (w * (p1 ** 3)).sum() < 0:
        p1 = -p1
    return int(np.argmin(p1))


def farthest_point_sample(coords: np.ndarray, start: int, n: int) -> np.ndarray:
    """CPU float64 with smallest-index tie-breaks — GPU reductions reorder and would not reproduce."""
    pts = coords.astype(np.float64)
    m = len(pts)
    if m <= n:
        sel = list(range(m)) + [i % m for i in range(n - m)]
        return np.array(sel[:n], np.int64)
    sel = np.empty(n, np.int64)
    d2 = np.full(m, np.inf)
    cur = int(start)
    for i in range(n):
        sel[i] = cur
        d2 = np.minimum(d2, ((pts - pts[cur]) ** 2).sum(1))
        cur = int(np.flatnonzero(d2 == d2.max())[0])
    return sel


def _part1by2(x):
    """Spread the low 10 bits of x so two zero bits sit between consecutive bits."""
    x = x & 0x3FF
    x = (x | (x << 16)) & 0x30000FF
    x = (x | (x << 8)) & 0x300F00F
    x = (x | (x << 4)) & 0x30C30C3
    x = (x | (x << 2)) & 0x9249249
    return x


def morton_order(coords: np.ndarray, grid: int = GRID) -> np.ndarray:
    """z-order over 1-voxel cells, ties broken by full-precision lexicographic order."""
    q = np.clip(((coords.astype(np.float64) + 0.5) * (grid - 1)).round().astype(np.int64),
                0, grid - 1)
    key = (_part1by2(q[:, 2]) << 2) | (_part1by2(q[:, 1]) << 1) | _part1by2(q[:, 0])
    fine = coords[:, 2].astype(np.float64) * 4 + coords[:, 1] * 2 + coords[:, 0]
    return np.lexsort((fine, key))


# -------------------------------------------------------------------------------------------
class Text2Shape:
    """All four models, resident. Construct once, call many times."""

    def __init__(self, ckpts: str, device: str = "cuda", ar_layers: int = 12,
                 mesh_res: int = MESH_RES, verbose: bool = True):
        self.device = torch.device(device)
        self.ckpts = ckpts
        self.verbose = verbose
        flat = os.path.exists(os.path.join(ckpts, "stage1SaliencyVaeConfig.json"))

        from transformers import CLIPTextModel, CLIPTokenizer
        clip = clip_path(ckpts)
        self.tok = CLIPTokenizer.from_pretrained(clip)
        self.clip = CLIPTextModel.from_pretrained(clip).to(self.device).eval()

        # --- stage 1: caption -> saliency code -> 64^3 occupancy + saliency
        sal_dir = ckpts if flat else os.path.join(ckpts, "saliency_vae")
        sal_cfg = json.load(open(
            os.path.join(sal_dir, "stage1SaliencyVaeConfig.json") if flat
            else os.path.join(sal_dir, "dual_decoder_vq_vae_lap100.json")))
        self.sal_enc = self._load_sal(sal_cfg, "encoder", sal_dir, flat)
        self.mask_dec = self._load_sal(sal_cfg, "mask_decoder", sal_dir, flat)
        self.sal_dec = self._load_sal(sal_cfg, "saliency_decoder", sal_dir, flat)

        self.mllm1 = Text3DModelMaskGITSingleCached(
            num_tokens=5625, embed_dim=768, num_heads=8, num_layers=24,
            hidden_dim=2048, dropout=0.0, verbose=False)
        mllm1_ckpt = _pick(ckpts, "stage1Text2Saliency.bin", "mllm1", "pytorch_model.bin")
        result = self.mllm1.load_state_dict(
            torch.load(mllm1_ckpt, map_location="cpu", weights_only=False), strict=False)
        _warn_checkpoint_mismatch(result, f"text2saliency ({mllm1_ckpt})")
        self.mllm1 = self.mllm1.to(self.device).eval()

        lattice = np.array([[z, y, x] for z in range(8) for y in range(8) for x in range(8)],
                           np.float32)
        self.lattice = torch.tensor((lattice / 7.0) - 0.5, device=self.device)

        # --- stage 3: centers + caption -> tokens -> mesh
        self.ar = Text3DQwenAR(num_layers=ar_layers, verbose=False)
        ar_ckpt = _pick(ckpts, "stage2TextSaliency2Shape.bin", "ar_9level", "pytorch_model.bin")
        self._load_ar(ar_ckpt)
        self.ar = self.ar.to(self.device).eval()

        shape_dir = ckpts if flat else os.path.join(ckpts, "shape_vae")
        shape_cfg = json.load(open(
            os.path.join(shape_dir, "stage2SupervoxelVaeConfig.json") if flat
            else os.path.join(shape_dir, "shape_vae_train_config_a4_cb10k_gencvt.json")))
        shape_cfg = rebase_ckpt_paths(shape_cfg, ckpts)
        self._apply_sparse(shape_cfg)
        self.shape_enc = self._load_shape(shape_cfg, "encoder", shape_dir, flat)
        self.shape_dec = self._load_shape(shape_cfg, "decoder", shape_dir, flat)
        if hasattr(self.shape_dec, "set_resolution"):
            self.shape_dec.set_resolution(mesh_res)
        self.mesh_res = mesh_res

        layers = import_module(
            f"{_PKG}.train.stage2TextSaliency2Shape.sparse_vae.layers")
        self._bidir_edges = layers.compute_bidir_edges

    # -- loading ------------------------------------------------------------------------------
    def _load_sal(self, cfg, name, root, flat):
        spec = cfg["models"][name]
        model = getattr(sal_vae, spec["name"])(**spec.get("args", {}))
        p = os.path.join(root, _SAL_FILE[name]) if flat \
            else os.path.join(root, f"{name}_ema0.9999_step{_SAL_LEGACY_STEP}.pt")
        sd = torch.load(p, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        result = model.load_state_dict(sd, strict=False)
        _warn_checkpoint_mismatch(result, f"{name} ({p})")
        return model.to(self.device).eval()

    def _load_shape(self, cfg, name, root, flat):
        spec = dict(cfg["models"][name])
        args = dict(spec.get("args", {}))
        args["use_fp16"] = False        # fp32 weights; autocast picks the compute dtype per block
        model = getattr(shape_vae, spec["name"])(**args)
        p = os.path.join(root, f"stage2SupervoxelVae{name.capitalize()}.pt") if flat \
            else os.path.join(root, f"{name}_step0030000.pt")
        result = model.load_state_dict(torch.load(p, map_location="cpu"), strict=False)
        _warn_checkpoint_mismatch(result, f"shape {name} ({p})")
        return model.to(self.device).eval()

    def _load_ar(self, path: str) -> None:
        """Load the shape generator, tolerating constant buffers but nothing else."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        result = self.ar.load_state_dict(state, strict=False)
        unexpected = [k for k in result.unexpected_keys if not k.endswith(("inv_freq", "freqs"))]
        if result.missing_keys or unexpected:
            raise RuntimeError(
                f"{path} does not match this model — check --ar-layers (the released generator is "
                f"12 layers).\n  missing: {result.missing_keys[:8]}\n"
                f"  unexpected (excluding constant buffers): {unexpected[:8]}")

    @staticmethod
    def _apply_sparse(cfg) -> None:
        backend = cfg.get("sparse", {}).get("backend") if isinstance(cfg.get("sparse"), dict) \
            else cfg.get("sparse_backend")
        conv = cfg.get("sparse", {}).get("conv_backend") if isinstance(cfg.get("sparse"), dict) \
            else cfg.get("sparse_conv_backend")
        if backend and hasattr(sp, "config"):
            sp.config.set_backend(backend)
        if conv:
            conv_cfg = import_module(
                f"{_PKG}.train.vendor.trellis2.modules.sparse.conv").config
            if hasattr(conv_cfg, "set_conv_backend"):
                conv_cfg.set_conv_backend(conv)

    def _sync(self) -> float:
        torch.cuda.synchronize()
        return time.time()

    # -- stages -------------------------------------------------------------------------------
    @torch.no_grad()
    def _encode_caption(self, caption: str):
        enc = self.tok([caption], padding="max_length", truncation=True, max_length=77,
                       return_tensors="pt")
        feats = self.clip(input_ids=enc.input_ids.to(self.device),
                          attention_mask=enc.attention_mask.to(self.device)).last_hidden_state
        return feats, enc.attention_mask.to(self.device)

    @torch.no_grad()
    def centers_for_caption(self, caption: str):
        """caption -> saliency -> supervoxel centers, stopping before the shape generator."""
        feats, mask = self._encode_caption(caption)
        code = self.mllm1.generate(feats, mask, self.lattice[None], num_steps=12)
        z = self.sal_enc.quantizer.indices_to_codes(code.view(-1, 8, 8, 8))
        coords = self.sal_enc._make_coords(z.shape, z.device)
        for attn in self.sal_enc.post_quantize_attns:
            z = attn(z, coords)
        occ_logits = self.mask_dec(z)
        occ_logits = occ_logits[0] if isinstance(occ_logits, tuple) else occ_logits
        sal_pred = self.sal_dec(z)
        sal_pred = sal_pred[0] if isinstance(sal_pred, tuple) else sal_pred
        occupancy = (torch.sigmoid(occ_logits) > 0.5).view(GRID, GRID, GRID).cpu().numpy()
        saliency = (2.0 * sal_pred.view(GRID, GRID, GRID) - 1.0).clamp(0, 1).float().cpu().numpy()

        occ_idx = np.argwhere(occupancy).astype(np.int32)
        occupied_idx = np.argwhere(occupancy).astype(np.int64)
        occupied_val = saliency[occupancy].astype(np.float16).astype(np.float32)
        if len(occupied_idx) == 0:
            return np.zeros((0, 3), np.float32), occ_idx, feats, mask

        centers = pycvt_from_field(occupied_idx, occupied_val)
        return centers, occ_idx, feats, mask

    @torch.no_grad()
    def generate_tokens(self, cvt: np.ndarray, feats, mask, max_iters: int = 256) -> torch.Tensor:
        """One token per supervoxel center, decoded by Jacobi fixed-point iteration."""
        n = len(cvt)
        coords = torch.tensor(np.asarray(cvt, np.float32), device=self.device)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.ar.jacobi_generate(
                text_features=feats, text_attention_mask=mask, coords=coords,
                token_3d_attention_mask=torch.ones(1, n, dtype=torch.long, device=self.device),
                max_iters=max_iters)[0]

    @torch.no_grad()
    def decode_tokens(self, tokens, cvt: np.ndarray, occ_idx: np.ndarray,
                      timings: Optional[Dict[str, float]] = None) -> trimesh.Trimesh:
        """Tokens plus their centers and occupancy -> mesh."""
        import o_voxel
        import cumesh

        tokens_t = tokens if torch.is_tensor(tokens) else torch.tensor(tokens, device=self.device)
        tokens_t = tokens_t.to(self.device).long()
        cvt_t = torch.tensor(np.asarray(cvt, np.float32), device=self.device)
        n, m = len(cvt), len(occ_idx)

        grid_coords = torch.cat(
            [torch.zeros(m, 1, dtype=torch.int32, device=self.device),
             torch.tensor(np.asarray(occ_idx, np.int32), device=self.device)], 1)
        grid_scaled = grid_coords.float()
        grid_scaled[:, 1:] = grid_scaled[:, 1:] / (GRID - 1) - 0.5
        vor_scaled = torch.cat([torch.zeros(n, 1, device=self.device), cvt_t], 1)

        t_bridge = self._sync()
        with torch.autocast("cuda", dtype=torch.float16):
            edges, _ = self._bidir_edges(
                grid_scaled[:, 1:4], vor_scaled[:, 1:4], self.shape_enc.bidir_k,
                torch.zeros(m, dtype=torch.long, device=self.device),
                torch.zeros(n, dtype=torch.long, device=self.device))
            zq = self.shape_enc.quantizer.get_output_from_indices(tokens_t.view(1, n, 1))[0]
            for attn in self.shape_enc.post_quantize_attns:
                zq = attn(zq, vor_scaled, vor_scaled, zq)
            zg = self.shape_enc.voronoi_to_grid_attn(zq, vor_scaled[:, 1:4],
                                                    grid_scaled[:, 1:4], edges)
            hv = zq
            for r in range(self.shape_enc.num_readout_rounds):
                sa = self.shape_enc.iter_selfattn(hv, vor_scaled, vor_scaled, hv)
                hv = hv + self.shape_enc.sa_scale[r] * (sa - hv)
                zg = zg + self.shape_enc.readout_scale[r] * self.shape_enc.iter_readout(
                    zg, hv, vor_scaled[:, 1:4], grid_scaled[:, 1:4], edges)
            if timings is not None:
                timings["bridge"] = (t_dec := self._sync()) - t_bridge
            vertices, intersected, _ = self.shape_dec(
                sp.SparseTensor(zg.to(torch.float16), grid_coords))
        if timings is not None:
            timings["shape_decode"] = (t_dg := self._sync()) - t_dec

        v, f = o_voxel.convert.flexible_dual_grid_to_mesh(
            vertices.coords[:, 1:4].to(torch.int32), vertices.feats.float(),
            (intersected.feats > 0).bool(), split_weight=None,
            grid_size=self.mesh_res, aabb=[[-0.5] * 3, [0.5] * 3])
        if timings is not None:
            timings["dual_grid"] = (t_cl := self._sync()) - t_dg

        cm = cumesh.CuMesh()
        cm.init(v.float(), f.to(torch.int32))
        cm.remove_small_connected_components(0.001)   # the dual grid emits thousands of shards
        cm.fill_holes(int(os.environ.get("T2S_FILL_HOLES", "10")))
        v2, f2 = cm.read()
        if timings is not None:
            timings["cleanup"] = self._sync() - t_cl
        return trimesh.Trimesh(v2.cpu().numpy(), f2.cpu().numpy(), process=False)

    @torch.no_grad()
    def __call__(self, caption: str) -> Tuple[trimesh.Trimesh, Dict[str, float]]:
        t: Dict[str, float] = {}
        t0 = self._sync()
        cvt, occ_idx, feats, mask = self.centers_for_caption(caption)
        t["saliency_and_cvt"] = (t1 := self._sync()) - t0
        if len(cvt) == 0:
            raise RuntimeError(f"the saliency stage produced an empty volume for {caption!r}")
        tokens = self.generate_tokens(cvt, feats, mask)
        t["autoregressive"] = (t2 := self._sync()) - t1
        mesh = self.decode_tokens(tokens, cvt, occ_idx, timings=t)
        t["total"] = self._sync() - t0
        t["supervoxels"] = float(len(cvt))
        t["occupied_cells"] = float(len(occ_idx))
        t["faces"] = float(len(mesh.faces))
        return mesh, t


def read_captions(path: str) -> List[Tuple[str, str]]:
    """A plain caption per line, or JSONL with {id, caption}."""
    out = []
    for i, line in enumerate(open(path)):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            d = json.loads(line)
            out.append((str(d.get("id", i)), d["caption"]))
        else:
            out.append((f"{i:04d}", line))
    return out


def _summarize(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    keys = ["saliency_and_cvt", "autoregressive", "bridge", "shape_decode",
            "dual_grid", "cleanup", "total"]
    ns = [r["supervoxels"] for r in rows]
    print("=" * 72)
    print(f"{len(rows)} captions, loading excluded")
    print(f"  {'supervoxels':<18} mean {np.mean(ns):8.0f}   min {min(ns):.0f}   max {max(ns):.0f}")
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], float)
        if len(v):
            print(f"  {k:<18} mean {v.mean():7.2f}s  min {v.min():6.2f}s  max {v.max():6.2f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Caption to mesh: one caption, or a batch timed at steady state.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--caption", help="a single caption; writes one mesh")
    src.add_argument("--captions", help="file of captions, one per line or JSONL {id, caption}")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ckpts", default=os.path.join(_REPO, "ckpts"),
                    help="checkpoint directory (release layout)")
    ap.add_argument("--ar-layers", type=int, default=12,
                    help="12 matches the released generator; other values cannot load it")
    ap.add_argument("--mesh-res", type=int, default=MESH_RES)
    ap.add_argument("--bench", action="store_true",
                    help="report steady-state per-object time; loading is excluded")
    ap.add_argument("--warmup", type=int, default=1,
                    help="with --bench, discard this many runs before measuring")
    ap.add_argument("--limit", type=int, default=None, help="stop after N captions")
    args = ap.parse_args(argv)

    det = apply_tf32()
    os.makedirs(args.out, exist_ok=True)
    cases = [("0000", args.caption)] if args.caption else read_captions(args.captions)
    if args.limit:
        cases = cases[:args.limit]

    t_load = time.time()
    pipe = Text2Shape(args.ckpts, ar_layers=args.ar_layers, mesh_res=args.mesh_res)
    torch.cuda.synchronize()
    print(f"[text2shape] loaded in {time.time() - t_load:.1f}s "
          f"(excluded from every number below) | SVOX_DET={'1' if det else '0'}", flush=True)

    log = open(os.path.join(args.out, "timings.jsonl"), "a")
    rows = []
    for k, (sid, caption) in enumerate(cases):
        warm = args.bench and k < args.warmup
        try:
            mesh, t = pipe(caption)
        except Exception as e:
            print(f"[{k + 1}/{len(cases)}] {sid} FAILED {e!r}", flush=True)
            continue
        if not (args.bench and warm):
            dst = os.path.join(args.out, f"{sid}.ply")
            mesh.export(dst)
            t["id"] = sid
            log.write(json.dumps({a: (round(b, 4) if isinstance(b, float) else b)
                                  for a, b in t.items()}) + "\n")
            log.flush()
            rows.append(t)
        tag = "warmup" if warm else f"{k + 1}/{len(cases)}"
        print(f"[{tag}] {sid} N={int(t['supervoxels'])} faces={int(t['faces'])} "
              f"total={t['total']:.2f}s", flush=True)

    if args.bench:
        _summarize(rows)
    print(f"[text2shape] done -> {args.out}", flush=True)


# python -m inference.text2shape --captions captions.txt --out runs/bench --bench --warmup 1
if __name__ == "__main__":
    main()
