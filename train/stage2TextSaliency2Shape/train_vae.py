"""Train the supervoxel autoencoder — the alphabet the shape generator writes in.

Purpose
    Everything downstream is defined by this model: the generator's entire training set is the
    token stream this encoder produced, so a differently-trained autoencoder invalidates it
    silently. This entry point exists so that alphabet is reproducible from this repository rather
    than from a private framework.

Input
    A JSON config naming the encoder/decoder classes and their arguments, the dataset root of
    `.vxz` voxelizations, and the matching supervoxel centers (`.ply`) from CVT sampling.
"""
import argparse
import json
import os
from typing import Any, Dict, Optional

import torch

from ..vendor.trellis2.modules import sparse as sp
from ..vendor.trellis2.trainers.vae.subdivision_vae_full import SubdivisionVaeFullTrainer
from . import vae as vae_module
from ..utils.config import flatten_args


def resolve_path(base_dir: Optional[str], path: Optional[str]) -> Optional[str]:
    """Config paths are relative to the config file, so a config stays portable."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir or ".", path))


def apply_sparse_settings(cfg: Dict[str, Any]) -> None:
    """Select the sparse backends before any model is built; they are process-wide.

    Accepts both the nested `sparse: {backend, conv_backend}` block the shipped config uses and
    the older flat `sparse_backend` / `sparse_conv_backend` keys, matching what
    `inference/text2shape.py` reads. Reading only the flat keys left the shipped config selecting
    flex_gemm at inference and silently falling back to the default backend for training.
    """
    nested = cfg.get("sparse") if isinstance(cfg.get("sparse"), dict) else {}
    backend = nested.get("backend", cfg.get("sparse_backend"))
    if backend:
        sp.config.set_backend(backend) if hasattr(sp, "config") else None
    conv = nested.get("conv_backend", cfg.get("sparse_conv_backend"))
    if conv:
        from ..vendor.trellis2.modules.sparse.conv import config as conv_config
        conv_config.set_conv_backend(conv) if hasattr(conv_config, "set_conv_backend") else None
    print(f"[train_vae] sparse backend={backend or 'default'} conv_backend={conv or 'default'}",
          flush=True)


def load_backbone(model: torch.nn.Module, weights: str, ckpts_dir: str, what: str,
                  prefixes: Optional[list] = None) -> None:
    """Load the pretrained backbone the released run fine-tuned from.

    The released autoencoder was not trained from scratch: it started from the TRELLIS sparse
    convolution backbone. Training without it converges somewhere else entirely, so a missing file
    warns and falls back to random initialisation rather than failing the run.

    The backbone file is written in its own namespace, while the model nests it under a submodule
    (`encoder.` for the encoder; the decoder is already flat). `models.<name>.require_loaded_prefixes`
    in the config names that nesting. Getting it wrong is silent — every key lands nowhere, the load
    "succeeds", and the backbone is quietly discarded — so the result is checked, not assumed.
    """
    path = weights if os.path.isabs(weights) else os.path.join(ckpts_dir, os.path.basename(weights))
    if not os.path.exists(path):
        print(f"[train_vae] WARNING: {what} backbone not found at {path}; starting from random "
              f"initialisation. The released run fine-tuned from it — fetch it with "
              f"ckpts/fetch_ckpts.sh to reproduce that lineage.", flush=True)
        return

    from safetensors.torch import load_file
    sd = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)

    # Try the declared nesting first, then the file as written; keep whichever places every key.
    candidates = [(p, {p + k: v for k, v in sd.items()}) for p in (prefixes or [])]
    candidates.append(("", sd))
    best_prefix, best = None, None
    for prefix, cand in candidates:
        try:
            result = model.load_state_dict(cand, strict=False)
        except RuntimeError as e:
            # Same names, different shapes: a file for a different model, or different args.
            raise SystemExit(
                f"{what}: {os.path.basename(path)} has parameters of the wrong shape for this "
                f"model. Check models.{what}.weights and models.{what}.args in the config.\n"
                f"  {str(e).splitlines()[0]}") from None
        if best is None or len(result.unexpected_keys) < len(best.unexpected_keys):
            best_prefix, best = prefix, result
        if not result.unexpected_keys:
            break

    if best.unexpected_keys:
        raise SystemExit(
            f"{what}: {os.path.basename(path)} does not fit this model — {len(best.unexpected_keys)} "
            f"of its tensors match no parameter (e.g. {best.unexpected_keys[:3]}). Loading it would "
            f"leave the backbone randomly initialised while reporting success. Check "
            f"models.{what}.require_loaded_prefixes in the config.")

    into = f"into {best_prefix!r}" if best_prefix else "flat"
    print(f"[train_vae] {what} backbone {os.path.basename(path)}: {len(sd)} tensors loaded {into}; "
          f"{len(best.missing_keys)} parameters start random (the heads this stage adds)", flush=True)


def build_model(spec: Dict[str, Any], device: torch.device, ckpts_dir: str,
                what: str = "") -> torch.nn.Module:
    """Look the class up by the name the config gives, exactly as the released runs did."""
    name = spec["name"]
    cls = getattr(vae_module, name, None)
    if cls is None:
        raise ValueError(
            f"unknown model class {name!r}. The flattened classes in vae.py expose the released "
            f"names as aliases; add one there if this config predates them.")
    model = cls(**spec.get("args", {})).to(device)
    if spec.get("weights"):
        load_backbone(model, spec["weights"], ckpts_dir, what or name,
                      spec.get("require_loaded_prefixes"))
    return model


def parse_ema_rates(value) -> list:
    if value is None:
        return []
    return [float(v) for v in (value if isinstance(value, (list, tuple)) else [value])]


def setup_distributed() -> torch.device:
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        return torch.device("cuda", local)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the supervoxel autoencoder.")
    ap.add_argument("--config", required=True, help="JSON config: models, dataset, trainer, resume")
    ap.add_argument("--output-dir", default=None, help="overrides the config's output_dir")
    ap.add_argument("--dataset-root", default=None, help="overrides dataset.root")
    ap.add_argument("--cvt-root", default=None, help="overrides dataset.cvt_root (supervoxel centers)")
    ap.add_argument("--max-steps", type=int, default=None, help="overrides trainer.max_steps")
    ap.add_argument("--ckpts", default=None,
                    help="where the pretrained backbones named by models.*.weights live "
                         "(default: the repository's ckpts/)")
    ap.add_argument("--validation-size", type=int, default=100,
                    help="reserve the first N records of dataset.train_captions")
    args = ap.parse_args(argv)

    base_dir = os.path.dirname(os.path.abspath(args.config))
    cfg = json.load(open(args.config))
    apply_sparse_settings(cfg)
    device = setup_distributed()

    dataset_cfg = flatten_args(cfg["dataset"])
    if not dataset_cfg.get("include_cvt_points", False):
        raise SystemExit(
            "dataset.include_cvt_points must be true: the encoder reads supervoxel centers, and "
            "without them it would train on a different input than the one inference provides")
    if args.dataset_root:
        dataset_cfg["root"] = args.dataset_root
    if args.cvt_root:
        dataset_cfg["cvt_root"] = args.cvt_root
    train_captions = dataset_cfg.get("train_captions")
    if not train_captions:
        raise SystemExit("dataset.train_captions is required to keep test objects out of training")

    from ..dataset.shape_vae import CENTER_SCALE, ShapeVaeDataset
    dataset = ShapeVaeDataset(
        resolve_path(base_dir, dataset_cfg["root"]),
        resolution=int(dataset_cfg.get("resolution", 1024)),
        cvt_root=resolve_path(base_dir, dataset_cfg.get("cvt_root")),
        cvt_suffix=dataset_cfg.get("cvt_suffix", "_saliency_volume_64_supervoxel_center.ply"),
        center_scale=float(dataset_cfg.get("center_scale", CENTER_SCALE)),
        num_threads=int(dataset_cfg.get("num_threads", 4)),
        include_cvt_points=True,
        max_active_voxels=dataset_cfg.get("max_active_voxels"),
        caption_index=resolve_path(base_dir, train_captions),
        skip_first_ids=args.validation_size,
    )

    ckpts_dir = args.ckpts or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ckpts")
    models = {name: build_model(spec, device, ckpts_dir, name)
              for name, spec in cfg["models"].items()}
    resolution = int(dataset_cfg.get("resolution", 1024))
    for m in models.values():
        if hasattr(m, "set_resolution"):
            m.set_resolution(resolution)
        m.train()

    trainer_cfg = flatten_args(cfg["trainer"])
    max_steps = int(args.max_steps or trainer_cfg.pop("max_steps"))
    trainer_cfg.pop("max_steps", None)
    batch_size_per_gpu = int(trainer_cfg.pop("batch_size_per_gpu"))
    batch_split = int(trainer_cfg.pop("batch_split", 1))

    out_dir = resolve_path(base_dir, args.output_dir or cfg["output_dir"])
    resume = cfg.get("resume", {})
    load_dir = resolve_path(base_dir, resume.get("load_dir")) or out_dir
    step = resume.get("step")

    trainer = SubdivisionVaeFullTrainer(
        models=models, dataset=dataset, output_dir=out_dir, load_dir=load_dir,
        step=int(step) if step is not None else None,
        max_steps=max_steps, batch_size_per_gpu=batch_size_per_gpu, batch_split=batch_split,
        **trainer_cfg,
    )
    trainer.run()


if __name__ == "__main__":
    main()
