"""Train the saliency autoencoder — the code the text model actually predicts.
"""
import argparse
import json
import os
from typing import Any, Dict, Optional

import torch

from . import vae as sal_vae
from ..utils.config import constructor_args, flatten_args


def resolve_path(base_dir: Optional[str], path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir or ".", path))


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


def build_model(spec: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Resolve by the name the config gives — the flattened classes in vae.py carry the released
    names, so a config written against the original package still resolves."""
    cls = getattr(sal_vae, spec["name"], None)
    if cls is None:
        raise ValueError(f"unknown model class {spec['name']!r}; expected one defined in "
                         f"stage1Text2Saliency/vae.py")
    return cls(**spec.get("args", {})).to(device)


def main(argv=None):
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description="Train the saliency autoencoder (dual decoder).")
    ap.add_argument("--config", required=True, help="JSON config: models, dataset, trainer")
    ap.add_argument("--output-dir", default=None, help="overrides the config's output_dir")
    ap.add_argument("--data-dir", default=None, help="overrides dataset.roots")
    ap.add_argument("--max-steps", type=int, default=None, help="overrides trainer.max_steps")
    ap.add_argument("--load-dir", default=None,
                    help="resume from this run directory (it holds ckpts/); None starts fresh")
    ap.add_argument("--load-step", type=int, default=0, help="step to resume from within --load-dir")
    ap.add_argument("--validation-size", type=int, default=100,
                    help="reserve the first N records of dataset.train_captions")
    args = ap.parse_args(argv)

    base_dir = os.path.dirname(os.path.abspath(args.config))
    cfg = json.load(open(args.config))
    device = setup_distributed()

    dataset_cfg = flatten_args(cfg["dataset"])
    if args.data_dir:
        dataset_cfg["roots"] = args.data_dir
    train_captions = dataset_cfg.pop("train_captions", None)
    dataset_cfg.pop("test_captions", None)
    if not train_captions:
        raise SystemExit("dataset.train_captions is required to keep test objects out of training")
    dataset_cfg["caption_index"] = resolve_path(base_dir, train_captions)
    dataset_cfg["skip_first_ids"] = args.validation_size
    from . import dataset_vae
    dataset_cls = getattr(dataset_vae, dataset_cfg.pop("name"), None)
    if dataset_cls is None:
        raise ValueError(f"unknown dataset class; see stage1Text2Saliency/dataset_vae.py")
    roots = resolve_path(base_dir, dataset_cfg.pop("roots", None))
    if not roots:
        raise SystemExit("dataset.roots is missing; set it in the config or pass --data-dir")
    dataset = dataset_cls(roots, **constructor_args(dataset_cfg, dataset_cls))

    models = {name: build_model(spec, device) for name, spec in cfg["models"].items()}
    for m in models.values():
        m.train()

    trainer_cfg = flatten_args(cfg["trainer"])
    trainer_name = trainer_cfg.pop("name")
    from . import trainer_vae
    trainer_cls = getattr(trainer_vae, trainer_name, None)
    if trainer_cls is None:
        raise ValueError(
            f"unknown trainer {trainer_name!r}; defined here is DualDecoderVQVAETrainer")
    max_steps = int(args.max_steps or trainer_cfg.pop("max_steps"))
    trainer_cfg.pop("max_steps", None)
    out_dir = resolve_path(base_dir, args.output_dir or cfg.get("output_dir", "outputs/saliency_vae"))

    # The base trainer takes resume as two required keyword-only arguments rather than defaulting
    # them, so they have to be passed even for a fresh run.
    trainer_cfg.setdefault("load_dir", resolve_path(base_dir, args.load_dir) if args.load_dir else None)
    trainer_cfg.setdefault("step", args.load_step)
    trainer = trainer_cls(models=models, dataset=dataset, output_dir=out_dir,
                          max_steps=max_steps, **trainer_cfg)
    trainer.run()


if __name__ == "__main__":
    main()
