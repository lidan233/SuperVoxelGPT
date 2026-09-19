"""Training loop for the saliency autoencoder.
"""
import contextlib
import copy
import glob
import math
import os
import time
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler


def _dist_info() -> Tuple[int, int]:
    """(rank, world_size) for the current process, (0, 1) when not launched under torchrun."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


# ---------------------------------------------------------------------------------------------
# Small helpers, inlined so this file has no dependency on the framework it came from.
# ---------------------------------------------------------------------------------------------
def recursive_to_device(data: Any, device, non_blocking: bool = False) -> Any:
    if hasattr(data, "to"):
        return data.to(device, non_blocking=non_blocking)
    if isinstance(data, (list, tuple)):
        return type(data)(recursive_to_device(d, device, non_blocking) for d in data)
    if isinstance(data, dict):
        return {k: recursive_to_device(v, device, non_blocking) for k, v in data.items()}
    return data


def dict_reduce(dicts: List[Dict], func, special_func: Dict = {}) -> Dict:
    """Reduce a list of flat-valued dicts key by key."""
    out = {}
    for key in {k for d in dicts for k in d}:
        vals = [d[key] for d in dicts if key in d]
        if isinstance(vals[0], dict):
            out[key] = dict_reduce(vals, func, special_func)
        else:
            out[key] = special_func[key](vals) if key in special_func else func(vals)
    return out


def zero_grad(params) -> None:
    for p in params:
        if p.grad is not None:
            p.grad.detach_() if p.grad.grad_fn is not None else p.grad.requires_grad_(False)
            p.grad.zero_()


def cycle(loader: DataLoader) -> Iterator:
    """Endless iteration that keeps the sampler's position, so a resume lands mid-epoch.

    Persistent workers hold their own copy of the sampler, so bumping `epoch` here never reaches
    them and every epoch would replay the first one's order. Rebuilding the loader at the epoch
    boundary respawns them against the updated state.
    """
    while True:
        for data in loader:
            if isinstance(loader.sampler, ResumableSampler):
                loader.sampler.idx += loader.batch_size
            yield data
        if isinstance(loader.sampler, ResumableSampler):
            loader.sampler.epoch += 1
            loader.sampler.idx = 0
            if loader.persistent_workers and loader.num_workers > 0:
                loader = _rebuild_loader(loader)


def _rebuild_loader(loader: DataLoader) -> DataLoader:
    """A fresh DataLoader over the same dataset and sampler, so workers see the new epoch."""
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        persistent_workers=loader.persistent_workers,
        prefetch_factor=loader.prefetch_factor,
        sampler=loader.sampler,
        collate_fn=loader.collate_fn,
    )


class ResumableSampler(Sampler):
    """Shuffling sampler that can restart from where a checkpoint left off.

    A plain sampler restarts the epoch on resume, so the objects already seen are seen again
    before the rest. Over a long run with a large corpus that skews what the model trained on
    without anything reporting it.
    """

    def __init__(self, dataset, shuffle: bool = True, seed: int = 0,
                 rank: int = 0, world_size: int = 1):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.idx = 0
        # Same permutation on every rank, each taking its own stride: without this a multi-GPU
        # run trains on N copies of the corpus instead of on the corpus.
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        # Even split; a ragged tail deadlocks DDP's all-reduce at the epoch boundary.
        self.num_samples = len(dataset) // self.world_size
        self.total_size = self.num_samples * self.world_size

    def __iter__(self) -> Iterator:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices[self.idx:])

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> Dict[str, int]:
        return {"epoch": self.epoch, "idx": self.idx}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self.epoch, self.idx = state["epoch"], state["idx"]


class AdaptiveGradClipper:
    """Clip to a percentile of the gradient norms seen so far, rather than a fixed value.

    A fixed threshold has to be guessed before the run, and the right value changes as training
    settles. This tracks a window of recent norms and clips at a percentile of it, so the limit
    follows the run. `max_norm` still caps it, for the early steps when the window is not full.
    """

    def __init__(self, max_norm: Optional[float] = None, min_norm: Optional[float] = None,
                 clip_percentile: float = 95.0, buffer_size: int = 1000):
        self.max_norm = max_norm
        self.min_norm = min_norm
        self.clip_percentile = clip_percentile
        self.buffer_size = buffer_size
        self._grad_norm = np.zeros(buffer_size, dtype=np.float32)
        self._max_norm = max_norm
        self._ptr = 0
        self._filled = 0

    def __repr__(self) -> str:
        return f"AdaptiveGradClipper(max_norm={self.max_norm}, p={self.clip_percentile})"

    def state_dict(self) -> Dict[str, Any]:
        return {"grad_norm": self._grad_norm, "max_norm": self._max_norm,
                "buffer_ptr": self._ptr, "buffer_length": self._filled}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._grad_norm = state["grad_norm"]
        self._max_norm = state["max_norm"]
        self._ptr = state["buffer_ptr"]
        self._filled = state["buffer_length"]

    def log(self) -> Dict[str, Any]:
        return {"max_norm": self._max_norm}

    def __call__(self, parameters, norm_type: float = 2.0) -> torch.Tensor:
        limit = self._max_norm if self._max_norm is not None else float("inf")
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=limit, norm_type=norm_type)
        if torch.isfinite(norm):
            self._grad_norm[self._ptr] = norm
            self._ptr = (self._ptr + 1) % self.buffer_size
            self._filled = min(self._filled + 1, self.buffer_size)
            if self._filled == self.buffer_size:
                self._max_norm = np.percentile(self._grad_norm, self.clip_percentile)
                if self.min_norm is not None:
                    self._max_norm = max(self._max_norm, self.min_norm)
                if self.max_norm is not None:
                    self._max_norm = min(self._max_norm, self.max_norm)
        return norm


# ---------------------------------------------------------------------------------------------
class DualDecoderVQVAETrainer:
    """Train one encoder and two decoders on the saliency field.

    Args:
        models: `encoder`, `mask_decoder` and `saliency_decoder`.
        dataset: yields `ss` and `ss_saliency`, both `[B, 1, R, R, R]` in [0, 1].
        output_dir: run directory; checkpoints land in `ckpts/` under it.
        max_steps: optimizer steps to run.
        batch_size_per_gpu / batch_split: the split exists for memory, not for scheduling — the
            microbatches are accumulated into one optimizer step, so the effective batch is
            unchanged by it.
        optimizer / lr_scheduler: resolved from `torch.optim` by name.
        grad_clip: a float, or `{"name": "AdaptiveGradClipper", "args": {...}}`.
        ema_rate: one rate or several; each gets its own checkpoint series.
        fp16_mode: `amp` or None. The released run used `amp`.
        lambda_saliency: weight of the saliency term against the occupancy term.
        lambda_bg: weight of the background-toward-zero term inside the saliency term.
        finetune_ckpt: `{model_name: path}`, loaded once at start. Shape mismatches are kept at
            the model's own initialization and reported rather than failing.
        load_dir / step: resume a run. Takes precedence over `finetune_ckpt`.
    """

    def __init__(self,
                 models: Dict[str, torch.nn.Module],
                 dataset,
                 *,
                 output_dir: str,
                 max_steps: int,
                 batch_size_per_gpu: int = 12,
                 batch_split: int = 1,
                 optimizer: Dict[str, Any] = {},
                 lr_scheduler: Optional[Dict[str, Any]] = None,
                 grad_clip: Any = None,
                 ema_rate: Any = 0.9999,
                 fp16_mode: Optional[str] = "amp",
                 lambda_saliency: float = 2.0,
                 lambda_bg: float = 0.1,
                 finetune_ckpt: Optional[Dict[str, str]] = None,
                 prev_ckpts_dir: Optional[str] = None,
                 load_dir: Optional[str] = None,
                 step: int = 0,
                 i_print: int = 1000,
                 i_log: int = 500,
                 i_save: int = 10000,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 **unused):
        if unused:
            # Options the released configs carry for parts of the upstream framework this trainer
            # does not implement. Named rather than ignored: silently dropping a training option
            # is how a run ends up not doing what its config says.
            print(f"[saliency-vae] ignoring unsupported options: {sorted(unused)}", flush=True)
        if fp16_mode not in ("amp", None):
            raise ValueError(f"fp16_mode must be 'amp' or None, not {fp16_mode!r}")

        self.models = models
        self.dataset = dataset
        self.output_dir = output_dir
        self.max_steps = int(max_steps)
        self.batch_size_per_gpu = int(batch_size_per_gpu)
        self.batch_split = int(batch_split)
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else list(ema_rate)
        self.fp16_mode = fp16_mode
        self.lambda_saliency = float(lambda_saliency)
        self.lambda_bg = float(lambda_bg)
        self.i_print, self.i_log, self.i_save = i_print, i_log, i_save
        self.num_workers, self.prefetch_factor = num_workers, prefetch_factor
        self.step = 0
        self._prefetched = None

        self.rank, self.world_size = _dist_info()
        self.is_master = self.rank == 0
        if self.world_size > 1:
            local = int(os.environ.get("LOCAL_RANK", self.rank))
            torch.cuda.set_device(local)
            self.device = torch.device("cuda", local)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # `model_params` below stays on the unwrapped modules so the optimizer, EMA copies and
        # checkpoint layout are identical on one GPU and on many; DDP shares those same tensors.
        self.training_models = models
        if self.world_size > 1:
            self.training_models = {
                name: DistributedDataParallel(m, device_ids=[self.device.index],
                                              output_device=self.device.index)
                for name, m in models.items()}

        self.model_params = [p for m in models.values() for p in m.parameters() if p.requires_grad]
        self.scaler = torch.amp.GradScaler() if fp16_mode == "amp" else None
        self.ema_params = [[p.detach().clone() for p in self.model_params] for _ in self.ema_rate]

        opt_name = optimizer.get("name", "AdamW")
        self.optimizer = getattr(torch.optim, opt_name)(self.model_params,
                                                        **optimizer.get("args", {}))
        self.lr_scheduler = None
        if lr_scheduler is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler["name"])(
                self.optimizer, **lr_scheduler.get("args", {}))

        if isinstance(grad_clip, (int, float)) or grad_clip is None:
            self.grad_clip = float(grad_clip) if grad_clip is not None else None
        elif grad_clip["name"] == "AdaptiveGradClipper":
            self.grad_clip = AdaptiveGradClipper(**grad_clip.get("args", {}))
        else:
            raise ValueError(f"unknown gradient clipper {grad_clip['name']!r}")

        self._prepare_dataloader()
        os.makedirs(os.path.join(output_dir, "ckpts"), exist_ok=True)

        if load_dir is not None and step:
            self.load(load_dir, step)
        elif finetune_ckpt:
            self.finetune_from(finetune_ckpt)
        elif prev_ckpts_dir:
            self._load_latest_from(prev_ckpts_dir)

        print(f"\n[saliency-vae] {len(self.dataset)} fields | batch {self.batch_size_per_gpu}"
              f" x {self.batch_split} split | {sum(p.numel() for p in self.model_params) / 1e6:.1f}M"
              f" trainable | fp16={self.fp16_mode} | clip={self.grad_clip}", flush=True)

    # -- data ----------------------------------------------------------------------------------
    def _prepare_dataloader(self) -> None:
        self.data_sampler = ResumableSampler(self.dataset, shuffle=True,
                                             rank=self.rank, world_size=self.world_size)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def load_data(self) -> List[Dict[str, torch.Tensor]]:
        """One batch, moved to the device and split into microbatches."""
        if self._prefetched is None:
            self._prefetched = recursive_to_device(next(self.data_iterator), self.device, True)
        data = self._prefetched
        self._prefetched = recursive_to_device(next(self.data_iterator), self.device, True)

        if self.batch_split == 1:
            return [data]
        tensors = {k: v for k, v in data.items() if torch.is_tensor(v)}
        n = next(iter(tensors.values())).shape[0]
        size = n // self.batch_split
        return [{k: v[i * size:(i + 1) * size] for k, v in tensors.items()}
                for i in range(self.batch_split)]

    # -- loss ----------------------------------------------------------------------------------
    def training_losses(self, ss: torch.Tensor, ss_saliency: torch.Tensor,
                        **kwargs) -> Tuple[Dict, Dict]:
        enc_out = self.training_models["encoder"](ss.float())
        z_q, indices = enc_out["z_q"], enc_out["indices"]

        logits = self.training_models["mask_decoder"](z_q)
        logits = logits[0] if isinstance(logits, tuple) else logits
        saliency = self.training_models["saliency_decoder"](z_q)
        saliency = saliency[0] if isinstance(saliency, tuple) else saliency

        target_saliency = ss_saliency
        target_mask = (target_saliency >= 0.5).float()

        # Occupancy: BCE with a soft Dice term. BCE alone is dominated by the empty majority of a
        # 64-cube, where predicting nothing anywhere already scores well.
        bce = F.binary_cross_entropy_with_logits(logits, target_mask)
        prob = torch.sigmoid(logits)
        b = prob.shape[0]
        pf, tf = prob.view(b, -1), target_mask.view(b, -1)
        eps = 1e-6
        soft_dice = ((2 * (pf * tf).sum(1) + eps) / (pf.sum(1) + tf.sum(1) + eps)).mean()
        soft_dice_loss = 1 - soft_dice
        with torch.no_grad():   # the same score at the decision threshold, for reading the log
            hf = (logits > 0).float().view(b, -1)
            hard_dice = ((2 * (hf * tf).sum(1) + eps) / (hf.sum(1) + tf.sum(1) + eps)).mean()
        classification_loss = bce + soft_dice_loss

        # Saliency: L1 where the object is, and a pull toward zero where it is not.
        fg, bg = target_mask.bool(), ~target_mask.bool()
        fg_loss = (F.l1_loss(saliency[fg], target_saliency[fg]) if fg.any()
                   else saliency.new_tensor(0.0))
        bg_loss = (F.mse_loss(saliency[bg], torch.zeros_like(saliency[bg])) if bg.any()
                   else saliency.new_tensor(0.0))
        saliency_loss = fg_loss + self.lambda_bg * bg_loss

        total = classification_loss + self.lambda_saliency * saliency_loss

        with torch.no_grad():
            enc = self.models["encoder"]
            size = getattr(enc, "codebook_size", 5625)
            q = int(getattr(enc, "num_quantizers", 1))
            if q > 1:
                usage = sum(indices[..., i].unique().numel() / size for i in range(q)) / q
            else:
                usage = indices.unique().numel() / size

        terms = {"loss": total, "recon_loss": total,
                 "classification_loss": classification_loss, "bce_loss": bce,
                 "soft_dice_loss": soft_dice_loss, "soft_dice": soft_dice, "hard_dice": hard_dice,
                 "saliency_loss": saliency_loss, "fg_saliency_loss": fg_loss,
                 "bg_saliency_loss": bg_loss, "codebook_usage": usage}
        return terms, {"lr": self.optimizer.param_groups[0]["lr"]}

    # -- one step ------------------------------------------------------------------------------
    def run_step(self, data_list: List[Dict[str, torch.Tensor]]) -> Dict[str, Dict]:
        amp = (partial(torch.autocast, device_type="cuda") if self.fp16_mode == "amp"
               else contextlib.nullcontext)
        losses, statuses = [], []
        zero_grad(self.model_params)

        for mb in data_list:
            with amp():
                loss, status = self.training_losses(**mb)
                scaled = loss["loss"] / len(data_list)
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            to_float = lambda d: {k: (v.item() if torch.is_tensor(v) else v) for k, v in d.items()}
            losses.append(to_float(loss))
            statuses.append(to_float(status))

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if isinstance(self.grad_clip, float):
            norm = torch.nn.utils.clip_grad_norm_(self.model_params, self.grad_clip)
        elif self.grad_clip is not None:
            norm = self.grad_clip(self.model_params)
        else:
            norm = torch.nn.utils.clip_grad_norm_(self.model_params, float("inf"))
        if torch.isfinite(norm):
            statuses[-1]["grad_norm"] = norm.item()

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif not any(p.grad is not None and not p.grad.isfinite().all() for p in self.model_params):
            self.optimizer.step()
        else:
            print("Warning: non-finite gradients; skipping this update.", flush=True)

        if self.lr_scheduler is not None:
            statuses[-1]["lr"] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        step_log = {"loss": dict_reduce(losses, np.mean),
                    "status": dict_reduce(statuses, np.mean,
                                          special_func={"grad_norm": np.max})}
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            step_log["grad_clip"] = self.grad_clip.log()
        self.update_ema()
        return step_log

    def update_ema(self) -> None:
        with torch.no_grad():
            for rate, params in zip(self.ema_rate, self.ema_params):
                for ema, live in zip(params, self.model_params):
                    ema.mul_(rate).add_(live.detach(), alpha=1.0 - rate)

    # -- checkpoints ---------------------------------------------------------------------------
    def _state_dicts(self, params) -> Dict[str, Dict[str, torch.Tensor]]:
        """Lay a flat parameter list back out as one state dict per model."""
        states = {name: model.state_dict() for name, model in self.models.items()}
        names = [(mname, pname) for mname, model in self.models.items()
                 for pname, p in model.named_parameters() if p.requires_grad]
        for i, (mname, pname) in enumerate(names):
            states[mname][pname] = params[i]
        return states

    def _load_into_params(self, params, states: Dict[str, Dict[str, torch.Tensor]]) -> None:
        names = [(mname, pname) for mname, model in self.models.items()
                 for pname, p in model.named_parameters() if p.requires_grad]
        for i, (mname, pname) in enumerate(names):
            params[i].data.copy_(states[mname][pname].data)

    def save(self) -> None:
        """Write the checkpoint set. The names are the contract the inference path reads.

        Rank 0 only; the others wait, so none runs ahead of a half-written checkpoint.
        """
        if not self.is_master:
            if self.world_size > 1:
                dist.barrier()
            return
        ckpts = os.path.join(self.output_dir, "ckpts")
        for name, state in self._state_dicts(self.model_params).items():
            torch.save(state, os.path.join(ckpts, f"{name}_step{self.step:07d}.pt"))
        for rate, params in zip(self.ema_rate, self.ema_params):
            for name, state in self._state_dicts(params).items():
                torch.save(state, os.path.join(ckpts, f"{name}_ema{rate}_step{self.step:07d}.pt"))

        misc = {"optimizer": self.optimizer.state_dict(), "step": self.step,
                "data_sampler": self.data_sampler.state_dict()}
        if self.scaler is not None:
            misc["scaler"] = self.scaler.state_dict()
        if self.lr_scheduler is not None:
            misc["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc["grad_clip"] = self.grad_clip.state_dict()
        torch.save(misc, os.path.join(ckpts, f"misc_step{self.step:07d}.pt"))
        print(f"  saved step {self.step}", flush=True)
        if self.world_size > 1:
            dist.barrier()

    def load(self, load_dir: str, step: int = 0) -> None:
        ckpts = os.path.join(load_dir, "ckpts")
        states = {}
        for name, model in self.models.items():
            state = torch.load(os.path.join(ckpts, f"{name}_step{step:07d}.pt"),
                               map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            states[name] = state
        self._load_into_params(self.model_params, states)

        for i, rate in enumerate(self.ema_rate):
            ema = {name: torch.load(os.path.join(ckpts, f"{name}_ema{rate}_step{step:07d}.pt"),
                                    map_location=self.device, weights_only=True)
                   for name in self.models}
            self._load_into_params(self.ema_params[i], ema)

        misc = torch.load(os.path.join(ckpts, f"misc_step{step:07d}.pt"),
                          map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(misc["optimizer"])
        # The config's learning rate wins over the one in the checkpoint: resuming at a rate the
        # config no longer names is a change nobody asked for and nothing reports.
        lr = self.optimizer_config.get("args", {}).get("lr")
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] = lr
        self.step = misc["step"]
        self.data_sampler.load_state_dict(misc["data_sampler"])
        if self.scaler is not None and "scaler" in misc:
            self.scaler.load_state_dict(misc["scaler"])
        if self.lr_scheduler is not None and "lr_scheduler" in misc:
            self.lr_scheduler.load_state_dict(misc["lr_scheduler"])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float) \
                and "grad_clip" in misc:
            self.grad_clip.load_state_dict(misc["grad_clip"])
        print(f"[saliency-vae] resumed from step {self.step}", flush=True)

    def finetune_from(self, finetune_ckpt: Dict[str, str]) -> None:
        """Start from named weights. A shape mismatch keeps the model's own initialization."""
        states = {}
        for name, model in self.models.items():
            own = model.state_dict()
            if name not in finetune_ckpt:
                print(f"  {name}: not in finetune_ckpt, left at initialization", flush=True)
                states[name] = own
                continue
            state = torch.load(finetune_ckpt[name], map_location=self.device, weights_only=True)
            kept = 0
            for k in list(state):
                if k in own and state[k].shape == own[k].shape:
                    kept += 1
                else:
                    state[k] = own[k] if k in own else state[k]
            model.load_state_dict(state, strict=False)
            states[name] = model.state_dict()
            print(f"  {name}: {kept}/{len(own)} tensors from {finetune_ckpt[name]}", flush=True)
        self._load_into_params(self.model_params, states)

    def _load_latest_from(self, run_dir: str) -> None:
        """Warm start from the newest step in an earlier run directory."""
        ckpts = os.path.join(run_dir, "ckpts")
        misc = glob.glob(os.path.join(ckpts, "misc_step*.pt"))
        if not misc:
            print(f"[saliency-vae] no checkpoints under {ckpts}; starting fresh", flush=True)
            return
        step = max(int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in misc)
        found = {name: os.path.join(ckpts, f"{name}_step{step:07d}.pt") for name in self.models}
        self.finetune_from({k: v for k, v in found.items() if os.path.exists(v)})

    # -- loop ----------------------------------------------------------------------------------
    def run(self) -> None:
        elapsed = 0.0
        last_print = 0.0
        while self.step < self.max_steps:
            t0 = time.time()
            step_log = self.run_step(self.load_data())
            elapsed += time.time() - t0
            self.step += 1

            if self.step % self.i_log == 0:
                loss = step_log["loss"]
                print(f"step {self.step}  loss {loss['loss']:.4f}"
                      f"  cls {loss['classification_loss']:.4f}"
                      f"  sal {loss['saliency_loss']:.4f}"
                      f"  dice {loss['hard_dice']:.4f}"
                      f"  codebook {loss['codebook_usage']:.3f}"
                      f"  lr {step_log['status'].get('lr', float('nan')):.2e}", flush=True)

            if self.step % self.i_print == 0:
                speed = self.i_print / max(elapsed - last_print, 1e-9) * 3600
                left = (self.max_steps - self.step) / max(speed, 1e-9)
                print(f"  {self.step}/{self.max_steps}"
                      f"  ({self.step / self.max_steps * 100:.1f}%)"
                      f"  {elapsed / 3600:.2f} h elapsed"
                      f"  {speed:.0f} steps/h  ETA {left:.1f} h", flush=True)
                last_print = elapsed

            if self.step % self.i_save == 0:
                self.save()

        self.save()
        print("[saliency-vae] done", flush=True)
