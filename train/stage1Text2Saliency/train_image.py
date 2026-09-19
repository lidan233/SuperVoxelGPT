#!/usr/bin/env python3
"""Train the saliency generator on renders — image2saliency.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments

from ..dataset import image2saliency as image_ds
from ..utils.config import add_config_flag, parse_with_config
from ..utils.conditioning import IMAGE_DINOV2
from ..utils.splits import training_and_validation_ids
from .model import Text3DModelMaskGITSingleCached


class SaliencyTrainer(Trainer):
    """Passes the batch straight through; the model returns its own loss and accuracy."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # The dataset speaks in modality-neutral names because the condition may be text or
        # renders; the model keeps the parameter names its released checkpoints were trained under.
        out = model(text_features=inputs["cond_features"],
                    text_attention_mask=inputs["cond_mask"],
                    token_3d_ids=inputs["token_ids"],
                    token_3d_coords=inputs["token_coords"],
                    token_3d_attention_mask=inputs.get("token_mask"))
        self._last_metrics = {"accuracy": float(out["accuracy"]),
                              "mask_ratio": float(out["mask_ratio"])}
        return (out["loss"], out) if return_outputs else out["loss"]


class MetricCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        extra = getattr(self.trainer, "_last_metrics", None)
        if logs is not None and extra:
            logs.update(extra)


class HighMaskRampCallback(TrainerCallback):
    """Advance the model's high-mask curriculum in step with training progress.

    The mask ratio the model practises is not a fixed property of the data, it is a curriculum:
    reconstructing a field from nothing is only learnable once the mostly-visible case is in hand.
    This moves that share from 0 to its target over the run.
    """

    def __init__(self, model):
        self.model = model

    def on_step_begin(self, args, state, control, **kwargs):
        if state.max_steps:
            self.model.set_training_progress(state.global_step / state.max_steps)


def build_dataset(args):
    train_ids, val_ids = training_and_validation_ids(
        args.train_split, args.test_split, args.validation_size, args.save_path)
    ds = image_ds.ImageToSaliencyDataset(
        code_dir=args.code_dir, condition_dir=args.condition_dir,
        global_mean_path=args.global_mean, num_views=args.num_views, grid=args.grid_size,
        code_key=args.code_key, max_samples=args.max_samples,
        repeat_factor=args.repeat_factor, train=True, allowed_ids=train_ids)
    eval_ds = None if not val_ids else image_ds.ImageToSaliencyDataset(
        code_dir=args.code_dir, condition_dir=args.condition_dir,
        global_mean_path=args.global_mean, num_views=args.num_views, grid=args.grid_size,
        code_key=args.code_key, repeat_factor=1, train=False, allowed_ids=val_ids)
    if eval_ds is not None and len(eval_ds) != args.validation_size:
        raise RuntimeError(f"only {len(eval_ds)}/{args.validation_size} validation objects have data")
    return ds, eval_ds, image_ds.collate


def planned_steps(n_items, epochs, world_batch, grad_accum):
    per_epoch = max(1, n_items // max(1, world_batch * grad_accum))
    return per_epoch * epochs


def main():
    ap = argparse.ArgumentParser(description="Train the saliency generator on renders (image2saliency).")
    add_config_flag(ap)
    ap.add_argument("--code-dir", required=True, help="per-object saliency codes")
    ap.add_argument("--condition-dir", required=True, help="precomputed view features")
    ap.add_argument("--global-mean", required=True,
                    help="training-split feature mean, frozen for validation/test. Not optional: "
                         "vision-tower patch features carry a "
                         "large component identical for every object, and left in it dominates the "
                         "projection's input scale")
    ap.add_argument("--num-views", type=int, default=1,
                    help="views drawn per step. Which views are drawn changes every epoch, so this "
                         "is an augmentation as much as a budget")
    ap.add_argument("--code-key", default="indices_flat", help="npz key holding the code")
    ap.add_argument("--grid-size", type=int, default=8, help="lattice edge; the code has grid^3 entries")
    ap.add_argument("--max-samples", type=int, default=None)
    split_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "data_prep", "3_train_test_data")
    ap.add_argument("--train-split", default=os.path.join(split_dir, "train19000_captions.json"))
    ap.add_argument("--test-split", default=os.path.join(split_dir, "test1000_captions.json"))
    ap.add_argument("--validation-size", type=int, default=100)
    ap.add_argument("--repeat-factor", type=int, default=1,
                    help="repeat the corpus this many times per epoch. It multiplies into the "
                         "step count, so raising it means lowering --max-epochs by the same factor")
    # Architecture. Defaults are the trained configuration; changing any of them makes existing
    # checkpoints unloadable.
    ap.add_argument("--num-tokens", type=int, default=5625, help="codebook size of the saliency autoencoder")
    ap.add_argument("--embed-dim", type=int, default=768)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--coord-num-freq-bands", type=int, default=64)
    ap.add_argument("--coord-max-freq", type=float, default=32.0)
    ap.add_argument("--coord-num-layers", type=int, default=3)
    ap.add_argument("--high-mask-prob", type=float, default=0.0,
                    help="probability of drawing a near-fully-masked step; the calibration finish "
                         "used 0.6, and training past that point degrades generation")
    # Schedule
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--pretrained-path", default=None, help="weights to warm-start from")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="stop after this many optimizer steps; -1 runs the full epochs. "
                         "Set it small to smoke-test the pipeline before committing a run")
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--micro-batch-size", type=int, default=32)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=3000)
    ap.add_argument("--eval-steps", type=int, default=3000)
    ap.add_argument("--keep-checkpoints", type=int, default=12,
                    help="checkpoints retained; the best one is often not the last")
    ap.add_argument("--max-plausible-steps", type=int, default=2_000_000,
                    help="refuse to start above this; a schedule defined against an unreachable "
                         "total decays so slowly that the rate never actually falls")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    args = parse_with_config(ap)

    spec = IMAGE_DINOV2

    train_ds, eval_ds, collate_fn = build_dataset(args)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    total = planned_steps(len(train_ds), args.max_epochs,
                          args.micro_batch_size * world, args.gradient_accumulation_steps)
    print(f"[saliency] planned optimizer steps: {total:,} "
          f"({len(train_ds):,} items, {args.max_epochs} epochs, "
          f"{args.micro_batch_size}x{world}x{args.gradient_accumulation_steps} per step)")
    if total > args.max_plausible_steps:
        raise SystemExit(
            f"refusing to start: {total:,} planned steps exceeds --max-plausible-steps "
            f"({args.max_plausible_steps:,}). The decay is defined against this total, so the "
            f"learning rate would stay effectively constant for the whole run. Lower "
            f"--max-epochs or --repeat-factor.")

    model = Text3DModelMaskGITSingleCached(
        text_hidden_dim=spec.feature_dim, num_tokens=args.num_tokens,
        embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        coord_num_freq_bands=args.coord_num_freq_bands,
        coord_max_freq=args.coord_max_freq, coord_num_layers=args.coord_num_layers,
        high_mask_prob=args.high_mask_prob)

    if args.pretrained_path:
        sd = torch.load(args.pretrained_path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[saliency] warm start from {args.pretrained_path}: "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if any("cond_proj" in k for k in missing):
            print("[saliency] cond_proj is freshly initialized — expected when switching modality, "
                  "since the projection is the one layer whose input dimension changed")

    targs = TrainingArguments(
        output_dir=args.save_path,
        num_train_epochs=args.max_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        label_names=["token_ids"],
        save_total_limit=args.keep_checkpoints,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = SaliencyTrainer(model=model, args=targs, train_dataset=train_ds,
                            eval_dataset=eval_ds,
                            data_collator=collate_fn)
    trainer.add_callback(MetricCallback(trainer))
    if args.high_mask_prob > 0:
        trainer.add_callback(HighMaskRampCallback(model))
    trainer.train()
    trainer.save_model(os.path.join(args.save_path, "final"))
    print("[saliency] done")


if __name__ == "__main__":
    main()
