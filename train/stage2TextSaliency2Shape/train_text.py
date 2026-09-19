#!/usr/bin/env python3
"""Train the shape generator on captions and supervoxel layouts — textcvt2shape.

Purpose
    Turns (caption, supervoxel centers) into the tokens that fill those centers. This is the lineage
    the architecture was tuned on, and the reference the image-conditioned entry beside it inherits
    from: the two share a model and differ only in which dataset they build and how wide one
    projection is.

Input
    A directory of per-object token files carrying both the shape tokens and the centers, and a
    directory of precomputed caption features.

Output
    Checkpoints and a training log.
"""
import argparse
import os

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from ..dataset import textcvt2shape as text_ds
from ..utils.config import add_config_flag, parse_with_config
from ..utils.conditioning import TEXT_CLIP
from ..utils.splits import training_and_validation_ids
from .model import Text3DQwenAR


class ShapeTrainer(Trainer):
    """Passes the batch straight through; the model returns its own loss and accuracy."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # The dataset speaks in modality-neutral names because the condition may be text or
        # renders; the model keeps the parameter names its released checkpoints were trained under.
        out = model(text_features=inputs["cond_features"],
                    text_attention_mask=inputs["cond_mask"],
                    token_3d_ids=inputs["token_ids"],
                    token_3d_coords=inputs["token_coords"],
                    token_3d_attention_mask=inputs.get("token_mask"))
        self._last_metrics = {"accuracy": float(out["accuracy"])}
        return (out["loss"], out) if return_outputs else out["loss"]


class ConsistencyTrainer(ShapeTrainer):
    """Train the model to decode in parallel, not just to predict the next token.

    Ordinary teacher forcing always shows the model a correct prefix, which is a context it never
    sees at inference: parallel decoding starts from a guess and iterates, so every step but the
    last reads tokens that are still wrong. A model trained only on correct prefixes therefore
    needs many iterations to reach its fixed point, and on long fields it may not reach the same
    one at all.

    So on a fraction `q` of the steps, the input prefix is replaced by the model's *own*
    unconverged draft — a few Jacobi iterations, deliberately stopped early — while the labels stay
    the ground truth. The model learns to produce the right answer from a wrong context, which is
    exactly the job parallel decoding gives it.
    """

    def __init__(self, *args, cllm_q: float = 0.0, cllm_iters: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.cllm_q = float(cllm_q)
        self.cllm_iters = int(cllm_iters)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not (self.cllm_q > 0 and model.training and torch.rand(1).item() < self.cllm_q):
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        core = model.module if hasattr(model, "module") else model
        # bf16 is not optional here: the draft runs through the same flash-attention path as
        # training, and fp32 activations make it refuse to run at all.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            draft = core.jacobi_generate(
                text_features=inputs["cond_features"],
                text_attention_mask=inputs["cond_mask"],
                coords=inputs["token_coords"],
                token_3d_attention_mask=inputs.get("token_mask"),
                max_iters=self.cllm_iters)
        out = model(text_features=inputs["cond_features"],
                    text_attention_mask=inputs["cond_mask"],
                    token_3d_ids=inputs["token_ids"],
                    token_3d_coords=inputs["token_coords"],
                    token_3d_attention_mask=inputs.get("token_mask"),
                    input_3d_ids=draft)          # the draft replaces the teacher-forced prefix
        self._last_metrics = {"accuracy": float(out["accuracy"]), "cllm": 1.0}
        return (out["loss"], out) if return_outputs else out["loss"]


class MetricCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        extra = getattr(self.trainer, "_last_metrics", None)
        if logs is not None and extra:
            logs.update(extra)


def build_dataset(args):
    train_ids, val_ids = training_and_validation_ids(
        args.train_split, args.test_split, args.validation_size, args.save_path)
    ds = text_ds.TextCvtToShapeDataset(
        token_dir=args.token_dir, condition_dir=args.condition_dir,
        max_tokens=args.max_tokens, jitter_vox=args.jitter_vox,
        max_samples=args.max_samples, grid=args.grid,
        repeat_factor=args.repeat_factor, train=True, allowed_ids=train_ids)
    eval_ds = None if not val_ids else text_ds.TextCvtToShapeDataset(
        token_dir=args.token_dir, condition_dir=args.condition_dir,
        max_tokens=args.max_tokens, jitter_vox=0.0, grid=args.grid,
        repeat_factor=1, train=False, allowed_ids=val_ids)
    if eval_ds is not None and len(eval_ds) != args.validation_size:
        raise RuntimeError(f"only {len(eval_ds)}/{args.validation_size} validation objects have data")
    return ds, eval_ds, text_ds.collate


def planned_steps(n_items, epochs, world_batch, grad_accum):
    per_epoch = max(1, n_items // max(1, world_batch * grad_accum))
    return per_epoch * epochs


def main():
    ap = argparse.ArgumentParser(description="Train the shape generator on captions (textcvt2shape).")
    add_config_flag(ap)
    ap.add_argument("--token-dir", required=True, help="per-object shape tokens and supervoxel centers")
    ap.add_argument("--condition-dir", required=True, help="precomputed caption features")
    ap.add_argument("--max-tokens", type=int, default=None, help="truncate objects longer than this")
    ap.add_argument("--jitter-vox", type=float, default=0.3,
                    help="coordinate jitter in voxels. Training only: extraction, evaluation and "
                         "inference must leave it at zero, or the token order stops being "
                         "reproducible")
    ap.add_argument("--grid", type=int, default=64, help="grid the Morton key is quantized against")
    ap.add_argument("--max-samples", type=int, default=None)
    split_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "data_prep", "3_train_test_data")
    ap.add_argument("--train-split", default=os.path.join(split_dir, "train19000_captions.json"))
    ap.add_argument("--test-split", default=os.path.join(split_dir, "test1000_captions.json"))
    ap.add_argument("--validation-size", type=int, default=100)
    ap.add_argument("--repeat-factor", type=int, default=1,
                    help="repeat the corpus this many times per epoch. It amortizes dataloader "
                         "startup over a small corpus, and it is what lets the bundled "
                         "single-object dataset fill an epoch without copies on disk")
    # Architecture. Defaults are the trained configuration; changing any of them makes existing
    # checkpoints unloadable.
    ap.add_argument("--vocab", type=int, default=10125, help="codebook size of the supervoxel autoencoder")
    ap.add_argument("--num-layers", type=int, default=12,
                    help="12 reproduces the released generator; other values cannot load into it")
    ap.add_argument("--hidden", type=int, default=896)
    ap.add_argument("--num-heads", type=int, default=14)
    ap.add_argument("--num-kv-heads", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=4864)
    ap.add_argument("--max-pos", type=int, default=4096)
    ap.add_argument("--coord-bands", type=int, default=16)
    ap.add_argument("--attn-impl", default="flash_attention_2")
    ap.add_argument("--cllm-q", type=float, default=0.0,
                    help="fraction of steps trained on the model's own unconverged draft "
                         "(consistency training); 0 disables it")
    ap.add_argument("--cllm-iters", type=int, default=8,
                    help="Jacobi iterations for that draft — deliberately short of convergence")
    ap.add_argument("--input-noise", type=float, default=0.1,
                    help="fraction of input tokens replaced at random, so the model can recover "
                         "from the wrong prefixes parallel decoding will feed it")
    # Schedule
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--pretrained-path", default=None, help="weights to warm-start from")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="stop after this many optimizer steps; -1 runs the full epochs. "
                         "Set it small to smoke-test the pipeline before committing a run")
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--micro-batch-size", type=int, default=4)
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
                    help="refuse to start above this; the decay is defined against the planned "
                         "total, so an unreachable total leaves the rate effectively constant")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    args = parse_with_config(ap)

    spec = TEXT_CLIP
    train_ds, eval_ds, collate_fn = build_dataset(args)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    total = planned_steps(len(train_ds), args.max_epochs,
                          args.micro_batch_size * world, args.gradient_accumulation_steps)
    print(f"[stage2TextSaliency2Shape] planned optimizer steps: {total:,} "
          f"({len(train_ds):,} items, {args.max_epochs} epochs)")
    if total > args.max_plausible_steps:
        raise SystemExit(f"refusing to start: {total:,} planned steps exceeds "
                         f"--max-plausible-steps ({args.max_plausible_steps:,}); the learning rate "
                         f"would stay effectively constant for the whole run")

    # flash-attention has no fp32 kernel: without a half-precision flag it fails inside the first
    # forward pass, several frames deep, with an error that does not mention precision. Catch it here.
    if args.attn_impl.startswith("flash") and not (args.bf16 or getattr(args, "fp16", False)):
        raise SystemExit(
            f"--attn-impl {args.attn_impl} requires --bf16 (or --fp16): flash-attention has no "
            f"fp32 kernel. Either pass --bf16, or use --attn-impl eager for a fp32 run.")

    # The model reads the noise fraction from the environment, so that a forward pass behaves the
    # same whether it was reached from here or from a notebook. Set it before the first step.
    os.environ["MLLM_INPUT_NOISE"] = str(args.input_noise)

    model = Text3DQwenAR(
        text_hidden_dim=spec.feature_dim, num_layers=args.num_layers, hidden=args.hidden,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, ffn=args.ffn,
        max_pos=args.max_pos, attn_impl=args.attn_impl)

    if args.pretrained_path:
        sd = torch.load(args.pretrained_path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[stage2TextSaliency2Shape] warm start: {len(missing)} missing, {len(unexpected)} unexpected")

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
    extra = dict(cllm_q=args.cllm_q, cllm_iters=args.cllm_iters) if args.cllm_q > 0 else {}
    trainer_cls = ConsistencyTrainer if args.cllm_q > 0 else ShapeTrainer
    trainer = trainer_cls(model=model, args=targs, train_dataset=train_ds,
                          eval_dataset=eval_ds,
                          data_collator=collate_fn, **extra)
    if args.cllm_q > 0:
        print(f"[stage2TextSaliency2Shape] consistency training on {args.cllm_q:.0%} of steps, "
              f"{args.cllm_iters} Jacobi iterations per draft")
    trainer.add_callback(MetricCallback(trainer))
    trainer.train()
    trainer.save_model(os.path.join(args.save_path, "final"))
    print("[stage2TextSaliency2Shape] done")


if __name__ == "__main__":
    main()
