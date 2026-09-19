"""Subdivision trainer plus the intersection and vertex losses.

Adds two losses on top of the per-sub-level BCE:
  - direct/intersected: BCE on intersection logits (3 axes)
  - direct/vertice: MSE on vertex offsets

The decoder must be SuperVoxelDecoder, which returns
  (vertices_pred, intersected_pred, subs_gt, subs)
"""
from __future__ import annotations
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...utils.data_utils import recursive_to_device
from .subdivision_vae import SubdivisionVaeTrainer


class SubdivisionVaeFullTrainer(SubdivisionVaeTrainer):
    """Adds the intersected (BCE) and vertice (MSE) auxiliary losses."""

    def __init__(self, *args,
                 lambda_intersected: float = 1.0,
                 lambda_vertice: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_intersected = float(lambda_intersected)
        self.lambda_vertice = float(lambda_vertice)

    def training_losses(
        self,
        vertices: sp.SparseTensor,
        intersected: sp.SparseTensor,
        cvt_points: sp.VarLenTensor,
        id: List[str] = None,
    ) -> Tuple[Dict, Dict]:
        enc_out = self.training_models["encoder"](vertices, intersected, cvt_points)
        z_q = enc_out["z_q"]
        vertices_pred, intersected_pred, subs_gt, subs = (
            self.training_models["decoder"](z_q, intersected)
        )

        terms = edict(loss=0.0)

        # ---- subdivision BCE losses (per level), as the base trainer computes them ----
        for i, (sub_gt, sub) in enumerate(zip(subs_gt, subs)):
            terms[f"bce_sub{i}"] = F.binary_cross_entropy_with_logits(
                sub.feats, sub_gt.float()
            )
            terms["loss"] = terms["loss"] + self.lambda_subdiv * terms[f"bce_sub{i}"]

        # keep ALL trainable params in DDP graph (find_unused_parameters=False requires every param used every step)
        _dummy = sum(p.sum() for m in self.training_models.values() for p in m.parameters() if p.requires_grad)
        terms["loss"] = terms["loss"] + 0.0 * _dummy

        # ---- intersected loss (BCE on 3-axis binary) ----
        if self.lambda_intersected > 0:
            terms["direct/intersected"] = F.binary_cross_entropy_with_logits(
                intersected_pred.feats.flatten(),
                intersected.feats.flatten().float(),
            )
            terms["loss"] = terms["loss"] + self.lambda_intersected * terms["direct/intersected"]

        # ---- vertice loss (MSE on xyz offsets) ----
        if self.lambda_vertice > 0:
            terms["direct/vertice"] = F.mse_loss(
                vertices_pred.feats, vertices.feats
            )
            terms["loss"] = terms["loss"] + self.lambda_vertice * terms["direct/vertice"]

        lr = self.optimizer.param_groups[0]['lr'] if hasattr(self, "optimizer") else None
        # One entry per level the decoder actually produced. Naming a fixed four made a shallower
        # decoder fail here, in a log statement, after the losses had already been computed.
        subs_txt = ", ".join(f"bce_sub{i}: {terms[f'bce_sub{i}']:.4f}" for i in range(len(subs)))
        print(f"LR: {lr}, {subs_txt}, "
              f"intersected: {terms.get('direct/intersected', torch.tensor(0)):.4f}, "
              f"vertice: {terms.get('direct/vertice', torch.tensor(0)):.4f}")
        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
        **kwargs,
    ) -> Dict:
        """Snapshot for the full decoder, which returns a 4-tuple in train mode."""
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        total_correct = 0
        total_count = 0
        bce_losses = []
        inter_bces = []
        vert_mses = []

        self.models["encoder"].eval()
        self.models["decoder"].eval()

        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device)

            self.models["encoder"].train()
            enc_out = self.models["encoder"](
                args["vertices"].to(torch.float16),
                args["intersected"].to(torch.float16),
                args["cvt_points"].to(torch.float16),
            )
            z_q = enc_out["z_q"]

            self.models["decoder"].train()  # train mode → returns 4-tuple
            vertices_pred, intersected_pred, subs_gt, subs = self.models["decoder"](
                z_q, args["intersected"]
            )

            for sub_gt, sub in zip(subs_gt, subs):
                pred = (sub.feats > 0).float()
                gt = sub_gt.float()
                total_correct += (pred == gt).sum().item()
                total_count += gt.numel()
                bce_losses.append(
                    F.binary_cross_entropy_with_logits(sub.feats, gt).item()
                )

            inter_bces.append(
                F.binary_cross_entropy_with_logits(
                    intersected_pred.feats.flatten(),
                    args["intersected"].feats.flatten().float(),
                ).item()
            )
            vert_mses.append(
                F.mse_loss(vertices_pred.feats, args["vertices"].feats).item()
            )

        self.models["encoder"].train()
        self.models["decoder"].train()

        accuracy = total_correct / total_count if total_count > 0 else 0.0
        avg_bce = sum(bce_losses) / len(bce_losses) if bce_losses else 0.0
        avg_inter = sum(inter_bces) / len(inter_bces) if inter_bces else 0.0
        avg_vert = sum(vert_mses) / len(vert_mses) if vert_mses else 0.0

        if verbose:
            print(f"[Snapshot] Acc: {accuracy:.4f}, BCE: {avg_bce:.4f}, "
                  f"Inter: {avg_inter:.4f}, Vert: {avg_vert:.4f}")

        return {
            "subdivision_accuracy": {"value": torch.tensor(accuracy), "type": "scalar"},
            "subdivision_bce": {"value": torch.tensor(avg_bce), "type": "scalar"},
            "intersected_bce": {"value": torch.tensor(avg_inter), "type": "scalar"},
            "vertice_mse": {"value": torch.tensor(avg_vert), "type": "scalar"},
        }
