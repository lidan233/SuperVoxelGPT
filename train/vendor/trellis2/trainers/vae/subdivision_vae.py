"""Subdivision-only trainer: the per-level BCE on subdivision decisions, and nothing else."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ..basic import BasicTrainer
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler

import os
import copy
import functools
from torch.utils.data import DataLoader


class SubdivisionVaeTrainer(BasicTrainer):
    """Trains on the subdivision BCE alone."""
    
    def __init__(
        self,
        *args,
        lambda_subdiv: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_subdiv = lambda_subdiv
        
    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100, batch_size=4):
        """No dataset snapshot: nothing here renders."""
        pass

    def prepare_dataloader(self, **kwargs):
        """Prepare dataloader."""
        num_workers = int(kwargs.get('num_workers', 4))
        prefetch_factor = kwargs.get('prefetch_factor', 2)
        prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)

        def _worker_init_fn(_worker_id: int) -> None:
            try:
                torch.set_num_threads(1)
            except Exception:
                pass

        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
            worker_init_fn=_worker_init_fn,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(
        self,
        vertices: sp.SparseTensor,
        intersected: sp.SparseTensor,
        cvt_points: sp.VarLenTensor,
        id: List[str] = None,
    ) -> Tuple[Dict, Dict]:
        """The subdivision BCE, summed over levels."""
        # Encoder forward
        enc_out = self.training_models["encoder"](vertices, intersected, cvt_points)
        z_q = enc_out["z_q"]
        
        # The decoder returns (subs_gt, subs) only.
        subs_gt, subs = self.training_models["decoder"](z_q, intersected)

        terms = edict(loss=0.0)


        for i, (sub_gt, sub) in enumerate(zip(subs_gt, subs)):
            terms[f"bce_sub{i}"] = F.binary_cross_entropy_with_logits(sub.feats, sub_gt.float())
            terms["loss"] = terms["loss"] + self.lambda_subdiv * terms[f"bce_sub{i}"]

        # Print the learning rate from the optimizer and BCE subdivision losses
        lr = self.optimizer.param_groups[0]['lr'] if hasattr(self, "optimizer") else None
        print(f"LR: {lr}, Debugging training losses, bce_sub0: {terms[f'bce_sub0']:.4f}, bce_sub1: {terms[f'bce_sub1']:.4f}, bce_sub2: {terms[f'bce_sub2']:.4f}, bce_sub3: {terms[f'bce_sub3']:.4f}")
        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
        **kwargs,
    ) -> Dict:
        """Records subdivision accuracy and BCE, and renders nothing."""
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
                args["cvt_points"].to(torch.float16)
            )
            z_q = enc_out["z_q"]

            # subdivision predictions
            self.models["decoder"].train()  # train mode is what returns subs_gt
            subs_gt, subs = self.models["decoder"](z_q, args["intersected"])
            
            # accuracy and BCE
            for sub_gt, sub in zip(subs_gt, subs):
                pred = (sub.feats > 0).float()
                gt = sub_gt.float()
                correct = (pred == gt).sum().item()
                total = gt.numel()
                total_correct += correct
                total_count += total
                bce = F.binary_cross_entropy_with_logits(sub.feats, gt).item()
                bce_losses.append(bce)

        self.models["encoder"].train()
        self.models["decoder"].train()

        accuracy = total_correct / total_count if total_count > 0 else 0.0
        avg_bce = sum(bce_losses) / len(bce_losses) if bce_losses else 0.0
        
        if verbose:
            print(f"[Snapshot] Subdivision Accuracy: {accuracy:.4f}, Avg BCE: {avg_bce:.4f}")

        # No samples: nothing is rendered.
        return {
            "subdivision_accuracy": {"value": torch.tensor(accuracy), "type": "scalar"},
            "subdivision_bce": {"value": torch.tensor(avg_bce), "type": "scalar"},
        }
