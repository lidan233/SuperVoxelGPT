#!/usr/bin/env python3
"""Captions -> CLIP text features, the conditioning the generators train on.

Purpose
    The text branch of both stages is trained on precomputed CLIP features rather than raw
    strings, so the corpus ships features, not captions. This writes them for one object.

Input
    A JSON file of captions, and the CLIP text tower from `ckpts/`. Three shapes are accepted,
    because the corpus and the bundled example are written differently:
    a `{"captions": [...]}` object, a bare list of strings, or one record from the released
    corpus files, whose captions live under `tier_0` … `tier_8` — nine granularities per object,
    from a full sentence down to a few words.

Output
    One npz per object: `text_features` (L, 77, 768) and `text_attention_mask` (L, 77).
    The loaders accept either a 2-D single-caption array or this 3-D stack, and pick one
    granularity at random during training -- which is what makes the model accept a terse
    prompt as readily as a full sentence.

Key idea
    Tokenization and pooling must match inference exactly. `text2shape.py` pads to
    max_length=77 and takes `last_hidden_state`; anything else -- shorter padding, a pooled
    vector -- produces features the checkpoints were never trained against, and the failure
    is silent: generation still runs, and only the shapes come out wrong.
"""
import argparse, json, os, sys
import numpy as np


def clip_path(ckpts: str) -> str:
    """Local copy first; reproducibility must not depend on the network. Mirrors text2shape.py."""
    p = os.environ.get("T2S_CLIP")
    if p:
        return p
    local = os.path.join(ckpts, "clipText")
    if os.path.isdir(local) and os.path.exists(os.path.join(local, "config.json")):
        return local
    return "openai/clip-vit-large-patch14"


def _tiers(rec: dict) -> list:
    """`tier_0` … `tier_N` in numeric order — the shape the released corpus files use."""
    keys = sorted((k for k in rec if k.startswith("tier_")),
                  key=lambda k: int(k.split("_")[1]))
    return [rec[k] for k in keys]


def _read_json_captions(d, want_id, path):
    """Accept the three shapes described in this module's docstring."""
    if isinstance(d, dict):
        if "captions" in d:
            return list(d["captions"])
        if any(k.startswith("tier_") for k in d):
            return _tiers(d)
        sys.exit(f"{path}: object has neither a `captions` list nor `tier_*` keys")
    records = list(d)
    if records and isinstance(records[0], dict):
        # A whole corpus file: train19000_captions.json / test1000_captions.json.
        rec = records[0] if want_id is None else \
            next((r for r in records if r.get("id") == want_id), None)
        if rec is None:
            sys.exit(f"{path}: no record with id {want_id!r}")
        return _tiers(rec)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description="Encode captions into CLIP text features.")
    ap.add_argument("--captions", required=True,
                    help="JSON with a `captions` list, a `tier_0`..`tier_8` record from the "
                         "released corpus files, or a plain text file")
    ap.add_argument("--id", default=None,
                    help="when --captions is a whole corpus file (a list of records), the object "
                         "id to encode; defaults to the first record")
    ap.add_argument("--ckpts", required=True, help="checkpoint directory (for clipText/)")
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-length", type=int, default=77, help="must match inference")
    args = ap.parse_args(argv)

    if args.captions.endswith(".json"):
        caps = _read_json_captions(json.load(open(args.captions)), args.id, args.captions)
    else:
        caps = [l.strip() for l in open(args.captions) if l.strip()]
    if not caps:
        sys.exit("no captions found in %s" % args.captions)

    import torch
    # TF32 off before the tower is built, matching inference. Measured on the bundled example
    # the features come out byte-identical either way, so this changes nothing today; it is set
    # so that the conditioning a model trains against cannot depend on the caller's globals.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import CLIPTextModel, CLIPTokenizer
    clip = clip_path(args.ckpts)
    print("[text] tower %s" % clip, flush=True)
    tok = CLIPTokenizer.from_pretrained(clip)
    mdl = CLIPTextModel.from_pretrained(clip).to(args.device).eval()

    feats, masks = [], []
    with torch.no_grad():
        for c in caps:
            enc = tok([c], padding="max_length", truncation=True,
                      max_length=args.max_length, return_tensors="pt")
            f = mdl(input_ids=enc.input_ids.to(args.device),
                    attention_mask=enc.attention_mask.to(args.device)).last_hidden_state
            feats.append(f[0].cpu().numpy().astype(np.float32))
            masks.append(enc.attention_mask[0].numpy().astype(np.int64))

    F = np.stack(feats); M = np.stack(masks)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, text_features=F, text_attention_mask=M)
    print("[text] %d captions -> %s  features%s mask%s" % (len(caps), args.out, F.shape, M.shape))
    for i, c in enumerate(caps):
        print("   tier_%d  tokens=%d  %s" % (i, int(M[i].sum()), c[:64]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
