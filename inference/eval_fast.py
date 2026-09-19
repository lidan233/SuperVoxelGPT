#!/usr/bin/env python3
"""Generate and render ten held-out objects, and time them.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate, time and render ten objects.")
    ap.add_argument("--ckpts", default=os.path.join(_REPO, "ckpts"))
    ap.add_argument("--out", default=os.path.join(_REPO, "assets", "teaser"))
    ap.add_argument("--sheet", default=os.path.join(_REPO, "assets", "teaser_sheet.png"))
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--res", type=int, default=480)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _REPO)
    import text2shape as T

    recs = json.load(open(os.path.join(_REPO, "inference", "fast_captions.json")))
    os.makedirs(args.out, exist_ok=True)

    pipe = T.Text2Shape(args.ckpts, verbose=False)
    rows = []
    for i, r in enumerate(recs):
        t0 = time.time()
        mesh, tm = pipe(r["caption"])
        el = time.time() - t0
        dst = os.path.join(args.out, "%02d_%s.ply" % (i, r["id"][:8]))
        mesh.export(dst)
        rows.append({"id": r["id"], "tier": r["tier"], "caption": r["caption"],
                     "supervoxels": int(tm.get("supervoxels", 0)),
                     "faces": len(mesh.faces), "sec": round(el, 2)})
        print("[%2d/%d] %s  N=%5d  faces=%8d  %.2fs" %
              (i + 1, len(recs), r["id"][:12], rows[-1]["supervoxels"],
               rows[-1]["faces"], el), flush=True)

    secs = [r["sec"] for r in rows]
    print("\nmedian %.2fs   mean %.2fs   min %.2fs   max %.2fs" %
          (statistics.median(secs), statistics.mean(secs), min(secs), max(secs)))
    json.dump(rows, open(os.path.join(args.out, "timings.json"), "w"), indent=1)

    rdir = os.path.join(args.out, "renders")
    subprocess.run([sys.executable, os.path.join(_REPO, "vis", "render_mesh.py"),
                    args.out, "--out", rdir,
                    "--views", str(args.views), "--res", str(args.res)], check=True)
    _sheet_with_timings(rdir, rows, args.sheet, args.views)
    print("sheet -> %s" % args.sheet)
    return 0


def _sheet_with_timings(rdir, rows, out_png, views, pad=8, label_w=118):
    """Stack each object's views into one row, with its generation time beside it."""
    from PIL import Image, ImageDraw, ImageFont
    pngs = sorted(p for p in os.listdir(rdir) if p.endswith(".png"))
    if not pngs:
        raise SystemExit("no renders in %s" % rdir)
    strips = [pngs[i:i + views] for i in range(0, len(pngs), views)]
    first = Image.open(os.path.join(rdir, pngs[0]))
    w, h = first.size

    sheet = Image.new("RGB", (label_w + views * (w + pad) - pad,
                              len(strips) * (h + pad) - pad), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
    except OSError:
        big = ImageFont.load_default()

    for i, strip in enumerate(strips):
        y = i * (h + pad)
        for j, name in enumerate(strip):
            sheet.paste(Image.open(os.path.join(rdir, name)).convert("RGB"),
                        (label_w + j * (w + pad), y))
        sec = rows[i]["sec"] if i < len(rows) else 0.0
        draw.text((14, y + h // 2 - 16), "%.2f s" % sec, fill=(20, 20, 20), font=big)
    sheet.save(out_png)


if __name__ == "__main__":
    sys.exit(main())
