"""Guarantee the delivered mesh is never rougher than the plain watertight one.

Purpose
    Per-vertex blending still leaves whole objects where refinement went badly. For dataset
    production that is unacceptable, since nobody will inspect tens of thousands of meshes. This is
    the unconditional backstop: the worst case per object becomes "no improvement", never "ruined".

Input
    The watertight mesh and the blended mesh.

Output
    Whichever of the two is not rougher, written to a single destination.

Key idea
    Roughness is measured view-independently as the median distance from each vertex to the centroid
    of its neighbours, expressed in voxels — no rendering, no reference mesh needed. Falling back
    requires the blended mesh to be both relatively worse (beyond a ratio) and absolutely rough
    (beyond a floor), so noise-level differences never trigger a needless reversion.
"""
import argparse, numpy as np, trimesh, torch
def loadn(p,b=0.9):
    m=trimesh.load(p,process=False,force='mesh')
    if isinstance(m,trimesh.Scene): m=trimesh.util.concatenate([g for g in m.geometry.values()])
    V=np.asarray(m.vertices,np.float64);c=(V.min(0)+V.max(0))/2;V=(V-c)*(2*b/(V.max(0)-V.min(0)).max())
    return trimesh.Trimesh(V,np.asarray(m.faces),process=False)
def rough(m):
    V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int64,device=dev)
    i0,i1,i2=F[:,0],F[:,1],F[:,2];src=torch.cat([i0,i1,i1,i2,i2,i0]);dst=torch.cat([i1,i0,i2,i1,i0,i2])
    deg=torch.zeros(len(V),device=dev).index_add_(0,src,torch.ones(len(src),device=dev)).clamp(min=1)[:,None]
    nb=torch.zeros_like(V).index_add_(0,src,V[dst]);return float(np.median(((V-nb/deg).norm(dim=1)/vox).cpu().numpy()))


def main():
    global dev, vox
    dev='cuda'; vox=2/1024

    ap=argparse.ArgumentParser()
    ap.add_argument("--wt",required=True); ap.add_argument("--blend",required=True); ap.add_argument("--out",required=True)


    ap.add_argument("--margin",type=float,default=1.2); ap.add_argument("--floor",type=float,default=0.03,help="absolute roughness floor (voxels) for blended; below this no fallback (noise level)")

    a=ap.parse_args()
    rw=rough(loadn(a.wt)); rb=rough(loadn(a.blend))

    if rb > rw*a.margin and rb > a.floor:
        trimesh.load(a.wt,process=False).export(a.out); pick=f"watertight (blended {rb:.3f} > wt {rw:.3f} x {a.margin}, fell back)"
    else:
        trimesh.load(a.blend,process=False).export(a.out); pick=f"blended (blended {rb:.3f} vs wt {rw:.3f}, kept)"
    print(f"[final-guard] using {pick} -> {a.out}",flush=True)


# ---------------------------------------------------------------------------
# Production invocation:
#
#   python c_final_guard.py --wt <tag>_rfb.ply --blend <tag>_blended.ply \
#       --out <tag>_final.ply
#
# Defaults kept: margin=1.2, floor=0.03. Fallback to watertight requires BOTH
# (blended roughness > watertight x 1.2) AND (blended roughness > 0.03 voxels),
# so noise-level differences never trigger it.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
