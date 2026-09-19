"""Keep sharpening only where it actually improved, vertex by vertex.

Purpose
    A rendering-driven refinement improves most of a surface and drifts on the rest. Accepting it
    wholesale trades wrinkles for creases; rejecting it wholesale throws away real detail. This
    picks per vertex.

Input
    Two meshes sharing topology — the blunt watertight one and its sharpened version — plus the
    original reference mesh they should both resemble.

Output
    One mesh, each vertex taken from whichever source agrees better with the reference.

    Still per-vertex, so an object where refinement went badly everywhere can come out worse
    overall. The whole-mesh guard in the next step is what catches that case.

Key idea
    Fidelity is measured against interpolated reference normals at each vertex, not against
    positions. Wrinkles are exactly the failure mode where normals scatter while positions barely
    move, so a normal-based criterion separates them from genuine detail, which hugs the reference
    normals. A vertex switches to the sharpened position only when it beats the watertight one by a
    margin, and the resulting mask is diffused over the mesh so the two sources never meet at a
    visible seam.
"""
import argparse, numpy as np, torch, trimesh, cubvh
def load_norm(p,bound=0.9):
    m=trimesh.load(p,process=False,force='mesh')
    if isinstance(m,trimesh.Scene): m=trimesh.util.concatenate([g for g in m.geometry.values() if isinstance(g,trimesh.Trimesh) and len(g.faces)])
    v=np.asarray(m.vertices,np.float64); c=(v.min(0)+v.max(0))/2; v=(v-c)*(2*bound/(v.max(0)-v.min(0)).max())
    return v.astype(np.float32), np.asarray(m.faces)
def vnorm(V,F):
    i0,i1,i2=F[:,0],F[:,1],F[:,2]; fn=torch.cross(V[i1]-V[i0],V[i2]-V[i0],dim=-1)
    vn=torch.zeros_like(V); idx=torch.cat([i0,i1,i2]); vn.scatter_add_(0,idx[:,None].expand(-1,3),fn.repeat(3,1))
    return torch.nn.functional.normalize(vn,dim=-1)


def main():
    global i0, i1, i2
    dev='cuda'
    ap=argparse.ArgumentParser()
    ap.add_argument("--before",required=True); ap.add_argument("--after",required=True)

    ap.add_argument("--ref",required=True); ap.add_argument("--out",required=True)

    ap.add_argument("--margin",type=float,default=0.002,help="the refined normal must beat watertight by this much to count as real sharpening")
    ap.add_argument("--smooth",type=int,default=5,help="mask diffusion steps over the mesh (prevents seams)")
    a=ap.parse_args()
    # GT
    gv,gf=load_norm(a.ref); GV=torch.tensor(gv,device=dev); GF=torch.tensor(gf,dtype=torch.int64,device=dev)


    bvh=cubvh.cuBVH(GV,GF.int()); gtvn=vnorm(GV,GF)

    # before (watertight) / after (refined), same topology
    bv,bf=load_norm(a.before); av,af=load_norm(a.after)

    assert len(bv)==len(av), f"before/after must have the same vertex count (holds when SDF refine freezes signs): {len(bv)} vs {len(av)}"
    # vertices correspond 1:1 (same FlexiCubes cube structure -> same dual vertex order); triangulation may differ slightly, so use before's faces
    Vb=torch.tensor(bv,device=dev); Va=torch.tensor(av,device=dev); F=torch.tensor(bf,dtype=torch.int64,device=dev)


    vnb=vnorm(Vb,F); vna=vnorm(Va,F)

    # interpolated GT normal at each vertex's nearest GT point (reference)
    ud,fid,uvw=bvh.unsigned_distance(Va.contiguous(),return_uvw=True); vi=GF[fid.long()]

    ngt=torch.nn.functional.normalize(uvw[:,0:1]*gtvn[vi[:,0]]+uvw[:,1:2]*gtvn[vi[:,1]]+uvw[:,2:3]*gtvn[vi[:,2]],dim=-1)
    dev_b=1-(vnb*ngt).sum(-1).abs(); dev_a=1-(vna*ngt).sum(-1).abs()

    alpha=(dev_a < dev_b-a.margin).float()   # use the refined vertex only if its normal is clearly closer to GT, else fall back to watertight
    # diffuse the mask over the mesh to avoid seams
    nv=len(Vb); i0,i1,i2=F[:,0],F[:,1],F[:,2]

    src=torch.cat([i0,i1,i1,i2,i2,i0]); dst=torch.cat([i1,i0,i2,i1,i0,i2])

    deg=torch.zeros(nv,device=dev).index_add_(0,src,torch.ones(len(src),device=dev)).clamp(min=1)
    for _ in range(a.smooth):
        nb=torch.zeros(nv,device=dev).index_add_(0,src,alpha[dst]); alpha=0.5*alpha+0.5*nb/deg
    Vf=Vb+alpha[:,None]*(Va-Vb)
    mo=trimesh.Trimesh(Vf.cpu().numpy(),bf,process=False); mo.export(a.out)

    print(f"[blend] refined vertices used: {(alpha>0.5).float().mean()*100:.1f}% (real detail); watertight elsewhere (wrinkles removed)  watertight={mo.is_watertight} -> {a.out}",flush=True)


# ---------------------------------------------------------------------------
# Production invocation:
#
#   python b_blend_keepbest.py --before <tag>_rfb.ply --after <tag>_rfa.ply \
#       --ref <raw.glb> --out <tag>_blended.ply
#
# Defaults kept: margin=0.002 (how much better the refined normal must be before
# its vertex is trusted), smooth=5 (mask diffusion steps, prevents seams).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
