"""Seal an arbitrary mesh into a single watertight surface — the precondition for compressibility.

Purpose
    superVoxelGPT compresses a shape into few tokens by projecting features onto supervoxel centers.
    That only works because the features are redundant: neighbouring regions of a surface say much
    the same thing, so a handful of centers can stand in for a lot of voxels.

    Open boundaries break the premise. When the input voxelization carries dangling bounding edges,
    the latent features have to shift in order to encode where those edges are — the redundancy the
    compression relies on is spent on recording boundary bookkeeping instead. In practice the
    supervoxel VAE then fails to converge at all.

    So watertightness here is not an aesthetic preference, nor a formality inherited from SDF
    conventions. It is what makes the representation compressible in the first place, which is why
    every mesh is forced through this stage before it enters the pipeline.

Input
    A triangle mesh in any common format. No assumptions about topology, orientation or manifoldness.

Output
    A watertight triangle mesh, decimated to a face budget and lightly Taubin-smoothed.

    Sealed but blunt. Isosurface extraction cannot represent a sharp edge — a crease in the source
    arrives here as a rounded ridge, and presmoothing the field to suppress staircase artifacts
    rounds it further. That is expected at this point, not a defect: sharpness is recovered in the
    next step by optimizing against multi-view renders of the original mesh, which can put creases
    back where the geometry actually had them instead of guessing from the blunt surface.

Key idea
    Extraction is tiled, but no topological decision is. A resolution high enough to keep thin
    features does not fit in memory as one block, and the obvious remedy — cut the volume up and
    extract each piece — is what produces cracked output, because two tiles that decide a shared
    cell's topology independently will disagree along the seam. So both decisions that could differ
    are taken once, over the whole volume, before any tile exists: the flood assigns inside and
    outside globally, and the case id of every cell, ambiguity resolution included, is computed
    against all cells at once. Tiles then only evaluate a verdict that has already been reached,
    which is what lets the seams close exactly rather than approximately.

Other information:
    Sign method. Distance is always a BVH nearest-triangle query; what differs is how the sign is
    assigned, and there are three options in reach:

      Ray stabbing (cubvh's signed_distance, mode='raystab'). Casts rays and counts crossings to
      decide parity. Available but unused here: it inherits the same closedness assumption as the
      flood while costing more, and it is sensitive to rays grazing coplanar faces.

      Analytic winding number (winding_number_branch/). Integrates over the surface, so it needs no
      closedness assumption at all and survives open boundaries, self-intersection and flipped
      normals. Orders of magnitude more cost per query. Prefer it when compute is not the
      constraint.

    The sharpening step also exposes cubvh's signed_distance in mode='watertight', which takes the
    sign from the nearest face's orientation. That yields a smoothly varying field rather than the
    binary one a flood produces, which reduces extraction staircasing — but it trusts the input's
    face orientations, which raw assets frequently get wrong.
"""
import argparse, os, sys, time, numpy as np, torch, trimesh
# FlexiCubes is vendored next to this file; this must precede the import below.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "FlexiCubes"))
from flexicubes import FlexiCubes
import cubvh, fast_simplification as fs

# Determinism settings are module-level on purpose: they must take effect on import, before any
# CUDA work happens, whether this file is run as a script or imported for its functions.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)   # makes index_add etc. deterministic -> dual vertices on tile boundaries are bit-identical

def load_norm(p, bound=0.9):
    m=trimesh.load(p,process=False,force='mesh')
    if isinstance(m,trimesh.Scene): m=trimesh.util.concatenate([g for g in m.geometry.values() if isinstance(g,trimesh.Trimesh) and len(g.faces)])
    v=np.asarray(m.vertices,np.float64); c=(v.min(0)+v.max(0))/2
    v=(v-c)*(2*bound/(v.max(0)-v.min(0)).max())
    return torch.tensor(v,dtype=torch.float32,device=dev), torch.tensor(np.asarray(m.faces),dtype=torch.int32,device=dev)

def compute_global_case_id(cells, sdf_flat, G, GG, R, fc):
    """Compute case_id globally (base occupancy + DMC ambiguity resolution, neighbours queried over all cells)
    so every tile shares it and boundary topology is consistent."""
    CC=fc.cube_corners.long().to(dev); cci=fc.cube_corners_idx.to(dev); ct=fc.check_table.to(dev)
    cg=cells[:,None,:].long()+CC[None,:,:]                          # [M,8,3]
    lin=cg[...,0]*GG+cg[...,1]*G+cg[...,2]                          # [M,8]
    occ8=(sdf_flat[lin]<0)                                          # [M,8]
    case=(occ8.int()*cci.unsqueeze(0)).sum(-1)                     # [M]
    pc=ct[case]; tc=pc[...,0]==1                                    # cells needing disambiguation
    pos=cells.long(); cl=pos[:,0]*(R*R)+pos[:,1]*R+pos[:,2]         # linear index over all cells
    order=torch.argsort(cl); sl=cl[order]
    ti=torch.nonzero(tc,as_tuple=True)[0]
    adj=pos[ti]+pc[ti,1:4]; wi=((adj>=0)&(adj<R)).all(-1); ti=ti[wi]; adj=adj[wi]
    al=adj[:,0]*(R*R)+adj[:,1]*R+adj[:,2]
    fp=torch.searchsorted(sl,al).clamp(max=len(sl)-1); mt=sl[fp]==al
    adj_g=order[fp]                                                 # global index of the neighbour cell
    adj_pc0=torch.zeros(len(al),dtype=pc.dtype,device=dev); adj_pc0[mt]=pc[adj_g[mt],0]
    inv=ti[adj_pc0==1]                                             # neighbour is also ambiguous -> flip
    case[inv]=pc[inv,-1]
    return case
def patch_global_case_id(fc, tile_gidx, case_g):
    """This tile's _get_case_id just returns the globally precomputed case (subset by surf_cubes)."""
    def g(occ_fx8, surf_cubes, res):
        return case_g[tile_gidx[surf_cubes]]
    fc._get_case_id=g

def _udf_at(vox, r):
    """Chunked UDF evaluation at voxel grid indices vox[K,3] in 0..r (world coords = idx*2/r - 1)."""
    out=torch.empty(len(vox),device=dev)
    for s in range(0,len(vox),a.chunk):
        e=min(s+a.chunk,len(vox)); pts=vox[s:e].float()*(2.0/r)-1.0
        out[s:e]=bvh.unsigned_distance(pts.contiguous(),return_uvw=False)[0]
    return out

def build_field_ctf(R, presmooth):
    """Coarse-to-fine field construction: evaluate the UDF only at surface-band grid points, never building a
    dense full-resolution UDF. Output is point-for-point identical to build_field (cells_all, dense sdf_vol, M);
    peak memory drops from 21 GB to ~6 GB."""
    G=R+1; GG=G*G
    band_eps=(a.band_eps+presmooth)*2.0/R; occ_eps=a.occ_eps*2.0/R
    keepv=(a.band_eps+presmooth)+2*presmooth+4     # conservative band width (voxels): band + presmooth's Euclidean reach (sqrt(3)*presmooth) + slack, so the presmooth stencil never reaches the placeholder values outside the band
    # ---- coarse-to-fine: evaluate band grid points only (halving chain, each level divides the next; any R that is a multiple of 64 works) ----
    levels=[R]; _r=R
    while _r>256 and _r%2==0: _r//=2; levels.append(_r)
    levels=sorted(set(levels))
    r0=levels[0]; G0=r0+1
    ii=torch.arange(G0,device=dev)
    vox=torch.stack(torch.meshgrid(ii,ii,ii,indexing='ij'),-1).reshape(-1,3)
    udf=_udf_at(vox,r0)
    m=udf<(keepv*2.0/r0); vox=vox[m]; del udf,ii,m; torch.cuda.empty_cache()
    cur=r0
    for r in levels[1:]:
        sc=r//cur
        off=torch.stack(torch.meshgrid(*[torch.arange(sc,device=dev)]*3,indexing='ij'),-1).reshape(-1,3)
        vox=(vox[:,None,:]*sc+off[None,:,:]).reshape(-1,3).clamp(0,r); del off
        udf=_udf_at(vox,r)
        m=udf<(keepv*2.0/r); vox=vox[m]; udf=udf[m]; cur=r; del m; torch.cuda.empty_cache()
    pvox=vox; pudf=udf   # band grid points (0..R) and their UDF
    # ---- floodfill over the G^3 grid, where only the band forms walls ----
    occ=torch.zeros((G,G,G),dtype=torch.bool,device=dev)
    sp=pvox[pudf<occ_eps]; occ[sp[:,0],sp[:,1],sp[:,2]]=True; del sp
    lab=cubvh.floodfill(occ); del occ; torch.cuda.empty_cache()
    insb=(lab[pvox[:,0],pvox[:,1],pvox[:,2]]!=lab[0,0,0].item()); del lab; torch.cuda.empty_cache()
    psdf=torch.where(insb,-pudf,pudf).half(); del insb
    # ---- scatter sparse values into a dense sdf_vol (downstream unchanged; non-band = large positive, and everything within presmooth's radius is inside the kept band, so band cells stay point-identical) ----
    sdf_vol=torch.full((G*G*G,),float(band_eps+10.0),dtype=torch.float16,device=dev)
    lin=pvox[:,0].long()*GG+pvox[:,1].long()*G+pvox[:,2].long()
    sdf_vol[lin]=psdf; sdf_vol=sdf_vol.view(G,G,G); del lin,psdf
    # ---- cells_all: every cell with at least one band corner ----
    bgp=pvox[pudf<band_eps]; del pvox,pudf; torch.cuda.empty_cache()
    cells=(bgp[:,None,:]-CC.int()[None,:,:]).reshape(-1,3); del bgp
    val=(cells>=0).all(1)&(cells<R).all(1)
    cells_all=torch.unique(cells[val].int(),dim=0); del cells,val; torch.cuda.empty_cache()
    if presmooth>0:
        import torch.nn.functional as _F; _s=sdf_vol[None,None]; del sdf_vol
        for _ in range(presmooth): _s=_F.avg_pool3d(_F.pad(_s,(1,1,1,1,1,1),mode='replicate'),3,stride=1)
        sdf_vol=_s[0,0]; del _s; torch.cuda.empty_cache()
    return cells_all, sdf_vol, len(cells_all)

def build_field(R, presmooth):
    """Dense field construction: udf -> floodfill -> sign x UDF. Returns cells_all, sdf_vol, M."""
    G=R+1; GG=G*G; Ntot=G*G*G
    lin1d=torch.arange(G,device=dev,dtype=torch.float32)*2.0/R-1.0
    udf_vol=torch.empty(Ntot,dtype=torch.float16,device=dev)
    for s in range(0,Ntot,a.chunk):
        e=min(s+a.chunk,Ntot); idx=torch.arange(s,e,device=dev)
        pts=torch.stack([lin1d[idx//GG],lin1d[(idx//G)%G],lin1d[idx%G]],-1)
        udf_vol[s:e]=bvh.unsigned_distance(pts,return_uvw=False)[0].half(); del idx,pts
    udf_vol=udf_vol.view(G,G,G)
    occ=udf_vol<(a.occ_eps*2.0/R); lab=cubvh.floodfill(occ); del occ; torch.cuda.empty_cache()
    inside=(lab!=lab[0,0,0].item()); del lab; torch.cuda.empty_cache()
    sdf_vol=udf_vol.clone(); sdf_vol[inside]*=-1; del inside; torch.cuda.empty_cache()
    band_eps=(a.band_eps+presmooth)*2.0/R; umin=udf_vol[:-1,:-1,:-1].clone()  # presmooth shifts the zero level set by ~N voxels, so widen the band by N as well or extraction leaves holes
    for dx,dy,dz in CC[1:].tolist(): umin=torch.minimum(umin,udf_vol[dx:dx+R,dy:dy+R,dz:dz+R])
    cells_all=torch.nonzero(umin<band_eps).int(); del umin,udf_vol; torch.cuda.empty_cache()
    if presmooth>0:   # de-staircase at the source: 3x3x3 smoothing of the SDF field before extraction (done in fp16; avg_pool accumulates in fp32 internally, so no quality loss, and it saves a 2.2 GB peak)
        import torch.nn.functional as _F; _s=sdf_vol[None,None]; del sdf_vol
        for _ in range(presmooth): _s=_F.avg_pool3d(_F.pad(_s,(1,1,1,1,1,1),mode='replicate'),3,stride=1)
        sdf_vol=_s[0,0]; del _s; torch.cuda.empty_cache()
    return cells_all, sdf_vol, len(cells_all)

def clean_floaters(m, min_frac=0.005):
    """scipy connected components: drop shards below min_frac of the faces (washout artifacts of presmooth),
    keeping the main body and any genuinely large parts."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    f0=np.asarray(m.faces); nv=len(m.vertices)
    e=np.vstack([f0[:,[0,1]],f0[:,[1,2]],f0[:,[2,0]]])
    ncomp,lab=connected_components(coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv)),directed=False)
    if ncomp<=1: return m
    flab=lab[f0[:,0]]; cnt=np.bincount(flab); thr=max(1,int(min_frac*len(f0)))
    keepc=np.nonzero(cnt>=thr)[0]; keep=f0[np.isin(flab,keepc)]
    used=np.unique(keep); remap=np.full(nv,-1,np.int64); remap[used]=np.arange(len(used))
    print(f"  floater cleanup: {ncomp} components -> kept {len(keepc)} (dropped {ncomp-len(keepc)} with <{thr} faces) {len(f0)}->{len(keep)} faces",flush=True)
    return trimesh.Trimesh(np.asarray(m.vertices)[used],remap[keep],process=False)

def run_pipeline(presmooth, taubin, res_start=None):
    """One full pass: field construction with automatic resolution backoff -> tiled extraction -> merge ->
    decimate -> Taubin. Returns (mesh, R)."""
    _bandk=a.band_eps/(a.band_eps+presmooth)   # presmooth widens the band, inflating M while the face count is unchanged; scale the estimate back by the band ratio
    t0=time.time(); R=res_start if res_start else a.res
    while True:
        cells_all,sdf_vol,M=(build_field_ctf if a.ctf else build_field)(R,presmooth)
        faces_est=FPC*M*_bandk
        if faces_est<=a.face_cap or R<=256: break
        R_new=max(256, round(R*(a.face_cap/faces_est)**0.5/64)*64)
        if R_new>=R: R_new=R-64
        print(f"! resolution backoff: ~{faces_est/1e6:.1f}M faces > cap {a.face_cap/1e6:.0f}M -> res {R}->{R_new}",flush=True)
        try:
            if a.downscale_log:
                with open(a.downscale_log,"a") as _lg: _lg.write(f"{stem},{a.res},{R_new},{M}\n")
        except Exception: pass
        del sdf_vol,cells_all; torch.cuda.empty_cache(); R=R_new
    G=R+1; GG=G*G
    case_g=compute_global_case_id(cells_all, sdf_vol.reshape(-1), G, GG, R, fc)   # global case_id, consistent across tile boundaries
    torch.cuda.empty_cache()   # release intermediates after case_id
    print(f"field built in {time.time()-t0:.1f}s  final res={R} (requested {a.res}) active cubes M={M/1e6:.1f}M ~{FPC*M*_bandk/1e6:.1f}M faces presmooth={presmooth}",flush=True)
    # ---- split into tiles along the longest axis ----
    span=cells_all.max(0).values - cells_all.min(0).values
    axis=int(torch.argmax(span)); lo=int(cells_all[:,axis].min()); hi=int(cells_all[:,axis].max())
    ntile=max(1, int(np.ceil(M/a.tile_max_cubes)))
    edges=np.linspace(lo,hi+1,ntile+1).astype(int)
    print(f"tiling: axis={['x','y','z'][axis]} {ntile} tiles overlap={a.overlap}",flush=True)
    gtfn_tri=gtv[gtf.long()]
    allV=[]; allF=[]; voff=0
    for ti in range(ntile):
        tt=time.time()
        c0=edges[ti]-(a.overlap if ti>0 else 0); c1=edges[ti+1]+(a.overlap if ti<ntile-1 else 0)
        sel=(cells_all[:,axis]>=c0)&(cells_all[:,axis]<c1)
        tile_gidx=torch.nonzero(sel,as_tuple=True)[0]
        cells=cells_all[sel]
        if len(cells)==0: continue
        cg=cells[:,None,:]+CC.int()[None,:,:]
        lin_c=cg[...,0].long()*GG+cg[...,1].long()*G+cg[...,2].long()
        uniq,inv=torch.unique(lin_c.reshape(-1),return_inverse=True); cube_fx8=inv.reshape(len(cells),8).int()
        verts0=torch.stack([uniq//GG,(uniq//G)%G,uniq%G],-1).float()*2.0/R-1.0
        sdf_init=sdf_vol.reshape(-1)[uniq].float().clamp(-1,1)
        del cg,lin_c,inv
        udf_v,fid_v,uvw_v=bvh.unsigned_distance(verts0.contiguous(),return_uvw=True)
        cp=(uvw_v[:,0:1]*gtfn_tri[fid_v.long(),0]+uvw_v[:,1:2]*gtfn_tri[fid_v.long(),1]+uvw_v[:,2:3]*gtfn_tri[fid_v.long(),2])
        gdir=torch.nn.functional.normalize(verts0-cp,dim=-1)
        scale=(2-1e-8)/(R*2); deform=(-a.grad_eta*gdir*udf_v[:,None].clamp(max=2.0/R))
        gv=verts0+ (deform.clamp(-scale,scale))
        patch_global_case_id(fc,tile_gidx,case_g)
        with torch.no_grad():
            v,f,_=fc(gv,sdf_init,cube_fx8,R,training=False)
        if len(f)>0:
            allV.append(v); allF.append(f.long()+voff); voff+=len(v)
        print(f"  tile {ti+1}/{ntile}: cubes={len(cells)/1e6:.1f}M -> {len(v)}v/{len(f)}f  {time.time()-tt:.1f}s",flush=True)
        del cells,uniq,cube_fx8,verts0,sdf_init,gv,v,f; torch.cuda.empty_cache()
    del sdf_vol,cells_all; torch.cuda.empty_cache()
    # ---- merge and dedupe on GPU ----
    t0=time.time()
    V=torch.cat(allV,0); F=torch.cat(allF,0); del allV,allF; torch.cuda.empty_cache()
    K=1<<15
    q=torch.round((V+1.0)*(K/2.0)).long().clamp(0,K)
    key=(q[:,0]*(K+1)+q[:,1])*(K+1)+q[:,2]
    uk,inv=torch.unique(key,return_inverse=True)
    Vu=torch.zeros((len(uk),3),device=dev,dtype=V.dtype); Vu[inv]=V
    F2=inv[F]; del F,key,q
    dg=(F2[:,0]==F2[:,1])|(F2[:,1]==F2[:,2])|(F2[:,0]==F2[:,2]); F2=F2[~dg]
    Fs=torch.sort(F2,dim=1).values; Fu=torch.unique(Fs,dim=0)
    ee=torch.cat([Fu[:,[0,1]],Fu[:,[1,2]],Fu[:,[0,2]]],0); ee=torch.sort(ee,dim=1).values
    euq,ecnt=torch.unique(ee,dim=0,return_counts=True); nb=int((ecnt==1).sum()); nnm=int((ecnt>=3).sum())
    print(f"GPU merge {time.time()-t0:.1f}s: {len(Vu)} vertices {len(Fu)} faces  boundary edges={nb} non-manifold={nnm}",flush=True)
    Vd,Fd=Vu,Fu
    if a.tile_faces>0 and len(Fu)>a.tile_faces:
        t0=time.time(); ratio=1-a.tile_faces/len(Fu)
        vv,ff=fs.simplify(Vu.cpu().numpy().astype(np.float32),Fu.cpu().numpy().astype(np.int32),target_reduction=float(ratio),agg=a.dec_agg)
        Vd=torch.tensor(vv,device=dev); Fd=torch.tensor(ff.astype(np.int64),device=dev)
        print(f"decimate(agg={a.dec_agg}) {time.time()-t0:.1f}s -> {len(Fd)} faces",flush=True)
    del V,Vu,F2,Fs,Fu,ee,euq; torch.cuda.empty_cache()
    # ---- GPU Taubin (moves vertices only on the manifold, topology untouched) ----
    if taubin>0:
        t0=time.time(); Fl=Fd.long(); i0,i1,i2=Fl[:,0],Fl[:,1],Fl[:,2]; on=torch.ones(len(Fl),device=dev)
        deg=torch.zeros(len(Vd),device=dev)
        deg.scatter_add_(0,i0,on*2); deg.scatter_add_(0,i1,on*2); deg.scatter_add_(0,i2,on*2); deg=deg.clamp(min=1)[:,None]
        e0,e1,e2=i0[:,None].expand(-1,3),i1[:,None].expand(-1,3),i2[:,None].expand(-1,3)
        for _ in range(taubin):
            for w in (0.5,-0.53):
                ns=torch.zeros_like(Vd)
                ns.scatter_add_(0,e0,Vd[i1]+Vd[i2]); ns.scatter_add_(0,e1,Vd[i0]+Vd[i2]); ns.scatter_add_(0,e2,Vd[i0]+Vd[i1])
                Vd=Vd+w*(ns/deg-Vd)
        print(f"GPU Taubin×{taubin} {time.time()-t0:.1f}s",flush=True)
    mm=trimesh.Trimesh(Vd.cpu().numpy(),Fd.cpu().numpy(),process=False); del Vd,Fd; torch.cuda.empty_cache()
    mm.remove_unreferenced_vertices()
    return mm, R

def run_oom(presmooth, taubin, res_start=None):
    """OOM handling: by default (skip_on_oom) record the id and skip rather than lowering res; otherwise retry lower."""
    rs=res_start if res_start else a.res
    for _att in range(5):
        try:
            return run_pipeline(presmooth, taubin, res_start=rs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
            if 'out of memory' not in str(_e).lower() and not isinstance(_e,torch.cuda.OutOfMemoryError): raise
            torch.cuda.empty_cache()
            if a.skip_on_oom:   # extreme mesh (still OOM under ctf, would need lower res) -> record and skip
                if a.skip_dir:   # concurrency-safe: one empty marker file per skipped id (distinct names, no race)
                    try:
                        os.makedirs(a.skip_dir,exist_ok=True)
                        open(os.path.join(a.skip_dir, os.path.basename(a.ref)),'w').close()
                    except Exception: pass
                print(f"! OOM@res{rs} would need lower res -> recording skip for {os.path.basename(a.ref)}, exiting",flush=True)
                import sys; sys.exit(3)
            rs_new=max(256, int(rs*0.8)//64*64)
            if rs_new>=rs: rs_new=rs-128
            if rs_new<256: print("! OOM persists at the lowest resolution",flush=True); raise
            print(f"! OOM@res{rs} -> retrying at res {rs}->{rs_new}",flush=True); rs=rs_new
    raise RuntimeError("OOM retries exhausted")


def main():
    global CC, FPC, R, a, bvh, dev, fc, gtf, gtv, m, stem
    dev="cuda"

    ap=argparse.ArgumentParser()
    ap.add_argument("--ref",required=True); ap.add_argument("--res",type=int,default=1024)

    ap.add_argument("--face_cap",type=float,default=16e6,help="face-count ceiling for automatic resolution backoff (margin under nvdiffrast's 16.7M hard limit); drops from --res until the estimate fits")
    ap.add_argument("--occ_eps",type=float,default=2.0); ap.add_argument("--band_eps",type=float,default=2.5)

    ap.add_argument("--grad_eta",type=float,default=0.5)
    ap.add_argument("--tile_max_cubes",type=float,default=8e6,help="max active cubes per tile; total M / this = tile count")
    ap.add_argument("--overlap",type=int,default=2,help="overlap in cube layers between tiles (for stitching)")
    ap.add_argument("--tile_faces",type=float,default=0,help="final decimate target; 0 = no simplification (default off: decimate can break watertightness)")
    ap.add_argument("--dec_agg",type=float,default=3.0,help="fast_simplification aggressiveness; <=3 preserves topology (watertight), the library default of 7 destroys thin structures")
    ap.add_argument("--taubin",type=int,default=0,help="Taubin smoothing iterations at the end (default 0: smoothing is presmooth's job; taubin 4 is only used when presmooth broke watertightness)")
    ap.add_argument("--presmooth",type=int,default=2,help="remove staircasing at the source: smooth the SDF field with a 3x3x3 kernel N times before extraction (production=2, halves chamfer; falls back to taubin automatically if thin walls lose watertightness)")
    ap.add_argument("--ctf",type=int,default=1,help="coarse-to-fine field construction (1 = multi-level, saves memory, default; 0 = fully dense, kept as the equivalence reference)")
    ap.add_argument("--skip_on_oom",type=int,default=1,help="1 = on OOM that would require lowering res, record the id and skip (exit code 3); 0 = retry at lower res")
    ap.add_argument("--skip_dir",default="",help="directory for skip markers (one empty file per id; safe under concurrent tasks)")
    ap.add_argument("--chunk",type=int,default=8_000_000)
    ap.add_argument("--out",required=True,help="output directory")
    ap.add_argument("--downscale_log",default="",help="optional csv appended when an object needs a resolution backoff")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)

    stem=os.path.splitext(os.path.basename(a.ref))[0][:34]
    print(f"=== tiled extraction {stem} res={a.res} face_cap={a.face_cap/1e6:.0f}M ===",flush=True)
    gtv,gtf=load_norm(a.ref); bvh=cubvh.cuBVH(gtv,gtf)

    fc=FlexiCubes(dev); CC=fc.cube_corners.long()

    T0=time.time()

    FPC=0.76   # empirical faces per active cube (measured 0.72~0.78 on sample 063; take the high end to stay safe)

    # ---- main driver: prefer presmooth; fall back if it breaks watertightness; lower res further if the actual
    #      face count exceeds the hard limit; degrade gracefully on OOM throughout ----
    HARD=16_700_000   # nvdiffrast's 2^24 face hard limit
    m, R = run_oom(a.presmooth, a.taubin)
    m = clean_floaters(m)                                          # drop shard components
    if a.presmooth>0 and not m.is_watertight:
        print(f"! presmooth={a.presmooth} broke watertightness (thin-wall washout that a wider band cannot fix) -> rebuilding with presmooth=0 + taubin{max(a.taubin,4)}",flush=True)
        m, R = run_oom(0, max(a.taubin,4)); m = clean_floaters(m)
    _ps_use=a.presmooth if m.is_watertight and a.presmooth>0 else 0
    _tb_use=0 if _ps_use>0 else max(a.taubin,4)
    tries=0
    while len(m.faces)>HARD and R>256 and tries<3:               # the face estimate can be off -> force another resolution backoff
        tries+=1; Rn=(min(int(R*(HARD/len(m.faces))**0.5),R-64)//64)*64
        print(f"! actual {len(m.faces)/1e6:.1f}M faces > 16.7M hard limit -> forcing res {R}->{Rn} rebuild",flush=True)
        m, R = run_oom(_ps_use, _tb_use, res_start=Rn); m = clean_floaters(m)
        if _ps_use>0 and not m.is_watertight:
            m, R = run_oom(0, max(a.taubin,4), res_start=Rn); m = clean_floaters(m)
    print(f"  final watertight={m.is_watertight}",flush=True)
    p=f"{a.out}/{stem}_tiled_r{R}.ply"; m.export(p)

    nax=(np.abs(m.face_normals).max(1)>0.99).mean()*100
    print(f"\n=== done in {time.time()-T0:.0f}s -> {len(m.vertices)}v/{len(m.faces)}f watertight={m.is_watertight} staircase={nax:.1f}% -> {p}",flush=True)


# ---------------------------------------------------------------------------
# Production invocation (what actually built the released dataset):
#
#   python a_tile_extract.py --ref <raw.glb> --res 1024 --out <dir>
#
# Everything else stayed at its default: presmooth=2, ctf=1 (narrow band),
# tile_faces=0 (no decimation — decimation breaks watertightness), dec_agg=3.0,
# taubin=0, band_eps/occ_eps defaults, face_cap=16e6, skip_on_oom=1.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
