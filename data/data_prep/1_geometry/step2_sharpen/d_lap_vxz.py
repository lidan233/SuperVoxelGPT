#!/usr/bin/env python3
"""Turn the finished mesh into the smoothed voxel occupancy that downstream stages consume.

Purpose
    The occupancy grid is a target a text model has to *predict*, and a smoother support is a
    learnable one. Saliency is computed on the sharp mesh (what is detailed); occupancy comes from
    here (where the object is). The two artifacts answer different questions, so smoothing after
    sharpening is not undoing work.

Input
    Finished meshes, listed one path per line, consumed by a pool of workers.

Output
    One sparse voxel file per object: dual-grid coordinates with per-cell vertex offsets and an
    intersection mask.

Key idea
    Uniform (umbrella) Laplacian smoothing on GPU via scatter-add over a precomputed degree table,
    then dual-grid voxelization. Component handling is an explicit switch: keeping only the largest
    component avoids shards whose dual grids overlap, which is what the released dataset used, but
    objects with genuinely detached parts can opt out. Workers coordinate through a lock-protected
    counter file, so any number of GPUs across any number of hosts can be launched independently
    and will not collide; outputs are skip-if-exists, so a killed worker is simply relaunched.
"""
import argparse,os,sys,fcntl,numpy as np,torch,trimesh
if os.environ.get("O_VOXEL_ROOT"): sys.path.insert(0, os.environ["O_VOXEL_ROOT"])
import o_voxel
dev="cuda"
# Runtime configuration, filled in by main(). Kept module-level so the worker helpers below read
# like the rest of the pipeline rather than threading a config object through every call.
BASE=LIST=OUTDIR=ERRF=None; CHUNK=6; RES=1024; ITERS=50; LAM=0.5; LARGEST_ONLY=True
paths=[]; END=0
def parse_args(argv=None):
    # Flags take precedence; each falls back to its environment variable, so the existing spawn
    # script (one worker per GPU, all sharing a flock'd queue) keeps working unchanged.
    ap=argparse.ArgumentParser(description="Laplacian smoothing + dual-grid voxelization worker.")
    ap.add_argument("--queue-dir",default=os.environ.get("DIRECT_ARRAY_BASE"),
                    help="flock queue directory holding next_task.txt / array_end.txt  [env DIRECT_ARRAY_BASE]")
    ap.add_argument("--meshlist",default=os.environ.get("MESHLIST"),
                    help="file with one *_final.ply path per line  [env MESHLIST]")
    ap.add_argument("--chunk",type=int,default=int(os.environ.get("DIRECT_ARRAY_CHUNK","6")),
                    help="ids claimed per lock acquisition  [env DIRECT_ARRAY_CHUNK]")
    ap.add_argument("--out-dir",default=os.environ.get("LAP_OUTDIR"),help="[env LAP_OUTDIR]")
    ap.add_argument("--res",type=int,default=1024,help="voxel grid resolution")
    ap.add_argument("--iters",type=int,default=50,help="Laplacian smoothing iterations")
    ap.add_argument("--lam",type=float,default=0.5,help="Laplacian step size lambda")
    ap.add_argument("--all-components",action="store_true",
                    default=os.environ.get("LAP_LARGEST_ONLY","1")!="1",
                    help="keep every component instead of only the largest  [env LAP_LARGEST_ONLY=0]")
    return ap.parse_args(argv)
def flock_do(lk,fn):
    with open(lk,"w") as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try: return fn()
        finally: fcntl.flock(f,fcntl.LOCK_UN)
def claim():
    def _c():
        nx=int(open(f"{BASE}/next_task.txt").read())
        if nx>END: return None
        new=min(nx+CHUNK,END+1); open(f"{BASE}/next_task.txt","w").write(str(new)); return (nx,new)
    return flock_do(f"{BASE}/queue.lock",_c)
def incr(name):
    def _i():
        n=int(open(f"{BASE}/{name}.txt").read()); open(f"{BASE}/{name}.txt","w").write(str(n+1))
    flock_do(f"{BASE}/{name}.lock",_i)
def logline(p,l):
    with open(p,"a") as f:
        fcntl.flock(f,fcntl.LOCK_EX); f.write(l); f.flush(); fcntl.flock(f,fcntl.LOCK_UN)
def normv(v):
    mn,mx=v.min(0),v.max(0);c=(mn+mx)/2;s=0.99999/max(float((mx-mn).max()),1e-12)
    return np.clip((v-c)*s,-0.5,0.5)
def setup(V,F):
    fl=F.long();i0,i1,i2=fl[:,0],fl[:,1],fl[:,2];on=torch.ones(len(fl),device=dev);deg=torch.zeros(len(V),device=dev)
    for i in (i0,i1,i2): deg.scatter_add_(0,i,on*2)
    return (i0,i1,i2,i0[:,None].expand(-1,3),i1[:,None].expand(-1,3),i2[:,None].expand(-1,3),deg.clamp(min=1)[:,None])
def lapm(V,c):
    i0,i1,i2,e0,e1,e2,deg=c;ns=torch.zeros_like(V);ns.scatter_add_(0,e0,V[i1]+V[i2]);ns.scatter_add_(0,e1,V[i0]+V[i2]);ns.scatter_add_(0,e2,V[i0]+V[i1]);return ns/deg
def lap_smooth(V,F):
    c=setup(V,F);V=V.clone()
    for _ in range(ITERS): V=V+LAM*(lapm(V,c)-V)
    return V
def write_vxz(path,Vs,Fs):
    Vs=np.clip(Vs,-0.5,0.5).astype(np.float32)
    vt=torch.from_numpy(Vs).float(); ft=torch.from_numpy(np.asarray(Fs)).long()
    coords,dv,inter=o_voxel.convert.mesh_to_flexible_dual_grid(vt,ft,grid_size=RES,aabb=([-0.5]*3,[0.5]*3))
    lin=coords[:,0].to(torch.int64)*(RES*RES)+coords[:,1].to(torch.int64)*RES+coords[:,2].to(torch.int64)
    o=lin.argsort(); coords=coords[o]; dv=dv[o]; inter=inter[o]
    dvu=((dv*RES-coords).clamp(0,1)*255.0).round().to(torch.uint8)
    itu=(inter[:,0].to(torch.uint8)+2*inter[:,1].to(torch.uint8)+4*inter[:,2].to(torch.uint8))[:,None]
    o_voxel.io.write_vxz(path,coords.to(torch.int32).cpu(),{"vertices":dvu.cpu(),"intersected":itu.cpu()},num_threads=1)
    return int(coords.shape[0])
def stem_of(p):
    b=os.path.basename(p)
    if b.endswith("_final.ply"): return b[:-10]
    if b.endswith("_wt.ply"): return b[:-7]
    return b.rsplit(".",1)[0]
def process(path):
    stem=stem_of(path)
    if os.path.exists(f"{OUTDIR}/{stem}.vxz"): return "skip"
    if not os.path.exists(path): return "nofile"
    m=trimesh.load(path,force="mesh",process=False)
    V=np.asarray(m.vertices,np.float64); F=np.asarray(m.faces)
    if len(V)==0 or len(F)==0: return "empty"
    if LARGEST_ONLY:
        # Keep only the biggest component. Topology safety: a smoothed multi-component mesh can
        # produce shards whose dual grids overlap. Note step 1's clean_floaters keeps every
        # component above 0.5% of the faces, so a genuinely multi-part object loses its extra parts
        # here. That is the deliberate trade the released dataset was built with.
        comps=trimesh.Trimesh(V,F,process=False).split(only_watertight=False)
        if len(comps)>1:
            big=max(comps,key=lambda c:len(c.faces)); V=np.asarray(big.vertices); F=np.asarray(big.faces)
    Vn=normv(V).astype(np.float32)
    Vt=torch.tensor(Vn,device=dev); Ft=torch.tensor(F.astype(np.int64),device=dev)
    Vs=lap_smooth(Vt,Ft).cpu().numpy()
    n=write_vxz(f"{OUTDIR}/{stem}.vxz",Vs,F); return f"OK v={n}"
def main(argv=None):
    global BASE,CHUNK,RES,LIST,ITERS,LAM,LARGEST_ONLY,OUTDIR,ERRF,paths,END
    a=parse_args(argv)
    BASE=a.queue_dir; CHUNK=a.chunk; RES=a.res; LIST=a.meshlist
    ITERS=a.iters; LAM=a.lam; LARGEST_ONLY=not a.all_components
    _def_out=os.path.join(os.path.dirname(LIST) if LIST else ".",
                          "lap_vxz" if LARGEST_ONLY else "lap_allcomp_vxz")
    OUTDIR=a.out_dir or _def_out; os.makedirs(OUTDIR,exist_ok=True)
    ERRF=os.environ.get("LAP_ERR", os.path.join(OUTDIR,"lap_err.txt"))
    paths=[l.strip() for l in open(LIST) if l.strip()]
    END=int(open(f"{BASE}/array_end.txt").read())
    print(f"[lap_vxz] up END={END} paths={len(paths)}",flush=True)
    while True:
        if os.path.exists(f"{BASE}/STOP_ALL"): break
        ck=claim()
        if ck is None: print("[drained]",flush=True); break
        for tid in range(*ck):
            if tid>=len(paths): continue
            try: r=process(paths[tid])
            except Exception as e: r="EXC:"+repr(e)[:70]
            ok=isinstance(r,str) and (r.startswith("OK") or r=="skip")
            if ok: incr("completed_count"); open(f"{BASE}/done/task_{tid}","w").close()
            else: incr("failed_count"); open(f"{BASE}/failed/task_{tid}","w").write(str(r)); logline(ERRF,f"{paths[tid]}\t{r}\n")
            if tid%500==0: print(f"[{tid}] {r}",flush=True)
    print("[lap_vxz done]",flush=True)

# one worker per GPU, all sharing one flock'd queue:
#   CUDA_VISIBLE_DEVICES=$g python d_lap_vxz.py --queue-dir <queue> --meshlist <meshlist.txt> \
#       --out-dir <vxz_dir>
if __name__=="__main__":
    main()
