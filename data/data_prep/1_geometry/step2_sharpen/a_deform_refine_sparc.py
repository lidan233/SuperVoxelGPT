"""Restore the creases that isosurface extraction rounded off.

Purpose
    A sealed mesh is blunt: marching cubes cannot represent a sharp edge, so every crease in the
    source becomes a rounded ridge. Since creases are precisely what makes a region salient
    downstream, they have to be recovered — by optimizing the field against renders of the original
    mesh rather than by any geometric heuristic.

Input
    The original reference mesh (the rendering target) and, optionally, a mesh to start the field
    from instead of rebuilding it.

Output
    Two meshes: the analytic initialization and the optimized result, so the gain is measured rather
    than assumed. Optionally a third, guarded pick between them.

Key idea
    A faithful implementation of rendering-based refinement: a depth + normal multi-view loss, with
    gradients flowing through differentiable marching cubes back into the field, and a light mask
    term to stabilize silhouettes.

    Two things keep it from diverging. The deformation is tanh-clamped to half a voxel, so topology
    and watertightness survive no matter what the loss asks for; and signs are frozen, so the field
    can move the surface but never flip a cell's inside/outside decision. Optimizing vertex
    positions without that clamp reliably fuzzes or diverges, because the normal's sensitivity to
    position scales inversely with triangle area — a five-order-of-magnitude amplification on small
    faces.

    Production additionally frees the field values themselves, not just the deformation: the clamp
    bounds how far a vertex can travel, while the field can move the surface further. A 3D Laplacian
    regularizer on the field is what stops that extra freedom from turning into wrinkles — it
    forbids high-frequency jitter while still permitting low-frequency sharpening.
"""
import os,sys,argparse,time
import numpy as np, torch, trimesh, nvdiffrast.torch as dr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "step1_watertight", "FlexiCubes"))
from flexicubes import FlexiCubes
import cubvh
dev='cuda'
def load_norm(p,bound=0.9):
    m=trimesh.load(p,process=False,force='mesh')
    if isinstance(m,trimesh.Scene):
        m=trimesh.util.concatenate([g for g in m.geometry.values() if isinstance(g,trimesh.Trimesh) and len(g.faces)])
    v=np.asarray(m.vertices,np.float64); c=(v.min(0)+v.max(0))/2
    v=(v-c)*(2*bound/(v.max(0)-v.min(0)).max())
    return torch.tensor(v,dtype=torch.float32,device=dev), torch.tensor(np.asarray(m.faces),dtype=torch.int32,device=dev)
def persp(fovy=0.7,n=0.1,f=10):
    t=np.tan(fovy/2);P=torch.zeros(4,4,device=dev);P[0,0]=1/t;P[1,1]=1/t;P[2,2]=-(f+n)/(f-n);P[2,3]=-2*f*n/(f-n);P[3,2]=-1;return P
_P=None;_U=None
def rand_cams(n,rad=2.6):
    global _P,_U
    if _P is None:_P=persp()
    if _U is None:_U=torch.tensor([0,1,0.],device=dev)
    az=torch.rand(n,device=dev)*(2*np.pi);el=torch.asin(torch.rand(n,device=dev)*1.7-0.85);ce=torch.cos(el)
    eye=torch.stack([rad*ce*torch.cos(az),rad*torch.sin(el),rad*ce*torch.sin(az)],-1)
    z=torch.nn.functional.normalize(eye,dim=-1);x=torch.nn.functional.normalize(torch.cross(_U.expand_as(z),z,dim=-1),dim=-1);y=torch.cross(z,x,dim=-1)
    V=torch.zeros(n,4,4,device=dev);V[:,3,3]=1;V[:,0,:3]=x;V[:,1,:3]=y;V[:,2,:3]=z
    V[:,0,3]=-(x*eye).sum(-1);V[:,1,3]=-(y*eye).sum(-1);V[:,2,3]=-(z*eye).sum(-1)
    return _P[None]@V
def vnorm(v,f):
    fl=f.long();i0,i1,i2=fl[:,0],fl[:,1],fl[:,2];idx=torch.cat([i0,i1,i2])
    fn=torch.cross(v[i1]-v[i0],v[i2]-v[i0],dim=-1);vn=torch.zeros_like(v);vn.scatter_add_(0,idx[:,None].expand(-1,3),fn.repeat(3,1))
    return torch.nn.functional.normalize(vn,dim=-1)
def render(ctx,v,f,mvp,res,vn=None):
    vh=torch.cat([v,torch.ones_like(v[:,:1])],1);clip=torch.einsum('bij,vj->bvi',mvp,vh).contiguous()
    rast,_=dr.rasterize(ctx,clip,f,[res,res]);mask=(rast[...,3:4]>0).float();mask=dr.antialias(mask,rast,clip,f)
    if vn is None: vn=vnorm(v,f)
    nrm,_=dr.interpolate(vn[None].contiguous(),rast,f);depth,_=dr.interpolate(clip[...,3:4].contiguous(),rast,f)
    return {"mask":mask,"normal":nrm*mask,"depth":depth*mask}

def compute_global_case_id(cells, sdf_flat, G, GG, R, fc):
    CCl=fc.cube_corners.long().to(dev); cci=fc.cube_corners_idx.to(dev); ct=fc.check_table.to(dev)
    cg=cells[:,None,:].long()+CCl[None,:,:]; lin=cg[...,0]*GG+cg[...,1]*G+cg[...,2]
    occ8=(sdf_flat[lin]<0); case=(occ8.int()*cci.unsqueeze(0)).sum(-1)
    pc=ct[case]; tc=pc[...,0]==1; pos=cells.long(); cl=pos[:,0]*(R*R)+pos[:,1]*R+pos[:,2]
    order=torch.argsort(cl); sl=cl[order]; ti=torch.nonzero(tc,as_tuple=True)[0]
    adj=pos[ti]+pc[ti,1:4]; wi=((adj>=0)&(adj<R)).all(-1); ti=ti[wi]; adj=adj[wi]
    al=adj[:,0]*(R*R)+adj[:,1]*R+adj[:,2]; fp=torch.searchsorted(sl,al).clamp(max=len(sl)-1); mt=sl[fp]==al
    adj_g=order[fp]; adj_pc0=torch.zeros(len(al),dtype=pc.dtype,device=dev); adj_pc0[mt]=pc[adj_g[mt],0]
    inv=ti[adj_pc0==1]; case[inv]=pc[inv,-1]; return case
def patch_case(fc, gidx, case_g):
    def g(occ_fx8, surf_cubes, res): return case_g[gidx[surf_cubes]]
    fc._get_case_id=g

# ---------------------------------------------------------------------------
# Production invocation:
#
#   python a_deform_refine_sparc.py --ref <raw.glb> --res 1024 --iters 200 \
#       --opt_sdf 1 --opt_weight 0 --sdf_smooth 50 \
#       --out_before <tag>_rfb.ply --out_after <tag>_rfa.ply
#
# What --opt_sdf 1 actually does: the grid-vertex deformation is ALWAYS optimized;
# opt_sdf adds the SDF values as a second parameter group on top of it. So production
# optimizes deformation AND SDF, and only leaves the FlexiCubes weights frozen
# (--opt_weight 0). The faithful Sparc3D setting is --opt_sdf 0, i.e. deformation
# alone; freeing the SDF sharpens further because the deformation is tanh-clamped to
# +/-half a voxel and cannot move the surface beyond that. --sdf_smooth 50 is what
# keeps the freed SDF from developing wrinkles. Defaults kept elsewhere:
# qef=0, dc=0, sign_mode=floodfill, amp=0, early_stop=1 (es_min=60, patience=15).
# ---------------------------------------------------------------------------

def run(a):
    T0=time.time();ctx=dr.RasterizeCudaContext()
    gtv,gtf=load_norm(a.ref);gtvn=vnorm(gtv,gtf);bvh=cubvh.cuBVH(gtv,gtf)
    # --base_mesh: build the field and initial deformation from this mesh (e.g. Taubin-smoothed watertight); the render loss still targets the GT (--ref)
    if a.base_mesh:
        bmv,bmf=load_norm(a.base_mesh); base_bvh=cubvh.cuBVH(bmv,bmf); base_fv=bmv[bmf.long()]
        print(f"refine starts from base_mesh({a.base_mesh}), rendering targets GT({a.ref})",flush=True)
    else:
        base_bvh=bvh; base_fv=gtv[gtf.long()]
    R=a.res;G=R+1;GG=G*G
    fc=FlexiCubes(dev,qef_reg_scale=a.qef_reg);CC=fc.cube_corners.long()

    # ---- field: udf -> binary floodfill sign -> SDF -> presmooth (identical to tile_extract, unchanged) ----
    lin1d=torch.arange(G,device=dev,dtype=torch.float32)*2.0/R-1.0
    udf_vol=torch.empty(G*G*G,dtype=torch.float16,device=dev)
    if a.sign_mode=='signed':   # smooth signed distance (reduces extraction staircasing): sdf = signed_distance, sign varies smoothly
        sdf_flat=torch.empty(G*G*G,dtype=torch.float16,device=dev)
        for s in range(0,G*G*G,8_000_000):
            e=min(s+8_000_000,G*G*G);idx=torch.arange(s,e,device=dev)
            pts=torch.stack([lin1d[idx//GG],lin1d[(idx//G)%G],lin1d[idx%G]],-1)
            sd=base_bvh.signed_distance(pts.contiguous(),mode='watertight')[0]
            sdf_flat[s:e]=sd.half(); udf_vol[s:e]=sd.abs().half(); del idx,pts,sd
        udf_vol=udf_vol.view(G,G,G); sdf_vol=sdf_flat.view(G,G,G); del sdf_flat
    else:                        # binary floodfill sign (the original approach)
        for s in range(0,G*G*G,8_000_000):
            e=min(s+8_000_000,G*G*G);idx=torch.arange(s,e,device=dev)
            pts=torch.stack([lin1d[idx//GG],lin1d[(idx//G)%G],lin1d[idx%G]],-1)
            udf_vol[s:e]=base_bvh.unsigned_distance(pts,return_uvw=False)[0].half();del idx,pts
        udf_vol=udf_vol.view(G,G,G)
        occ=udf_vol<(a.occ_eps*2.0/R);lab=cubvh.floodfill(occ);inside=(lab!=lab[0,0,0].item());del occ,lab
        sdf_vol=udf_vol.clone();sdf_vol[inside]*=-1;del inside
    band_eps=(a.band_eps+a.presmooth)*2.0/R;umin=udf_vol[:-1,:-1,:-1].clone()
    for dx,dy,dz in CC[1:].tolist(): umin=torch.minimum(umin,udf_vol[dx:dx+R,dy:dy+R,dz:dz+R])
    cells_all=torch.nonzero(umin<band_eps).int();del umin
    if a.presmooth>0:
        import torch.nn.functional as _F;_s=sdf_vol[None,None]
        for _ in range(a.presmooth):_s=_F.avg_pool3d(_F.pad(_s,(1,1,1,1,1,1),mode='replicate'),3,stride=1)
        sdf_vol=_s[0,0];del _s
    torch.cuda.empty_cache()

    case_g=compute_global_case_id(cells_all, sdf_vol.reshape(-1), G, GG, R, fc)
    patch_case(fc, torch.arange(len(cells_all),device=dev), case_g)
    cg=cells_all[:,None,:]+CC.int()[None,:,:]
    lin_c=cg[...,0].long()*GG+cg[...,1].long()*G+cg[...,2].long()
    uniq,inv=torch.unique(lin_c.reshape(-1),return_inverse=True);cube_fx8=inv.reshape(len(cells_all),8).int()
    verts0=torch.stack([uniq//GG,(uniq//G)%G,uniq%G],-1).float()*2.0/R-1.0
    sdf_init=sdf_vol.reshape(-1)[uniq].float().clamp(-1,1)
    # 6-neighbourhood over the SDF field (for the smoothness regularizer); uniq is already sorted (torch.unique sorts by default)
    NBR=None
    if a.sdf_smooth>0 and (a.full or a.opt_sdf):
        N=len(uniq); NBR=torch.full((N,6),-1,dtype=torch.long,device=dev)
        for j,off in enumerate([1,-1,G,-G,GG,-GG]):
            nb=uniq+off; pos=torch.searchsorted(uniq,nb).clamp(max=N-1); val=uniq[pos]==nb; NBR[val,j]=pos[val]
        NBRV=(NBR>=0).float()
    del cg,lin_c,inv,sdf_vol,udf_vol;torch.cuda.empty_cache()

    # ---- Step 3's analytic deformation (x - eta*grad UDF) is the init; step 4 optimizes only this deformation ----
    scale=(2-1e-8)/(R*2)
    udf_v,fid_v,uvw_v=base_bvh.unsigned_distance(verts0.contiguous(),return_uvw=True)
    gtfn=base_fv
    cp=(uvw_v[:,0:1]*gtfn[fid_v.long(),0]+uvw_v[:,1:2]*gtfn[fid_v.long(),1]+uvw_v[:,2:3]*gtfn[fid_v.long(),2])
    gdir=torch.nn.functional.normalize(verts0-cp,dim=-1)
    d0=(-a.grad_eta*gdir*udf_v[:,None].clamp(max=2.0/R)).clamp(-scale,scale)
    deform=torch.nn.Parameter(torch.atanh((d0/scale*0.99).clamp(-0.98,0.98)))   # init = analytic deformation
    sdf_sign=torch.sign(sdf_init).clamp(min=-1); sdf_sign[sdf_sign==0]=1        # freeze the sign so topology cannot jump
    OPT_SDF=bool(a.full or a.opt_sdf); OPT_W=bool(a.full or a.opt_weight)
    sdf=torch.nn.Parameter(sdf_init.clone()) if OPT_SDF else sdf_init
    weight=torch.nn.Parameter(torch.zeros((len(cells_all),21),device=dev)) if OPT_W else None
    print(f"optimizing deform{'+SDF' if OPT_SDF else ''}{'+weights' if OPT_W else ''}  lap={a.lap_reg}  grid={len(verts0)}",flush=True)

    # QEF grad_func: analytic UDF gradient normalize(x-cp) removes MC ripples (same as tile_extract); [legacy] --dc uses GT face normals (faceted)
    _gfv=gtv[gtf.long()]; gt_fn=torch.nn.functional.normalize(torch.cross(_gfv[:,1]-_gfv[:,0],_gfv[:,2]-_gfv[:,0],dim=-1),dim=-1)
    GRADF=None
    if a.qef or a.dc:
        def GRADF(pts):
            with torch.no_grad():                                 # cp = nearest GT point (target, detached); pts stays live so QEF remains differentiable
                _,fid,uvw=bvh.unsigned_distance(pts.contiguous(),return_uvw=True)
                if a.dc: return gt_fn[fid.long()]                 # [legacy] face normal
                cp=uvw[:,0:1]*_gfv[fid.long(),0]+uvw[:,1:2]*_gfv[fid.long(),1]+uvw[:,2:3]*_gfv[fid.long(),2]
            return torch.nn.functional.normalize(pts-cp,dim=-1)   # analytic gradient, pts stays live
    def extract(training=False):
        gv=verts0+scale*torch.tanh(deform)
        gf=None if training else GRADF                            # linear interpolation while optimizing (differentiable and stable); the final export uses QEF to remove ripples
        if weight is not None:
            return fc(gv,sdf,cube_fx8,R,beta_fx12=weight[:,:12],alpha_fx8=weight[:,12:20],gamma_f=weight[:,20],training=training,grad_func=gf)
        return fc(gv, sdf, cube_fx8, R, training=training,grad_func=gf)
    # before optimization (analytic init)
    with torch.no_grad():
        V,F,_=extract(training=bool(a.export_train)); trimesh.Trimesh(V.cpu().numpy(),F.cpu().numpy(),process=False).export(a.out_before)
        print(f"[before] {len(V)}v training={bool(a.export_train)} -> {a.out_before}",flush=True)
        if len(F) > a.face_cap:
            import math
            nr=max(128, int(a.res*math.sqrt(a.face_cap/len(F)))//8*8)
            print(f"[RES_DOWN] faces={len(F)} > cap{int(a.face_cap)} -> res {a.res}->{nr}",flush=True)
            open(a.out_after+".suggest_res","w").write(str(nr))
            return 42

    # memory: inside the loop the bvh is only used by GRADF, so release bvh/base_bvh when GRADF is None (the qef=0 default)
    if GRADF is None:
        del bvh, base_bvh, gt_fn, _gfv; torch.cuda.empty_cache()
    plist=[{'params':[deform],'lr':a.lr}]
    if OPT_SDF: plist.append({'params':[sdf],'lr':a.sdf_lr})
    if OPT_W: plist.append({'params':[weight],'lr':a.lr})
    opt=torch.optim.Adam(plist);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.iters)
    es_ema=None; es_best=float('inf'); es_best_it=0   # early stop: EMA-smoothed loss trend
    es_best_deform=None; es_best_sdf=None             # snapshot of the best parameters (export the best, not the state at stop time)
    for it in range(a.iters):
        opt.zero_grad();mvp=rand_cams(a.batch)
        with torch.no_grad(): tgt=render(ctx,gtv,gtf,mvp,a.train_res,vn=gtvn)
        _amp=torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=bool(a.amp))
        with _amp:
            V,F,L_dev=extract(training=True)                       # differentiable MC; gradients flow back to the deformation (and to SDF/weights under --full)
            buf=render(ctx,V,F.int(),mvp,a.train_res)
            Ldp=((((buf['depth']-tgt['depth'])*tgt['mask'])**2).sum(-1)+1e-8).sqrt().mean()*1.0
            Ln=((buf['normal']-tgt['normal'])**2*tgt['mask']).mean()*1.0
            Lm=(buf['mask']-tgt['mask']).abs().mean()*0.5
            loss=Ldp+Ln+Lm
        if OPT_W: loss=loss + L_dev.mean()*a.ldev + weight[:,:20].abs().mean()*0.1
        if a.sdf_smooth>0 and NBR is not None:   # 3D Laplacian on the SDF field: the right way to suppress wrinkles
            nb=sdf[NBR.clamp(min=0)]                       # [N,6] neighbour SDF values
            lap_sdf=sdf - (nb*NBRV).sum(1)/NBRV.sum(1).clamp(min=1)   # high-frequency component of the SDF field
            loss=loss + lap_sdf.pow(2).mean()*a.sdf_smooth
        if a.lap_reg>0:   # explicit Laplacian penalty on the output mesh (current F)
            fl2=F.long();i0,i1,i2=fl2[:,0],fl2[:,1],fl2[:,2]
            e2=torch.cat([i0,i1,i2]);nb=torch.cat([i1,i0,i2,i1,i0,i2])  # approximate adjacency
            src=torch.cat([i0,i1,i1,i2,i2,i0]);dst=torch.cat([i1,i0,i2,i1,i0,i2])
            nsum=torch.zeros_like(V).index_add_(0,src,V[dst]);deg=torch.zeros(len(V),device=dev).index_add_(0,src,torch.ones(len(src),device=dev)).clamp(min=1)[:,None]
            Llap=((V-nsum/deg)*(a.res/2.0)).pow(2).sum(-1).mean()*a.lap_reg
            loss=loss+Llap
        _l=loss.item()
        loss.backward();opt.step();sch.step()
        if OPT_SDF:  # freeze signs: the SDF may not flip sign (keeps topology watertight and matches the frozen case_id)
            with torch.no_grad(): sdf.data=torch.where(sdf_sign>0, sdf.data.clamp(min=1e-4), sdf.data.clamp(max=-1e-4))
        # ---- early stop: EMA-smoothed loss (alpha=0.2); after warmup, stop once es_patience iterations pass without a new low ----
        es_ema=_l if es_ema is None else 0.2*_l+0.8*es_ema
        if es_ema < es_best-1e-6:
            es_best=es_ema; es_best_it=it
            es_best_deform=deform.detach().clone()                    # snapshot the best
            if OPT_SDF: es_best_sdf=sdf.detach().clone()
        if a.early_stop and it>=a.es_min and it-es_best_it>=a.es_patience:
            print(f"[early-stop] it{it} EMA{es_ema:.4f} no new low for {a.es_patience} iterations (best {es_best:.4f}@it{es_best_it}) -> stopping",flush=True)
            break
        if it%40==0 or it==a.iters-1:
            with torch.no_grad(): dmax=(verts0+scale*torch.tanh(deform)-verts0).abs().max().item()
            print(f"it{it:4d} loss={_l:.4f} EMA={es_ema:.4f} Ldp={Ldp.item():.4f} Ln={Ln.item():.4f} |deform|max={dmax:.5f}",flush=True)
    # on early stop: restore the best parameters before exporting (export the optimum, not the rebound at stop time)
    if es_best_deform is not None:
        with torch.no_grad():
            deform.data.copy_(es_best_deform)
            if OPT_SDF and es_best_sdf is not None: sdf.data.copy_(es_best_sdf)
        print(f"[early-stop] restoring best@it{es_best_it} (EMA {es_best:.4f}) before export",flush=True)
    with torch.no_grad():
        V,F,_=extract(training=bool(a.export_train))
    mo_after=trimesh.Trimesh(V.cpu().numpy(),F.cpu().numpy(),process=False); mo_after.export(a.out_after)
    print(f"[after] {len(V)}v -> {a.out_after}  total {time.time()-T0:.0f}s",flush=True)

    # ---- guard: compare view-independent high-frequency roughness before/after; fall back to watertight if it got rougher ----
    def _rough(m):
        import scipy.sparse as _sp
        Vv=np.asarray(m.vertices); Ff=np.asarray(m.faces); nv=len(Vv)
        e=np.vstack([Ff[:,[0,1]],Ff[:,[1,2]],Ff[:,[2,0]]]); e=np.vstack([e,e[:,::-1]])
        A=_sp.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv)).tocsr(); A.data[:]=1; A=A.minimum(1)
        dg=np.asarray(A.sum(1)).ravel(); dg[dg==0]=1
        return float(np.median(np.linalg.norm(Vv-(A@Vv)/dg[:,None],axis=1)))
    if a.out_final:
        mo_before=trimesh.load(a.out_before,process=False)
        rb=_rough(mo_before); ra=_rough(mo_after); ra_vox=ra/(2.0/R); rb_vox=rb/(2.0/R)
        if ra > rb*a.guard_margin and ra_vox > a.guard_floor:   # fall back only when it is both rougher than before and absolutely rough enough (not just noise)
            mo_before.export(a.out_final); pick=f"watertight (before) [drifted: median roughness {rb_vox:.3f}->{ra_vox:.3f} voxels]"
        else:
            mo_after.export(a.out_final); pick=f"refined (after) [median roughness {rb_vox:.3f}->{ra_vox:.3f} voxels, OK]"
        print(f"[guard] -> using {pick} -> {a.out_final}",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ref",required=True)
    ap.add_argument("--out_before",required=True); ap.add_argument("--out_after",required=True)
    ap.add_argument("--res",type=int,default=1024); ap.add_argument("--iters",type=int,default=400)
    ap.add_argument("--batch",type=int,default=6); ap.add_argument("--train_res",type=int,default=1024)
    ap.add_argument("--face_cap",type=float,default=8.0e6,help="nvdiffrast hard limit is 2^23 = 8.39M faces; if [before] exceeds it, exit 42 so the caller can retry at lower res")
    ap.add_argument("--lr",type=float,default=1e-2)
    ap.add_argument("--occ_eps",type=float,default=2.0); ap.add_argument("--band_eps",type=float,default=2.5)
    ap.add_argument("--presmooth",type=int,default=2); ap.add_argument("--grad_eta",type=float,default=0.5)
    ap.add_argument("--sign_mode",default="floodfill",help="floodfill = binary sign (original); signed = cubvh smooth signed distance (reduces extraction staircasing)")
    ap.add_argument("--dc",type=int,default=0,help="[legacy] 1 = grad_func uses GT face normals through QEF (faceted, not recommended)")
    ap.add_argument("--qef",type=int,default=0,help="[off by default] QEF adds noise on flat regions (gradient noise gets amplified); plain linear interpolation is more stable. Enable only for purely curved objects where ripples matter")
    ap.add_argument("--qef_reg",type=float,default=0.2,help="FlexiCubes qef_reg_scale (0.2 is stable and suppresses spikes on flat regions)")
    ap.add_argument("--base_mesh",default="",help="mesh to start refinement from (e.g. the Taubin-smoothed watertight one); the field and deformation start there while rendering still targets the GT")
    ap.add_argument("--full",type=int,default=0,help="1 = optimize SDF and FlexiCubes weights together (sharper, but no longer Sparc3D)")
    ap.add_argument("--opt_sdf",type=int,default=0,help="ablation: enable SDF optimization alone"); ap.add_argument("--opt_weight",type=int,default=0,help="ablation: enable FlexiCubes weight optimization alone")
    ap.add_argument("--export_train",type=int,default=0,help="export [before] with training=True (to test whether quad-split removes the texture artifacts)")
    ap.add_argument("--amp",type=int,default=0,help="1 = bf16 autocast in the forward pass to save memory (matmuls lose precision; geometry/positions stay fp32)")
    ap.add_argument("--early_stop",type=int,default=1,help="1 = early stop once the EMA-smoothed loss fails to set a new low for es_patience iterations (saves time, near-identical quality)")
    ap.add_argument("--es_min",type=int,default=60,help="minimum iterations before early stop may trigger (warmup)")
    ap.add_argument("--es_patience",type=int,default=15,help="stop after this many iterations without the EMA setting a new low (progress stalled or reversed)")
    ap.add_argument("--ldev",type=float,default=0.25); ap.add_argument("--sdf_lr",type=float,default=3e-3)
    ap.add_argument("--lap_reg",type=float,default=0.0,help="Laplacian high-frequency penalty on the output mesh (legacy; fights the vertices, not recommended)")
    ap.add_argument("--sdf_smooth",type=float,default=0.0,help="3D Laplacian smoothness regularizer on the SDF field: forbids high-frequency jitter (wrinkles) while still allowing low-frequency sharpening")
    ap.add_argument("--out_final",default="",help="guarded final output: fall back to watertight if roughness increased, otherwise keep the refined mesh")
    ap.add_argument("--guard_margin",type=float,default=1.5,help="fall back when after-roughness exceeds before-roughness by this factor")
    ap.add_argument("--guard_floor",type=float,default=0.04,help="...and only when the after median roughness (voxels) also exceeds this floor; already-smooth results always keep the refined mesh")
    return run(ap.parse_args()) or 0


if __name__ == "__main__":
    sys.exit(main())
