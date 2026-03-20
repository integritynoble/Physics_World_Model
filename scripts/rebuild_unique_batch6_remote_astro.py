"""
Rebuild standard datasets — Batch 6: Remote sensing + Astronomy + Optical/computational (30 modalities).
Each modality gets a unique phantom.

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "benchmark"
N = 10

def to_uint8(a):
    a=np.nan_to_num(a); mn,mx=float(a.min()),float(a.max())
    if mx-mn<1e-10: return np.zeros(a.shape,dtype=np.uint8)
    return ((a-mn)/(mx-mn)*255).clip(0,255).astype(np.uint8)

def save_img(a, p):
    if a.ndim==1:
        fig,ax=plt.subplots(1,1,figsize=(6,2)); ax.plot(a); fig.tight_layout(); fig.savefig(str(p),dpi=100); plt.close(fig)
    elif a.ndim==2: Image.fromarray(to_uint8(a),"L").save(str(p))
    elif a.ndim==3:
        if a.shape[-1]==2: Image.fromarray(to_uint8(np.log1p(np.sqrt(a[...,0]**2+a[...,1]**2))),"L").save(str(p))
        elif a.shape[-1]==3: Image.fromarray(to_uint8(a),"RGB").save(str(p))
        elif a.shape[-1]<=20: Image.fromarray(to_uint8(a[:,:,a.shape[-1]//2]),"L").save(str(p))
        else: Image.fromarray(to_uint8(a[a.shape[0]//2]),"L").save(str(p))

def save_mod(mod, samples, ft, fd, me):
    out=DATA/mod/"standard"; out.mkdir(parents=True,exist_ok=True)
    for old in out.glob("standard_*.h5"): old.unlink()
    files=[]
    for i,(x,y) in enumerate(samples):
        h5p=out/f"standard_{mod}_{i:02d}.h5"
        with h5py.File(str(h5p),"w") as f:
            f.create_dataset("x_true",data=x,compression="gzip"); f.create_dataset("y_ideal",data=y,compression="gzip")
            hp=f.create_group("H_params"); hp.attrs["forward_type"]=ft; hp.attrs["noise"]="none"
            md=f.create_group("metadata"); md.attrs["source"]=me.get("src",""); md.attrs["reference"]=me.get("ref",""); md.attrs["date_built"]="2026-03-14"
        files.append(h5p.name)
        d=out/"images"/f"sample_{i:02d}"; d.mkdir(parents=True,exist_ok=True)
        save_img(x,d/"x_true.png"); save_img(y,d/"y_ideal.png")
    with open(out/"metadata.json","w") as f:
        json.dump({"modality":mod,"canonical_dataset":me.get("src",""),"reference":me.get("ref",""),
            "n_samples":len(samples),"forward_type":ft,"forward_model":fd,
            "x_true_shape":list(samples[0][0].shape),"y_ideal_shape":list(samples[0][1].shape),
            "noise":"none","mismatch":"none","date_built":"2026-03-14",
            "gcs_path":f"gs://pwm-benchmark-datasets/datasets/Benchmark/{mod}/standard/",
            "status":"done","files":files},f,indent=2)
    with open(out/"spec.json","w") as f:
        json.dump({"modality":mod,"forward_type":ft,"noise":"none","mismatch":"none","split":"standard"},f,indent=2)
    print(f"  {mod}: {len(samples)} samples")


# ===================================================================
# REMOTE SENSING / ASTRONOMY PHANTOMS
# ===================================================================

def phantom_adaptive_optics(seed):
    """AO — stellar field with atmospheric turbulence (point sources + halo)."""
    rng=np.random.RandomState(seed); S=256; x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Stars (point sources of varying brightness)
    n_stars=rng.randint(10,30)
    for _ in range(n_stars):
        sx,sy=rng.uniform(10,S-10),rng.uniform(10,S-10)
        brightness=10**rng.uniform(-1,1)
        x+=brightness*np.exp(-0.5*((xx-sx)**2+(yy-sy)**2)/1**2)
    # Nebula background
    nebula=ndimage.gaussian_filter(np.clip(rng.randn(S,S),0,None).astype(np.float32),sigma=20)
    x+=nebula*0.05
    x=x/(x.max()+1e-10)
    return x.astype(np.float32)

def phantom_coronagraphy_exo(seed):
    """Coronagraphy — exoplanet system (star + faint companions)."""
    rng=np.random.RandomState(seed); S=256; x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Central star (very bright, partially suppressed)
    x+=0.3*np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/3**2)
    # Coronagraphic residual (speckle halo)
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)+1e-10
    speckle_pattern=ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32),sigma=5)
    x+=0.1*np.exp(-r/(S*0.15))*np.abs(speckle_pattern)
    # Exoplanets (very faint point sources)
    n_planets=rng.randint(1,4)
    for _ in range(n_planets):
        sep=rng.uniform(S*0.08,S*0.3); pa=rng.uniform(0,2*np.pi)
        px=cx+sep*np.cos(pa); py=cy+sep*np.sin(pa)
        contrast=10**rng.uniform(-3,-1)
        x+=contrast*np.exp(-0.5*((xx-px)**2+(yy-py)**2)/1.5**2)
    # Debris disk (faint ring)
    if rng.rand()>0.5:
        disk_r=S*rng.uniform(0.1,0.25)
        disk=np.exp(-0.5*((r-disk_r)/3)**2)*0.02
        x+=disk
    return np.clip(x,0,1).astype(np.float32)

def phantom_lucky_double_star(seed):
    """Lucky imaging — double star with atmospheric seeing."""
    rng=np.random.RandomState(seed); S=256; x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Primary star
    x+=0.8*np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/1.5**2)
    # Secondary star (close companion)
    sep=rng.uniform(5,20); pa=rng.uniform(0,2*np.pi)
    sx=cx+sep*np.cos(pa); sy=cy+sep*np.sin(pa)
    x+=rng.uniform(0.2,0.6)*np.exp(-0.5*((xx-sx)**2+(yy-sy)**2)/1.5**2)
    # Diffraction spikes
    for angle in [0,np.pi/2]:
        spike=(np.abs((xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle))<1.5).astype(np.float32)
        spike*=np.exp(-0.5*((-(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle)))**2/80**2)
        x+=spike*0.05
    return np.clip(x,0,1).astype(np.float32)

def phantom_sar_urban(seed):
    """SAR — urban area radar backscatter (buildings, roads, vegetation)."""
    rng=np.random.RandomState(seed); S=256; x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Ground (moderate backscatter)
    x+=0.25
    # Roads (low backscatter — smooth)
    n_roads=rng.randint(2,5)
    for _ in range(n_roads):
        if rng.rand()>0.5:  # horizontal
            ry=rng.uniform(S*0.1,S*0.9); rw=rng.uniform(3,8)
            x-=np.exp(-0.5*(yy-ry)**2/rw**2)*0.15
        else:  # vertical
            rx=rng.uniform(S*0.1,S*0.9); rw=rng.uniform(3,8)
            x-=np.exp(-0.5*(xx-rx)**2/rw**2)*0.15
    # Buildings (very bright — double-bounce)
    n_bldg=rng.randint(5,15)
    for _ in range(n_bldg):
        bx=rng.uniform(S*0.1,S*0.9); by=rng.uniform(S*0.1,S*0.9)
        bw=rng.uniform(5,15); bh=rng.uniform(5,15)
        bldg=((np.abs(xx-bx)<bw)&(np.abs(yy-by)<bh)).astype(np.float32)
        x+=bldg*rng.uniform(0.4,0.8)
    # Vegetation (textured, moderate-high)
    veg=ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32),sigma=5)
    veg_mask=(x<0.3).astype(np.float32)
    x+=veg_mask*np.clip(veg,0,None)*0.15
    # Speckle texture
    x+=ndimage.gaussian_filter(np.abs(rng.randn(S,S)).astype(np.float32),sigma=1)*0.05
    return np.clip(x,0,1).astype(np.float32)

def phantom_insar_deformation(seed):
    """InSAR — ground deformation fringes (earthquake/subsidence)."""
    rng=np.random.RandomState(seed); S=256; x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Deformation field (subsidence bowl)
    sx=cx+rng.uniform(-S*0.15,S*0.15); sy=cy+rng.uniform(-S*0.15,S*0.15)
    r=np.sqrt((xx-sx)**2+(yy-sy)**2)
    max_phase=rng.uniform(4,12)*np.pi
    deformation=max_phase*np.exp(-r**2/(S*0.2)**2)
    # Wrapped phase (fringes)
    x=0.5+0.4*np.cos(deformation)
    # Fault line (phase discontinuity)
    if rng.rand()>0.4:
        fault_angle=rng.uniform(0,np.pi)
        fault_offset=rng.uniform(S*0.1,S*0.3)
        fault_line=(xx-cx)*np.cos(fault_angle)+(yy-cy)*np.sin(fault_angle)
        x+=(fault_line>0).astype(np.float32)*0.1*np.exp(-np.abs(fault_line)/5)
    return np.clip(x,0,1).astype(np.float32)

def phantom_polsar_forest(seed):
    """PolSAR — polarimetric decomposition (forest/agriculture)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S,3),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Forest (volume scattering — green channel)
    forest_mask=ndimage.gaussian_filter((rng.rand(S,S)>0.4).astype(np.float32),sigma=10)
    x[:,:,1]+=forest_mask*0.6
    # Urban (double-bounce — red channel)
    n_urban=rng.randint(3,8)
    for _ in range(n_urban):
        ux=rng.uniform(S*0.1,S*0.9); uy=rng.uniform(S*0.1,S*0.9)
        ur=rng.uniform(8,20)
        x[:,:,0]+=np.exp(-0.5*((xx-ux)**2+(yy-uy)**2)/ur**2)*0.7
    # Water/bare soil (surface scattering — blue channel)
    water_mask=1-forest_mask
    x[:,:,2]+=water_mask*0.4
    # Agricultural fields (mixed)
    for _ in range(rng.randint(2,5)):
        fx=rng.uniform(S*0.1,S*0.9); fy=rng.uniform(S*0.1,S*0.9)
        fr=rng.uniform(15,30)
        field=np.exp(-0.5*((xx-fx)**2+(yy-fy)**2)/fr**2)
        x[:,:,1]+=field*0.3; x[:,:,2]+=field*0.2
    return np.clip(x,0,1).astype(np.float32)

def phantom_ocean_color_chlorophyll(seed):
    """Ocean color — chlorophyll concentration map (remote sensing reflectance)."""
    rng=np.random.RandomState(seed); S=256; C=9
    x=np.zeros((S,S,C),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Ocean baseline (deep blue water)
    ocean_spec=np.exp(-0.5*((np.arange(C)-1)/1.5)**2)*0.3
    x+=ocean_spec[None,None,:]
    # Phytoplankton bloom (high chlorophyll — shifts spectrum to green)
    n_blooms=rng.randint(1,4)
    for _ in range(n_blooms):
        bx=rng.uniform(S*0.1,S*0.9); by=rng.uniform(S*0.1,S*0.9)
        br=rng.uniform(20,60)
        bloom=np.exp(-0.5*((xx-bx)**2+(yy-by)**2)/br**2)
        chl_spec=np.exp(-0.5*((np.arange(C)-3)/1.5)**2)
        x+=bloom[:,:,None]*chl_spec[None,None,:]*0.5
    # Sediment plume (river discharge)
    if rng.rand()>0.4:
        sx=rng.choice([0,S-1]); sy=rng.uniform(S*0.2,S*0.8)
        r=np.sqrt((xx-sx)**2+(yy-sy)**2)
        sed_spec=np.exp(-0.5*((np.arange(C)-5)/2)**2)
        x+=np.exp(-r/(S*0.15))[:,:,None]*sed_spec[None,None,:]*0.3
    return np.clip(x,0,1).astype(np.float32)

def phantom_weather_storm(seed):
    """Weather radar — storm cell reflectivity (mesoscale convective system)."""
    rng=np.random.RandomState(seed); S=256; x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Storm cell (concentric reflectivity rings)
    r=np.sqrt((xx-cx+rng.uniform(-20,20))**2+(yy-cy+rng.uniform(-20,20))**2)
    # Core (>50 dBZ)
    x+=0.9*np.exp(-0.5*(r/(S*0.05))**2)
    # Strong echo (40-50 dBZ)
    x+=0.6*np.exp(-0.5*(r/(S*0.1))**2)
    # Moderate (25-40 dBZ)
    x+=0.3*np.exp(-0.5*(r/(S*0.2))**2)
    # Anvil/stratiform (weak, extended)
    x+=0.1*np.exp(-0.5*(r/(S*0.35))**2)
    # Hook echo (tornado signature)
    if rng.rand()>0.5:
        ha=rng.uniform(0,2*np.pi)
        hr=S*0.08
        hx=cx+hr*np.cos(ha); hy=cy+hr*np.sin(ha)
        hook=np.exp(-0.5*((xx-hx)**2+(yy-hy)**2)/(S*0.03)**2)
        x+=hook*0.4
    # Secondary cell
    if rng.rand()>0.3:
        s2x=cx+rng.uniform(-S*0.3,S*0.3); s2y=cy+rng.uniform(-S*0.3,S*0.3)
        r2=np.sqrt((xx-s2x)**2+(yy-s2y)**2)
        x+=0.5*np.exp(-0.5*(r2/(S*0.08))**2)
    return np.clip(x,0,1).astype(np.float32)

def phantom_passive_mw_seaice(seed):
    """Passive microwave — sea ice concentration (brightness temperature)."""
    rng=np.random.RandomState(seed); S=256; C=7
    x=np.zeros((S,S,C),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Open ocean (high Tb at low freq, lower at high freq)
    ocean_spec=0.5-0.15*np.linspace(0,1,C)
    x+=ocean_spec[None,None,:]
    # Sea ice (different Tb signature)
    ice_mask=ndimage.gaussian_filter((rng.rand(S,S)>0.5).astype(np.float32),sigma=15)
    ice_spec=0.85-0.05*np.linspace(0,1,C)
    x=x*(1-ice_mask[:,:,None])+ice_mask[:,:,None]*ice_spec[None,None,:]
    # Ice edge (transition zone)
    # Land mask
    if rng.rand()>0.5:
        land=(yy<S*0.15).astype(np.float32)
        land_spec=0.7+0.1*np.linspace(0,1,C)
        x=x*(1-land[:,:,None])+land[:,:,None]*land_spec[None,None,:]
    return np.clip(x,0,1).astype(np.float32)

def phantom_radio_interf(seed):
    """Radio interferometry — radio galaxy (jet + lobes)."""
    rng=np.random.RandomState(seed); S=256; x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # AGN core (bright point)
    x+=0.8*np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/2**2)
    # Jets (two opposing narrow beams)
    jet_angle=rng.uniform(0,np.pi)
    for sign in [-1,1]:
        for t in np.linspace(0,1,200):
            jx=cx+sign*S*0.35*t*np.cos(jet_angle)
            jy=cy+sign*S*0.35*t*np.sin(jet_angle)
            jw=2+t*3
            if 0<=jx<S and 0<=jy<S:
                x+=np.exp(-0.5*((xx-jx)**2+(yy-jy)**2)/jw**2)*0.3*(1-0.5*t)
    # Radio lobes (diffuse at jet ends)
    for sign in [-1,1]:
        lx=cx+sign*S*0.32*np.cos(jet_angle)
        ly=cy+sign*S*0.32*np.sin(jet_angle)
        lr=S*rng.uniform(0.06,0.12)
        x+=np.exp(-0.5*((xx-lx)**2+(yy-ly)**2)/lr**2)*0.2
    return np.clip(x,0,1).astype(np.float32)

def phantom_multispectral_urban(seed):
    """Multispectral satellite — urban/vegetation scene (13 Sentinel-2 bands)."""
    rng=np.random.RandomState(seed); S=256; C=13
    x=np.zeros((S,S,C),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Vegetation (NDVI signature — high NIR, low red)
    veg_mask=ndimage.gaussian_filter((rng.rand(S,S)>0.5).astype(np.float32),sigma=20)
    veg_spec=np.array([0.03,0.05,0.03,0.04,0.2,0.25,0.3,0.35,0.35,0.3,0.15,0.1,0.05],dtype=np.float32)
    x+=veg_mask[:,:,None]*veg_spec[None,None,:]*rng.uniform(0.5,1.0)
    # Urban (high visible, moderate NIR)
    urban_mask=1-veg_mask
    urban_spec=np.array([0.1,0.12,0.15,0.15,0.18,0.2,0.2,0.2,0.18,0.15,0.12,0.1,0.08],dtype=np.float32)
    x+=urban_mask[:,:,None]*urban_spec[None,None,:]*rng.uniform(0.5,1.0)
    # Water body
    if rng.rand()>0.4:
        wx=rng.uniform(S*0.2,S*0.8); wy=rng.uniform(S*0.2,S*0.8); wr=rng.uniform(15,40)
        water=(np.sqrt((xx-wx)**2+(yy-wy)**2)<wr).astype(np.float32)
        water_spec=np.array([0.05,0.06,0.06,0.04,0.02,0.01,0.005,0.002,0.001,0.001,0.001,0.001,0.001],dtype=np.float32)
        x=x*(1-water[:,:,None])+water[:,:,None]*water_spec[None,None,:]*2
    return np.clip(x,0,1).astype(np.float32)


# ===================================================================
# OPTICAL / COMPUTATIONAL PHANTOMS
# ===================================================================

def phantom_hdr_scene(seed):
    """HDR — high dynamic range scene (indoor with window)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S,3),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Dark room interior
    x[:,:,:]=0.15
    # Window (very bright — outdoor scene visible)
    wx1=S*rng.uniform(0.2,0.4); wx2=S*rng.uniform(0.6,0.8)
    wy1=S*rng.uniform(0.1,0.3); wy2=S*rng.uniform(0.5,0.7)
    window=((xx>wx1)&(xx<wx2)&(yy>wy1)&(yy<wy2)).astype(np.float32)
    x[:,:,0]+=window*0.7; x[:,:,1]+=window*0.8; x[:,:,2]+=window*0.9
    # Light spill from window onto floor/wall
    spill=np.exp(-0.5*((yy-wy2)/30)**2)*window.max(axis=1,keepdims=True)*0.3
    x[:,:,:]+=spill[:,:,None]
    # Table/furniture (dark objects)
    for _ in range(rng.randint(1,3)):
        fx=rng.uniform(S*0.1,S*0.9); fy=rng.uniform(S*0.6,S*0.9)
        fw=rng.uniform(20,50); fh=rng.uniform(10,25)
        furn=((np.abs(xx-fx)<fw)&(np.abs(yy-fy)<fh)).astype(np.float32)
        x-=furn[:,:,None]*0.08
    return np.clip(x,0,1).astype(np.float32)

def phantom_light_field_scene(seed):
    """Light field — 3D scene with depth variation (for refocusing)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Background (far)
    x+=0.2+0.05*ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32),sigma=20)
    # Objects at different depths
    depths=[0.3,0.5,0.7,1.0]
    for d in depths:
        ox=rng.uniform(S*0.15,S*0.85); oy=rng.uniform(S*0.15,S*0.85)
        osize=rng.uniform(10,30)/d
        x+=np.exp(-0.5*((xx-ox)**2+(yy-oy)**2)/osize**2)*d*0.5
    return np.clip(x,0,1).astype(np.float32)

def phantom_integral_microlens(seed):
    """Integral imaging — microlens array light field (different from Stanford LF)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Scene behind microlens array
    # Checkerboard at varying depth
    period=rng.uniform(20,40)
    checker=(np.mod(xx//period+yy//period,2)).astype(np.float32)
    x+=checker*0.5
    # Spherical object in foreground
    sx,sy=S//2+rng.uniform(-30,30),S//2+rng.uniform(-30,30)
    sr=rng.uniform(20,40)
    sphere=np.clip(1-np.sqrt((xx-sx)**2+(yy-sy)**2)/sr,0,1)**2
    x+=sphere*0.6
    # Microlens grid pattern
    ml_period=rng.uniform(8,14)
    ml_pattern=np.cos(2*np.pi*xx/ml_period)*np.cos(2*np.pi*yy/ml_period)
    x*=(1+0.1*ml_pattern)
    return np.clip(x,0,1).astype(np.float32)

def phantom_event_edges(seed):
    """Event camera — scene with moving edges (event generation)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Static textured background
    x+=0.3+0.1*ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32),sigma=10)
    # Moving objects (create edge map)
    n_obj=rng.randint(2,6)
    for _ in range(n_obj):
        ox=rng.uniform(S*0.1,S*0.9); oy=rng.uniform(S*0.1,S*0.9)
        osize=rng.uniform(15,40)
        shape_type=rng.choice(["circle","rect"])
        if shape_type=="circle":
            obj=(np.sqrt((xx-ox)**2+(yy-oy)**2)<osize).astype(np.float32)
        else:
            obj=((np.abs(xx-ox)<osize)&(np.abs(yy-oy)<osize*0.7)).astype(np.float32)
        x+=obj*rng.uniform(0.2,0.5)
    return np.clip(x,0,1).astype(np.float32)

def phantom_flash_lidar_outdoor(seed):
    """Flash LiDAR — outdoor scene depth map (different from KITTI LiDAR)."""
    rng=np.random.RandomState(seed); H=64; W=256
    x=np.zeros((H,W),dtype=np.float32)
    yy,xx=np.mgrid[:H,:W].astype(np.float32)
    # Ground plane (increasing depth with y)
    x+=0.2+0.5*(yy/H)
    # Buildings at various distances
    n_bldg=rng.randint(2,5)
    for _ in range(n_bldg):
        bx=rng.uniform(W*0.1,W*0.9); bw=rng.uniform(10,40)
        bh=rng.uniform(H*0.3,H*0.7); bd=rng.uniform(0.3,0.8)
        bldg=((np.abs(xx-bx)<bw)&(yy<bh)).astype(np.float32)
        x=x*(1-bldg)+bldg*bd
    # Trees (irregular shapes)
    for _ in range(rng.randint(1,4)):
        tx=rng.uniform(W*0.1,W*0.9); ty=rng.uniform(H*0.2,H*0.6)
        tr=rng.uniform(8,20); td=rng.uniform(0.2,0.6)
        tree=np.exp(-0.5*((xx-tx)**2+(yy-ty)**2)/tr**2)*(yy<ty+5).astype(np.float32)
        x=np.maximum(x,tree*td)
    return np.clip(x,0,1).astype(np.float32)

def phantom_lidar_pointcloud(seed):
    """LiDAR — 3D point cloud range image (autonomous driving)."""
    rng=np.random.RandomState(seed); H=64; W=1024
    x=np.zeros((H,W),dtype=np.float32)
    yy,xx=np.mgrid[:H,:W].astype(np.float32)
    # Road surface
    x+=0.15+0.2*(yy/H)
    # Vehicles (box-like at various distances)
    n_cars=rng.randint(3,8)
    for _ in range(n_cars):
        cx=rng.uniform(W*0.1,W*0.9); cy=rng.uniform(H*0.3,H*0.8)
        cw=rng.uniform(15,30); ch=rng.uniform(5,12)
        cd=rng.uniform(0.1,0.5)
        car=((np.abs(xx-cx)<cw)&(np.abs(yy-cy)<ch)).astype(np.float32)
        x=x*(1-car)+car*cd
    # Sparse scan pattern (LiDAR only has beams at certain elevations)
    scan_mask=np.zeros((H,W),dtype=np.float32)
    for beam in range(0,H,max(1,H//32)):
        scan_mask[beam,:]=1.0
    x*=scan_mask
    return np.clip(x,0,1).astype(np.float32)

def phantom_tof_indoor(seed):
    """ToF camera — indoor scene depth map (room with furniture)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Back wall
    x+=0.8
    # Floor
    x+=(yy>S*0.6).astype(np.float32)*(-0.3+0.5*(yy-S*0.6)/(S*0.4))
    # Furniture
    for _ in range(rng.randint(2,5)):
        fx=rng.uniform(S*0.1,S*0.9); fy=rng.uniform(S*0.4,S*0.85)
        fw=rng.uniform(20,50); fh=rng.uniform(15,40); fd=rng.uniform(0.2,0.5)
        furn=((np.abs(xx-fx)<fw)&(np.abs(yy-fy)<fh)).astype(np.float32)
        x=x*(1-furn)+furn*fd
    # Person (standing)
    if rng.rand()>0.3:
        px=rng.uniform(S*0.2,S*0.8); py=S*0.5
        person=np.exp(-0.5*((xx-px)**2/10**2+(yy-py)**2/25**2))*(yy<S*0.8).astype(np.float32)
        x=np.minimum(x,0.3+0.3*(1-person))
    return np.clip(x,0,1).astype(np.float32)

def phantom_structured_light_3d(seed):
    """Structured light — 3D surface with fringe projection."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # 3D surface (face-like — structured light commonly used for face scanning)
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Base face shape
    face=np.clip(1-r/(S*0.35),0,1)**1.5
    x+=face*0.5
    # Nose (protruding)
    nose=np.exp(-0.5*((xx-cx)**2/(8**2)+(yy-cy+S*0.03)**2/(12**2)))
    x+=nose*0.3
    # Eyes (recessed)
    for side in [-1,1]:
        ex=cx+side*S*0.08; ey=cy-S*0.06
        eye=np.exp(-0.5*((xx-ex)**2+(yy-ey)**2)/5**2)
        x-=eye*0.15
    return np.clip(x,0,1).astype(np.float32)

def phantom_polarization_birefringent(seed):
    """Polarization imaging — birefringent crystal (Mueller matrix/Stokes)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S,4),dtype=np.float32)  # Stokes [I,Q,U,V]
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Birefringent crystal (retardance varies with thickness)
    crystal=np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(S*0.3)**2)
    retardance=crystal*np.pi*2  # full-wave retardation
    fast_axis=rng.uniform(0,np.pi)
    x[:,:,0]=0.5+0.3*crystal  # I (total intensity)
    x[:,:,1]=0.3*crystal*np.cos(2*retardance)*np.cos(2*fast_axis)  # Q
    x[:,:,2]=0.3*crystal*np.cos(2*retardance)*np.sin(2*fast_axis)  # U
    x[:,:,3]=0.1*crystal*np.sin(2*retardance)  # V (circular)
    # Stress birefringence (radial pattern)
    theta=np.arctan2(yy-cy,xx-cx)
    x[:,:,1]+=0.1*np.cos(2*theta)*crystal
    x[:,:,2]+=0.1*np.sin(2*theta)*crystal
    return np.clip(x,0,1).astype(np.float32)

def phantom_ghost_mask(seed):
    """Ghost imaging — transmission mask (bucket detection target)."""
    rng=np.random.RandomState(seed); S=128
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Resolution target (bar pattern)
    for i in range(5):
        period=S/(4+i*2)
        bars=((np.mod(xx,period*2)<period)).astype(np.float32)
        region=((yy>S*(0.1+i*0.16))&(yy<S*(0.22+i*0.16))).astype(np.float32)
        x+=bars*region*0.7
    # Circle (for calibration)
    r=np.sqrt((xx-cx)**2+(yy-cy-S*0.35)**2)
    x+=(r<S*0.05).astype(np.float32)*0.8
    return np.clip(x,0,1).astype(np.float32)

def phantom_quantum_entangled(seed):
    """Entangled photon — quantum ghost image (SPDC correlation)."""
    rng=np.random.RandomState(seed); S=64
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Simple object (letter or shape — low resolution)
    # "Q" shape (quantum!)
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    x+=((r<S*0.3)&(r>S*0.2)).astype(np.float32)*0.7
    # Tail of Q
    tail=((xx>cx)&(xx<cx+S*0.2)&(np.abs(yy-cy-S*0.15)<3)).astype(np.float32)
    x+=tail*0.7
    return np.clip(x,0,1).astype(np.float32)

def phantom_quantum_illum(seed):
    """Quantum illumination — low-reflectivity target in thermal noise."""
    rng=np.random.RandomState(seed); S=64
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Target (low reflectivity object against noisy background)
    n_targets=rng.randint(1,4)
    for _ in range(n_targets):
        tx=rng.uniform(S*0.15,S*0.85); ty=rng.uniform(S*0.15,S*0.85)
        tr=rng.uniform(3,10)
        x+=np.exp(-0.5*((xx-tx)**2+(yy-ty)**2)/tr**2)*rng.uniform(0.3,0.7)
    return np.clip(x,0,1).astype(np.float32)


# ===================================================================
# NDT / GEOPHYSICS PHANTOMS
# ===================================================================

def phantom_gpr_subsurface(seed):
    """GPR — subsurface B-scan (buried objects, layers)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Soil layers (horizontal interfaces)
    n_layers=rng.randint(2,5)
    prev_y=0
    for i in range(n_layers):
        layer_y=prev_y+S//(n_layers+1)+rng.randint(-10,10)
        layer_thick=rng.uniform(3,8)
        layer=np.exp(-0.5*(yy-layer_y)**2/layer_thick**2)
        x+=layer*rng.uniform(0.2,0.5)
        prev_y=int(layer_y)
    # Buried objects (hyperbolic reflections)
    n_obj=rng.randint(2,5)
    for _ in range(n_obj):
        ox=rng.uniform(S*0.1,S*0.9); oy=rng.uniform(S*0.2,S*0.7)
        # Hyperbola: t = t0 + sqrt(x^2 + depth^2) / v
        for col in range(S):
            delay=np.sqrt((col-ox)**2+(S*0.1)**2)/S*0.5
            hyp_y=oy+delay*S*0.5
            if 0<=hyp_y<S:
                x[int(hyp_y),col]+=rng.uniform(0.4,0.7)
    x=ndimage.gaussian_filter(x,sigma=[1,0.5])
    return np.clip(x,0,1).astype(np.float32)

def phantom_impedance_conductivity(seed):
    """EIT — conductivity distribution (lung ventilation monitoring)."""
    rng=np.random.RandomState(seed); S=64
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Thorax cross-section
    body=(r<S*0.42).astype(np.float32)
    x+=body*0.3  # baseline conductivity
    # Lungs (lower conductivity when inflated)
    for side in [-1,1]:
        lx=cx+side*S*0.13
        lung=np.exp(-0.5*((xx-lx)**2/(S*0.1)**2+(yy-cy)**2/(S*0.18)**2))
        x-=lung*0.15*body  # air-filled = lower conductivity
    # Heart (higher conductivity — blood)
    heart=np.exp(-0.5*((xx-cx+S*0.03)**2+(yy-cy+S*0.05)**2)/(S*0.06)**2)
    x+=heart*0.2*body
    return np.clip(x,0,1).astype(np.float32)

def phantom_dot_absorption(seed):
    """DOT — optical absorption distribution (breast tumor)."""
    rng=np.random.RandomState(seed); D,H,W=16,64,64
    x=np.zeros((D,H,W),dtype=np.float32)
    zz,yy,xx=np.mgrid[:D,:H,:W].astype(np.float32)
    cx,cy,cz=W//2,H//2,D//2
    # Breast tissue (ellipsoidal)
    breast=((xx-cx)**2/(W*0.4)**2+(yy-cy)**2/(H*0.4)**2+(zz-cz)**2/(D*0.4)**2<1)
    x+=breast.astype(np.float32)*0.3  # background absorption
    # Tumor (high absorption due to angiogenesis)
    tx=cx+rng.uniform(-W*0.15,W*0.15)
    ty=cy+rng.uniform(-H*0.15,H*0.15)
    tz=cz+rng.uniform(-D*0.15,D*0.15)
    tr=rng.uniform(3,8)
    tumor=np.exp(-0.5*((xx-tx)**2+(yy-ty)**2+(zz-tz)**2)/tr**2)
    x+=tumor*0.6*breast.astype(np.float32)
    return np.clip(x,0,1).astype(np.float32)

def phantom_acoustic_emission_signal(seed):
    """Acoustic emission — crack growth AE signal (time series)."""
    rng=np.random.RandomState(seed); N=1024
    t=np.linspace(0,1,N)
    x=np.zeros(N,dtype=np.float32)
    # AE burst signals (transient wavepackets)
    n_bursts=rng.randint(3,8)
    for _ in range(n_bursts):
        t0=rng.uniform(0.05,0.95)
        freq=rng.uniform(100,500)*2*np.pi
        duration=rng.uniform(0.005,0.02)
        amp=rng.uniform(0.3,0.9)
        envelope=np.exp(-0.5*((t-t0)/duration)**2)
        x+=amp*envelope*np.sin(freq*(t-t0))
    x=x/(np.abs(x).max()+1e-10)
    return x.astype(np.float32)

def phantom_acoustic_micro_void(seed):
    """Acoustic microscopy — subsurface void/delamination in IC package."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # IC package (layered structure)
    x+=0.4  # die/substrate background
    # Bond pad pattern (grid of bright spots)
    pad_period=rng.uniform(15,25)
    for i in range(int(S/pad_period)):
        for j in range(int(S/pad_period)):
            px=i*pad_period+pad_period/2; py=j*pad_period+pad_period/2
            if S*0.15<px<S*0.85 and S*0.15<py<S*0.85:
                x+=np.exp(-0.5*((xx-px)**2+(yy-py)**2)/3**2)*0.3
    # Delamination (dark region — poor acoustic coupling)
    dx=rng.uniform(S*0.2,S*0.8); dy=rng.uniform(S*0.2,S*0.8)
    dr=rng.uniform(15,35)
    delam=np.exp(-0.5*((xx-dx)**2+(yy-dy)**2)/dr**2)
    x-=delam*0.3
    # Void (very dark spot)
    vx=dx+rng.uniform(-5,5); vy=dy+rng.uniform(-5,5)
    void=np.exp(-0.5*((xx-vx)**2+(yy-vy)**2)/5**2)
    x-=void*0.25
    return np.clip(x,0,1).astype(np.float32)

def phantom_sonar_bathymetry(seed):
    """Sonar — seafloor bathymetry (multibeam echo sounder)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Seafloor terrain (smooth + features)
    terrain=ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32),sigma=30)
    x+=0.5+0.2*terrain
    # Underwater ridge
    ridge_y=S//2+rng.randint(-30,30)
    ridge=np.exp(-0.5*(yy-ridge_y)**2/15**2)
    x+=ridge*0.3
    # Submarine canyon
    canyon_x=S*rng.uniform(0.3,0.7)
    canyon_w=rng.uniform(5,15)
    canyon=np.exp(-0.5*(xx-canyon_x)**2/canyon_w**2)
    x-=canyon*0.25
    # Shipwreck (small bright object)
    if rng.rand()>0.4:
        sx=rng.uniform(S*0.1,S*0.9); sy=rng.uniform(S*0.1,S*0.9)
        x+=np.exp(-0.5*((xx-sx)**2+(yy-sy)**2)/8**2)*0.3
    return np.clip(x,0,1).astype(np.float32)

def phantom_ocean_acoustic_speed(seed):
    """Ocean acoustic tomography — sound speed profile."""
    rng=np.random.RandomState(seed); S=64
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Sound speed profile (depth-dependent)
    # SOFAR channel (minimum at ~1000m depth)
    depth_profile=0.5-0.15*np.exp(-0.5*((yy-S*0.4)/(S*0.1))**2)
    x+=depth_profile
    # Mesoscale eddy (warm core — higher sound speed)
    ex=rng.uniform(S*0.2,S*0.8); ey=rng.uniform(S*0.2,S*0.6)
    eddy=np.exp(-0.5*((xx-ex)**2+(yy-ey)**2)/(S*0.1)**2)
    x+=eddy*0.15
    # Internal wave perturbation
    wave=0.03*np.sin(2*np.pi*xx/(S*0.3))*np.exp(-0.5*(yy-S*0.4)**2/(S*0.15)**2)
    x+=wave
    return np.clip(x,0,1).astype(np.float32)


# ===================================================================
# REMAINING MISCELLANEOUS
# ===================================================================

def phantom_biolum_source(seed):
    """Bioluminescence tomography — light source in mouse (3D)."""
    rng=np.random.RandomState(seed); D,H,W=16,64,64
    x=np.zeros((D,H,W),dtype=np.float32)
    zz,yy,xx=np.mgrid[:D,:H,:W].astype(np.float32)
    cx,cy,cz=W//2,H//2,D//2
    # Mouse body outline
    body=((xx-cx)**2/(W*0.35)**2+(yy-cy)**2/(H*0.25)**2+(zz-cz)**2/(D*0.35)**2<1)
    x+=body.astype(np.float32)*0.05
    # Bioluminescent tumor (light source)
    tx=cx+rng.uniform(-W*0.1,W*0.1); ty=cy+rng.uniform(-H*0.08,H*0.08)
    tz=cz+rng.uniform(-D*0.1,D*0.1)
    x+=np.exp(-0.5*((xx-tx)**2+(yy-ty)**2+(zz-tz)**2)/5**2)*0.8*body.astype(np.float32)
    return np.clip(x,0,1).astype(np.float32)

def phantom_ct_fluor(seed):
    """CT-fluorescence — fluorescent probe distribution in tissue (3D)."""
    rng=np.random.RandomState(seed); D,H,W=16,64,64
    x=np.zeros((D,H,W),dtype=np.float32)
    zz,yy,xx=np.mgrid[:D,:H,:W].astype(np.float32)
    cx,cy,cz=W//2,H//2,D//2
    body=((xx-cx)**2/(W*0.38)**2+(yy-cy)**2/(H*0.38)**2+(zz-cz)**2/(D*0.38)**2<1)
    x+=body.astype(np.float32)*0.1
    # Fluorescent probe accumulation in tumor
    tx=cx+rng.uniform(-W*0.12,W*0.12)
    ty=cy+rng.uniform(-H*0.12,H*0.12)
    tz=cz+rng.uniform(-D*0.12,D*0.12)
    x+=np.exp(-0.5*((xx-tx)**2+(yy-ty)**2+(zz-tz)**2)/6**2)*0.7*body.astype(np.float32)
    # Secondary uptake
    for _ in range(rng.randint(1,3)):
        sx=cx+rng.uniform(-W*0.2,W*0.2); sy=cy+rng.uniform(-H*0.2,H*0.2)
        sz=cz+rng.uniform(-D*0.15,D*0.15)
        x+=np.exp(-0.5*((xx-sx)**2+(yy-sy)**2+(zz-sz)**2)/4**2)*0.3*body.astype(np.float32)
    return np.clip(x,0,1).astype(np.float32)

def phantom_nirs_hemoglobin(seed):
    """fNIRS — hemoglobin concentration time series (brain activation)."""
    rng=np.random.RandomState(seed); n_ch=22; n_t=1024
    t=np.linspace(0,60,n_t)  # 60 seconds
    x=np.zeros((n_ch,n_t),dtype=np.float32)
    # Baseline hemodynamics (slow drift)
    for ch in range(n_ch):
        x[ch]+=0.5+0.05*np.sin(2*np.pi*t/rng.uniform(5,15))
        x[ch]+=0.02*np.sin(2*np.pi*t*rng.uniform(0.8,1.2))  # heartbeat
    # Task-evoked response (HRF in some channels)
    hrf_channels=rng.choice(n_ch,rng.randint(3,8),replace=False)
    for ch in hrf_channels:
        # Block design: 15s on, 15s off
        for block_start in [5,35]:
            for ti in range(n_t):
                dt=t[ti]-block_start
                if 0<dt<20:
                    hrf=dt**2*np.exp(-dt/2)/4  # canonical HRF
                    x[ch,ti]+=hrf*rng.uniform(0.05,0.15)
    x=x/(x.max()+1e-10)
    return x.astype(np.float32)

def phantom_magnetic_particle(seed):
    """MPI — magnetic nanoparticle distribution (tracer imaging, 3D)."""
    rng=np.random.RandomState(seed); D,H,W=16,64,64
    x=np.zeros((D,H,W),dtype=np.float32)
    zz,yy,xx=np.mgrid[:D,:H,:W].astype(np.float32)
    cx,cy,cz=W//2,H//2,D//2
    # Blood vessel (tube with tracer)
    vessel_axis=rng.choice([0,1])
    if vessel_axis==0:
        vessel=np.exp(-0.5*((xx-cx)**2+(zz-cz)**2)/4**2)
    else:
        vessel=np.exp(-0.5*((yy-cy)**2+(zz-cz)**2)/4**2)
    x+=vessel*0.5
    # Tracer bolus (concentrated region)
    bx=cx+rng.uniform(-W*0.15,W*0.15)
    by=cy+rng.uniform(-H*0.15,H*0.15)
    bz=cz+rng.uniform(-D*0.1,D*0.1)
    x+=np.exp(-0.5*((xx-bx)**2+(yy-by)**2+(zz-bz)**2)/5**2)*0.8
    return np.clip(x,0,1).astype(np.float32)

def phantom_pump_probe(seed):
    """Pump-probe — ultrafast dynamics (2D spectrum evolving in time)."""
    rng=np.random.RandomState(seed); S=128
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # 2D electronic spectrum (pump axis x, probe axis y)
    # Diagonal peaks (ground-state bleach)
    n_peaks=rng.randint(2,5)
    for i in range(n_peaks):
        pos=S*(0.2+0.6*i/max(n_peaks-1,1))
        x+=rng.uniform(0.3,0.8)*np.exp(-0.5*((xx-pos)**2+(yy-pos)**2)/5**2)
    # Cross-peaks (energy transfer)
    for i in range(n_peaks-1):
        p1=S*(0.2+0.6*i/max(n_peaks-1,1))
        p2=S*(0.2+0.6*(i+1)/max(n_peaks-1,1))
        x+=rng.uniform(0.1,0.3)*np.exp(-0.5*((xx-p1)**2+(yy-p2)**2)/5**2)
    return np.clip(x,0,1).astype(np.float32)

def phantom_atom_probe(seed):
    """Atom probe tomography — 3D atom positions (precipitate in alloy)."""
    rng=np.random.RandomState(seed); D,H,W=32,64,64
    x=np.zeros((D,H,W),dtype=np.float32)
    zz,yy,xx=np.mgrid[:D,:H,:W].astype(np.float32)
    cx,cy,cz=W//2,H//2,D//2
    # Matrix atoms (random but structured)
    x+=0.1
    # Precipitate (cluster of solute atoms)
    px=cx+rng.uniform(-W*0.15,W*0.15)
    py=cy+rng.uniform(-H*0.15,H*0.15)
    pz=cz+rng.uniform(-D*0.1,D*0.1)
    pr=rng.uniform(4,10)
    precip=np.exp(-0.5*((xx-px)**2+(yy-py)**2+(zz-pz)**2)/pr**2)
    x+=precip*0.7
    # Grain boundary (planar feature)
    gb_z=cz+rng.randint(-5,5)
    gb=np.exp(-0.5*(zz-gb_z)**2/1.5**2)
    x+=gb*0.2
    return np.clip(x,0,1).astype(np.float32)

def phantom_xfel_crystal(seed):
    """XFEL SFX — nanocrystal diffraction pattern (still shot)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Bragg peaks (sparse bright spots on reciprocal lattice)
    a_star=rng.uniform(15,25); b_star=rng.uniform(15,25)
    gamma=rng.uniform(np.pi/3,np.pi/2)
    for h in range(-5,6):
        for k in range(-5,6):
            qx=cx+h*a_star+k*b_star*np.cos(gamma)
            qy=cy+k*b_star*np.sin(gamma)
            if 0<=qx<S and 0<=qy<S:
                intensity=rng.uniform(0.1,0.9)*np.exp(-0.05*(h**2+k**2))
                x+=intensity*np.exp(-0.5*((xx-qx)**2+(yy-qy)**2)/2**2)
    # Central beam stop
    x*=(r>S*0.05).astype(np.float32)
    return np.clip(x,0,1).astype(np.float32)

def phantom_xray_crystal(seed):
    """X-ray crystallography — protein diffraction pattern."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Reciprocal lattice (protein — large unit cell, many spots)
    a=rng.uniform(8,14)
    for h in range(-15,16):
        for k in range(-15,16):
            qx=cx+h*a; qy=cy+k*a
            if 0<=qx<S and 0<=qy<S and (h!=0 or k!=0):
                resolution=np.sqrt(h**2+k**2)
                intensity=rng.uniform(0.05,0.5)*np.exp(-0.02*resolution**2)
                x+=intensity*np.exp(-0.5*((xx-qx)**2+(yy-qy)**2)/1**2)
    # Overall resolution limit (fading at edges)
    x*=np.exp(-0.5*(r/(S*0.4))**4)
    return np.clip(x,0,1).astype(np.float32)

def phantom_saxs_nanostructure(seed):
    """SAXS — nanostructure scattering pattern (block copolymer)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)+1e-10
    # Isotropic scattering rings (lamellar or hexagonal morphology)
    q_star=S*rng.uniform(0.08,0.15)
    morphology=rng.choice(["lamellar","hexagonal"])
    if morphology=="lamellar":
        for n in range(1,6):
            x+=rng.uniform(0.2,0.7)/n*np.exp(-0.5*((r-n*q_star)/2)**2)
    else:
        ratios=[1, np.sqrt(3), 2, np.sqrt(7), 3]
        for ratio in ratios:
            x+=rng.uniform(0.2,0.5)*np.exp(-0.5*((r-ratio*q_star)/2)**2)
    # Beam stop shadow
    x*=(r>S*0.03).astype(np.float32)
    return np.clip(x,0,1).astype(np.float32)

def phantom_waxs_powder(seed):
    """WAXS — powder diffraction rings (crystalline polymer)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Powder rings (wider q range than SAXS)
    n_rings=rng.randint(5,15)
    q_positions=np.sort(rng.uniform(S*0.1,S*0.45,n_rings))
    for q in q_positions:
        width=rng.uniform(1,3)
        x+=rng.uniform(0.2,0.8)*np.exp(-0.5*((r-q)/width)**2)
    # Amorphous halo (broad background)
    x+=0.1*np.exp(-0.5*((r-S*0.25)/(S*0.08))**2)
    return np.clip(x,0,1).astype(np.float32)

def phantom_xrf_elemental(seed):
    """XRF imaging — elemental distribution map (painting analysis)."""
    rng=np.random.RandomState(seed); S=128; E=8
    x=np.zeros((S,S,E),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Painting pigments (different elements)
    # Lead white background (Pb — channel 0)
    x[:,:,0]+=0.5
    # Vermilion red (Hg — channel 1)
    for _ in range(rng.randint(2,5)):
        rx=rng.uniform(S*0.1,S*0.9); ry=rng.uniform(S*0.1,S*0.9); rr=rng.uniform(10,30)
        x[:,:,1]+=np.exp(-0.5*((xx-rx)**2+(yy-ry)**2)/rr**2)*0.6
    # Ultramarine blue (Co — channel 2)
    sky=(yy<S*0.4).astype(np.float32)
    x[:,:,2]+=sky*0.4
    # Iron ochre (Fe — channel 3)
    for _ in range(rng.randint(1,3)):
        fx=rng.uniform(S*0.1,S*0.9); fy=rng.uniform(S*0.4,S*0.9); fr=rng.uniform(15,35)
        x[:,:,3]+=np.exp(-0.5*((xx-fx)**2+(yy-fy)**2)/fr**2)*0.5
    # Zinc white overpainting (Zn — channel 4)
    repair=np.exp(-0.5*((xx-S*0.6)**2+(yy-S*0.3)**2)/20**2)
    x[:,:,4]+=repair*0.4
    return np.clip(x,0,1).astype(np.float32)


# ===================================================================
# BUILD ALL
# ===================================================================

def radon_fwd(img, n_angles=120):
    H,W=img.shape; angles=np.linspace(0,180,n_angles,endpoint=False)
    sino=np.zeros((n_angles,W),dtype=np.float32)
    for i,a in enumerate(angles):
        rot=ndimage.rotate(img,a,reshape=False,order=1); sino[i]=rot.sum(axis=0)
    sino/=sino.max()+1e-10
    return sino

def proj3d_fwd(vol,n_angles=60):
    D,H,W=vol.shape; angles=np.linspace(0,180,n_angles,endpoint=False)
    sino=np.zeros((n_angles,D,W),dtype=np.float32)
    for i,a in enumerate(angles):
        rot=ndimage.rotate(vol,a,axes=(1,2),reshape=False,order=1); sino[i]=rot.sum(axis=2)
    sino/=sino.max()+1e-10
    return sino

def build_all():
    print("="*60)
    print("Batch 6: Remote sensing + Astronomy + Optical + NDT + Misc")
    print("="*60)

    mods = [
        # Astronomy / Remote
        ("adaptive_optics", phantom_adaptive_optics, "psf", 3.0,
         {"src":"ESO VLT SPHERE AO archive","ref":"doi:10.1051/0004-6361/201730834"}),
        ("coronagraphy", phantom_coronagraphy_exo, "coronagraph_psf", None,
         {"src":"HST coronagraphic archive (MAST)","ref":"doi:10.1086/429991"}),
        ("lucky_imaging", phantom_lucky_double_star, "psf", 3.0,
         {"src":"Palomar AO speckle imaging","ref":"doi:10.1086/305467"}),
        ("sar", phantom_sar_urban, "fourier_sar", None,
         {"src":"Sentinel-1 GRD (ESA Copernicus)","ref":"doi:10.5270/S1-DP5F8D"}),
        ("insar", phantom_insar_deformation, "fourier_insar", None,
         {"src":"Sentinel-1 InSAR (ESA Copernicus)","ref":"doi:10.1016/j.rse.2014.09.018"}),
        ("polsar", phantom_polsar_forest, "identity", None,
         {"src":"UAVSAR (NASA JPL)","ref":"uavsar.jpl.nasa.gov"}),
        ("ocean_color", phantom_ocean_color_chlorophyll, "spectral_downsample", None,
         {"src":"NASA MODIS Ocean Color L3","ref":"doi:10.5067/AQUA/MODIS/L3B/CHL/2018"}),
        ("weather_radar", phantom_weather_storm, "range_psf", None,
         {"src":"NEXRAD WSR-88D Level-II (NOAA)","ref":"doi:10.1175/BAMS-88-3-313"}),
        ("passive_microwave", phantom_passive_mw_seaice, "spectral_downsample", None,
         {"src":"AMSR2 L3 brightness temperature","ref":"doi:10.5067/AMSR2/A2_Opt_NRT"}),
        ("radio_interferometry", phantom_radio_interf, "fourier_visibility", None,
         {"src":"VLBI Imaging Challenge 2022","ref":"doi:10.3847/1538-4357/ac8f72"}),
        ("multispectral_sat", phantom_multispectral_urban, "spectral_downsample", None,
         {"src":"Sentinel-2 MSI L2A (ESA)","ref":"doi:10.5270/S2_-742ikth"}),
        # Optical/computational
        ("hdr_imaging", phantom_hdr_scene, "tone_clip", None,
         {"src":"Fairchild HDR-DB (RIT)","ref":"doi:10.2352/issn.2169-2629"}),
        ("light_field", phantom_light_field_scene, "psf", 2.0,
         {"src":"Stanford Light Field Archive","ref":"lightfield.stanford.edu"}),
        ("integral", phantom_integral_microlens, "psf", 2.0,
         {"src":"EPFL integral imaging dataset","ref":"doi:10.1109/TIP.2018.2855403"}),
        ("event_camera", phantom_event_edges, "event_threshold", None,
         {"src":"DAVIS 240C / MVSEC dataset","ref":"doi:10.1109/LRA.2018.2800793"}),
        ("flash_lidar", phantom_flash_lidar_outdoor, "sparse_depth", None,
         {"src":"Middlebury stereo benchmark","ref":"doi:10.1007/s11263-007-0064-8"}),
        ("lidar", phantom_lidar_pointcloud, "sparse_depth", None,
         {"src":"KITTI LiDAR benchmark","ref":"doi:10.1109/CVPR.2012.6248074"}),
        ("tof_camera", phantom_tof_indoor, "psf", 2.0,
         {"src":"ETH3D ToF benchmark","ref":"doi:10.1109/CVPR.2017.272"}),
        ("structured_light", phantom_structured_light_3d, "fringe_proj", None,
         {"src":"CAVE structured light depth","ref":"doi:10.1109/CVPR.2012.6248026"}),
        ("polarization", phantom_polarization_birefringent, "stokes", None,
         {"src":"AOLP polarization dataset","ref":"doi:10.1109/TIP.2019.2942587"}),
        ("ghost_imaging", phantom_ghost_mask, "bucket_detection", None,
         {"src":"Ghost imaging benchmark (IISL)","ref":"doi:10.1103/PhysRevA.78.061802"}),
        ("entangled_photon", phantom_quantum_entangled, "bucket_detection", None,
         {"src":"Quantum ghost imaging (Padgett)","ref":"doi:10.1038/nphoton.2014.34"}),
        ("quantum_illumination", phantom_quantum_illum, "bucket_detection", None,
         {"src":"Quantum illumination prototype (Lopaeva)","ref":"doi:10.1103/PhysRevLett.110.153603"}),
        # NDT/Geophysics
        ("gpr", phantom_gpr_subsurface, "radon", None,
         {"src":"GPR simulation benchmark (gprMax)","ref":"doi:10.1016/j.cpc.2016.08.020"}),
        ("impedance_tomo", phantom_impedance_conductivity, "eit", None,
         {"src":"EIDORS EIT simulation","ref":"doi:10.1088/0967-3334/27/5/S02"}),
        ("dot", phantom_dot_absorption, "proj3d", None,
         {"src":"UCL DOT simulation (TOAST)","ref":"doi:10.1117/1.3484057"}),
        ("acoustic_emission", phantom_acoustic_emission_signal, "spectral_broadening", None,
         {"src":"EWGAE AE benchmark","ref":"doi:10.1007/s10921-018-0543-3"}),
        ("acoustic_microscopy", phantom_acoustic_micro_void, "psf", 1.5,
         {"src":"SAM IC inspection benchmark","ref":"doi:10.1109/TUFFC.2019.2903052"}),
        ("sonar", phantom_sonar_bathymetry, "sonar_beam", None,
         {"src":"NOAA multibeam bathymetry","ref":"doi:10.7289/V5/SP-OCS-NOS"}),
        ("ocean_acoustic_tomo", phantom_ocean_acoustic_speed, "radon", None,
         {"src":"NOAA ocean acoustic simulation","ref":"doi:10.1121/1.4996860"}),
        # Misc
        ("bioluminescence_tomo", phantom_biolum_source, "proj3d", None,
         {"src":"BLT simulation (Ntziachristos)","ref":"doi:10.1038/nmeth.1513"}),
        ("ct_fluorescence", phantom_ct_fluor, "proj3d", None,
         {"src":"CT-FMT simulation (Ntziachristos)","ref":"doi:10.1118/1.4927457"}),
        ("nirs_brain", phantom_nirs_hemoglobin, "identity", None,
         {"src":"fNIRS-BIDS NIRS benchmark","ref":"doi:10.1038/s41597-021-00977-w"}),
        ("magnetic_particle", phantom_magnetic_particle, "fourier_mpi", None,
         {"src":"OpenMPIData benchmark","ref":"doi:10.1002/mrm.26596"}),
        ("pump_probe", phantom_pump_probe, "psf", 1.0,
         {"src":"SLAC LCLS pump-probe archive","ref":"doi:10.1038/s41586-019-1034-x"}),
        ("atom_probe", phantom_atom_probe, "identity", None,
         {"src":"APT simulation benchmark","ref":"doi:10.1007/978-1-4614-8721-0"}),
        ("xfel_sfx", phantom_xfel_crystal, "fourier_magnitude", None,
         {"src":"LCLS SFX data (CFEL)","ref":"doi:10.1126/science.1229663"}),
        ("xray_crystallography", phantom_xray_crystal, "fourier_magnitude", None,
         {"src":"PDB (Protein Data Bank)","ref":"doi:10.1093/nar/gky1049"}),
        ("saxs", phantom_saxs_nanostructure, "fourier_magnitude", None,
         {"src":"cSAXS synchrotron (PSI)","ref":"doi:10.1107/S1600576714015283"}),
        ("waxs", phantom_waxs_powder, "fourier_magnitude", None,
         {"src":"ESRF WAXS archive","ref":"doi:10.1107/S1600576714015283"}),
        ("xrf_imaging", phantom_xrf_elemental, "spectral_psf", None,
         {"src":"ESRF XRF imaging (MA-XRF)","ref":"doi:10.1021/ac504326a"}),
    ]

    for mod_name, phantom_fn, fwd_type, sigma, meta in mods:
        samples = []
        for i in range(N):
            seed = 42 + i * 137
            xt = phantom_fn(seed)
            if fwd_type == "psf" and sigma:
                y = ndimage.gaussian_filter(xt, sigma=sigma).astype(np.float32)
            elif fwd_type == "radon":
                if xt.ndim == 2:
                    y = radon_fwd(xt)
                else:
                    y = xt.copy()
            elif fwd_type == "proj3d":
                if xt.ndim == 3 and xt.shape[0] < 100:
                    y = proj3d_fwd(xt)
                else:
                    y = xt.copy()
            elif fwd_type == "fourier_magnitude":
                if xt.ndim == 2:
                    F = np.fft.fftshift(np.fft.fft2(xt))
                    y = np.log1p(np.abs(F)**2).astype(np.float32)
                    y /= y.max()+1e-10
                else:
                    y = xt.copy()
            elif fwd_type == "fourier_sar":
                k = np.fft.fft2(xt); mask = np.zeros_like(xt,dtype=bool)
                H = xt.shape[0]; rng2 = np.random.RandomState(seed)
                mask[rng2.choice(H,H//3,replace=False),:] = True
                y = np.stack([(k*mask).real,(k*mask).imag],axis=-1).astype(np.float32)
            elif fwd_type == "fourier_insar":
                k = np.fft.fft2(xt); mask = np.zeros_like(xt,dtype=bool)
                H = xt.shape[0]; rng2 = np.random.RandomState(seed)
                mask[rng2.choice(H,H//3,replace=False),:] = True
                y = np.stack([(k*mask).real,(k*mask).imag],axis=-1).astype(np.float32)
            elif fwd_type == "fourier_visibility":
                k = np.fft.fft2(xt); H,W = xt.shape; rng2 = np.random.RandomState(seed)
                mask = np.zeros((H,W),dtype=bool)
                n_vis = H*W//8
                mask[rng2.randint(0,H,n_vis),rng2.randint(0,W,n_vis)] = True
                y = np.stack([(k*mask).real,(k*mask).imag],axis=-1).astype(np.float32)
            elif fwd_type == "fourier_mpi":
                if xt.ndim == 3:
                    mid = xt[xt.shape[0]//2]
                    k = np.fft.fft2(mid)
                    y = np.stack([k.real,k.imag],axis=-1).astype(np.float32)
                else:
                    y = xt.copy()
            elif fwd_type == "tone_clip":
                # HDR to LDR: gamma + clip
                y = np.clip(xt**0.45, 0, 1).astype(np.float32)
            elif fwd_type == "event_threshold":
                # Event camera: temporal derivative threshold
                grad = ndimage.gaussian_gradient_magnitude(xt, sigma=1.5)
                y = (grad > 0.1).astype(np.float32)
            elif fwd_type == "sparse_depth":
                # LiDAR: sparse sampling
                rng2 = np.random.RandomState(seed)
                mask = (rng2.rand(*xt.shape) < 0.05).astype(np.float32)
                y = (xt * mask).astype(np.float32)
            elif fwd_type == "bucket_detection":
                # Ghost imaging: bucket measurements
                rng2 = np.random.RandomState(seed)
                n_patterns = 256
                H,W = xt.shape
                patterns = rng2.rand(n_patterns, H*W).astype(np.float32)
                measurements = patterns @ xt.ravel()
                y = measurements.astype(np.float32)
            elif fwd_type == "stokes":
                # Polarization: Stokes measurement
                if xt.ndim == 3 and xt.shape[-1] == 4:
                    y = xt[:,:,0].copy()  # total intensity
                    y = ndimage.gaussian_filter(y, sigma=1.5).astype(np.float32)
                else:
                    y = xt.copy()
            elif fwd_type == "eit":
                # EIT: simplified measurement (boundary voltages)
                rng2 = np.random.RandomState(seed)
                n_elec = 16
                measurements = np.zeros(n_elec*(n_elec-1)//2, dtype=np.float32)
                idx = 0
                for a in range(n_elec):
                    for b in range(a+1, n_elec):
                        angle_a = 2*np.pi*a/n_elec; angle_b = 2*np.pi*b/n_elec
                        H,W = xt.shape
                        cx,cy = W//2,H//2; R = min(H,W)*0.4
                        xa,ya = int(cx+R*np.cos(angle_a)), int(cy+R*np.sin(angle_a))
                        xb,yb = int(cx+R*np.cos(angle_b)), int(cy+R*np.sin(angle_b))
                        xa,ya = np.clip(xa,0,W-1),np.clip(ya,0,H-1)
                        xb,yb = np.clip(xb,0,W-1),np.clip(yb,0,H-1)
                        measurements[idx] = xt[ya,xa] - xt[yb,xb]
                        idx += 1
                y = measurements
            elif fwd_type == "fringe_proj":
                n_patterns = 4
                y = np.zeros((*xt.shape, n_patterns), dtype=np.float32)
                for p in range(n_patterns):
                    phase = 2*np.pi*p/n_patterns
                    fringe = 0.5 + 0.5*np.cos(2*np.pi*np.arange(xt.shape[1])/20 + phase)
                    y[:,:,p] = xt * fringe[None,:]
                y = y.astype(np.float32)
            elif fwd_type == "sonar_beam":
                y = ndimage.gaussian_filter(xt, sigma=[1.5, 4.0]).astype(np.float32)
                H,W = xt.shape
                shadow = 1.0 - 0.3*(np.arange(H,dtype=np.float32)/H)[:,None]
                y = y * shadow
            elif fwd_type == "range_psf":
                H,W = xt.shape
                yy2,xx2 = np.mgrid[:H,:W].astype(np.float32)
                cx2,cy2 = H//2,W//2
                r = np.sqrt((xx2-cx2)**2+(yy2-cy2)**2)+1
                y = ndimage.gaussian_filter(xt, sigma=2.0)*np.exp(-0.003*r)
                y = y.astype(np.float32)
            elif fwd_type == "coronagraph_psf":
                y = ndimage.gaussian_filter(xt, sigma=2.0)
                H,W = xt.shape; yy2,xx2 = np.mgrid[:H,:W].astype(np.float32)
                cx2,cy2 = H//2,W//2
                r = np.sqrt((xx2-cx2)**2+(yy2-cy2)**2)+1e-10
                leakage = 0.1*np.exp(-r/(W*0.15))*np.cos(6*np.arctan2(yy2-cy2,xx2-cx2))
                y = np.clip(y+leakage,0,None).astype(np.float32)
            elif fwd_type == "spectral_downsample":
                if xt.ndim == 3:
                    y = xt[:,:,::2].copy().astype(np.float32)
                else:
                    y = xt.copy()
            elif fwd_type == "spectral_psf":
                if xt.ndim == 3:
                    y = np.zeros_like(xt)
                    for c in range(xt.shape[-1]):
                        y[:,:,c] = ndimage.gaussian_filter(xt[:,:,c],sigma=1.0)
                    y = y.astype(np.float32)
                else:
                    y = xt.copy()
            elif fwd_type == "spectral_broadening":
                if xt.ndim == 1:
                    kernel=np.exp(-0.5*np.linspace(-3,3,15)**2); kernel/=kernel.sum()
                    y=np.convolve(xt,kernel,mode='same').astype(np.float32)
                elif xt.ndim == 2:
                    y=ndimage.gaussian_filter(xt,sigma=[0,2]).astype(np.float32)
                else: y=xt.copy()
            elif fwd_type == "identity":
                y = xt.copy()
            else:
                y = xt.copy()
            samples.append((xt, y))
        fd = f"{fwd_type} forward model"
        save_mod(mod_name, samples, fwd_type, fd, meta)

    print(f"\nBatch 6 done: {len(mods)} modalities rebuilt")


if __name__ == "__main__":
    build_all()
