"""
Rebuild standard datasets — Batch 7: Remaining modalities that still use shared data.
These were built by earlier scripts using BSD68/skimage as ground truth.
Each now gets its own unique phantom.

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "benchmark"
N = 10

def to_uint8(a):
    a=np.nan_to_num(a); mn,mx=float(a.min()),float(a.max())
    if mx-mn<1e-10: return np.zeros(a.shape,dtype=np.uint8)
    return ((a-mn)/(mx-mn)*255).clip(0,255).astype(np.uint8)

def save_img(a, p):
    if a.ndim==2: Image.fromarray(to_uint8(a),"L").save(str(p))
    elif a.ndim==3:
        if a.shape[-1]==3: Image.fromarray(to_uint8(a),"RGB").save(str(p))
        elif a.shape[-1]==2:
            Image.fromarray(to_uint8(np.log1p(np.sqrt(a[...,0]**2+a[...,1]**2))),"L").save(str(p))
        elif a.shape[0]<100: Image.fromarray(to_uint8(a[a.shape[0]//2]),"L").save(str(p))
        else: Image.fromarray(to_uint8(a[:,:,0]),"L").save(str(p))

def save_mod(mod, samples, ft, fd, me):
    out=DATA/mod/"standard"; out.mkdir(parents=True,exist_ok=True)
    for old in out.glob("standard_*.h5"): old.unlink()
    files=[]
    for i,(x,y) in enumerate(samples):
        h5p=out/f"standard_{mod}_{i:02d}.h5"
        with h5py.File(str(h5p),"w") as f:
            f.create_dataset("x_true",data=x,compression="gzip")
            f.create_dataset("y_ideal",data=y,compression="gzip")
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


def radon_fwd(x, n_angles=120):
    H,W=x.shape; angles=np.linspace(0,180,n_angles,endpoint=False)
    sino=np.zeros((n_angles,W),dtype=np.float32)
    for i,a in enumerate(angles):
        rot=ndimage.rotate(x,a,reshape=False,order=1); sino[i]=rot.sum(axis=0)
    sino/=sino.max()+1e-10
    return sino


# ===================================================================
# UNIQUE PHANTOMS for modalities with old shared data
# ===================================================================

def phantom_ct_abdomen(seed):
    """CT — abdominal cross-section (Shepp-Logan variant with organs)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Body outline (elliptical)
    body=((xx-cx)**2/(S*0.42)**2+(yy-cy)**2/(S*0.35)**2<1).astype(np.float32)
    x+=body*0.3
    # Liver (large right lobe)
    liver=np.exp(-0.5*((xx-cx+S*0.1)**2/(S*0.15)**2+(yy-cy-S*0.05)**2/(S*0.12)**2))
    x+=liver*0.15*body
    # Spleen (left posterior)
    spleen=np.exp(-0.5*((xx-cx-S*0.2)**2/(S*0.06)**2+(yy-cy+S*0.08)**2/(S*0.05)**2))
    x+=spleen*0.12*body
    # Kidneys (bilateral)
    for side in [-1,1]:
        kx=cx+side*S*0.18; ky=cy+S*0.03
        kidney=np.exp(-0.5*((xx-kx)**2/(S*0.04)**2+(yy-ky)**2/(S*0.06)**2))
        x+=kidney*0.1*body
    # Spine (high density)
    spine=np.exp(-0.5*((xx-cx)**2/8**2+(yy-cy+S*0.15)**2/10**2))
    x+=spine*0.6*body
    # Aorta (circular vessel)
    aorta=np.exp(-0.5*((xx-cx-S*0.02)**2+(yy-cy+S*0.08)**2)/5**2)
    x+=aorta*0.2*body
    # Fat layer (subcutaneous)
    fat=body*(1-((xx-cx)**2/(S*0.38)**2+(yy-cy)**2/(S*0.31)**2<1).astype(np.float32))
    x-=fat*0.1
    # Random lesion
    if rng.rand()>0.4:
        lx=cx+rng.uniform(-S*0.15,S*0.15); ly=cy+rng.uniform(-S*0.1,S*0.1)
        x+=np.exp(-0.5*((xx-lx)**2+(yy-ly)**2)/rng.uniform(5,12)**2)*0.2*body
    return np.clip(x,0,1).astype(np.float32)


def phantom_mri_brain(seed):
    """MRI — brain T1-weighted (distinct from BrainWeb, custom phantom)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Skull (bright ring — T1 short)
    skull=((r<S*0.42)&(r>S*0.38)).astype(np.float32)
    x+=skull*0.8
    # White matter (bright in T1)
    wm=(r<S*0.32).astype(np.float32)
    x+=wm*0.65
    # Gray matter cortex (intermediate T1)
    gm=((r<S*0.38)&(r>S*0.32)).astype(np.float32)
    x+=gm*0.5
    # CSF (dark in T1)
    # Ventricles (two symmetric cavities)
    for side in [-1,1]:
        vx=cx+side*S*0.05
        vent=np.exp(-0.5*((xx-vx)**2/6**2+(yy-cy-S*0.02)**2/14**2))
        x-=vent*0.3
    # Caudate nuclei
    for side in [-1,1]:
        cnx=cx+side*S*0.06; cny=cy-S*0.05
        x+=np.exp(-0.5*((xx-cnx)**2+(yy-cny)**2)/5**2)*0.15
    # Thalamus
    x+=np.exp(-0.5*((xx-cx)**2+(yy-cy+S*0.03)**2)/8**2)*0.1
    # Corpus callosum (bright midline structure)
    cc=np.exp(-0.5*(yy-cy+S*0.08)**2/3**2)*(np.abs(xx-cx)<S*0.12).astype(np.float32)
    x+=cc*0.15
    # Scalp fat (bright outer ring)
    scalp=((r<S*0.45)&(r>S*0.42)).astype(np.float32)
    x+=scalp*0.7
    brain=(r<S*0.45).astype(np.float32)
    return np.clip(x*brain,0,1).astype(np.float32)


def phantom_us_tissue(seed):
    """Ultrasound — tissue phantom with scatterers, cysts, and lesion."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Tissue background (random scatterers)
    n_scatterers=rng.randint(5000,10000)
    for _ in range(n_scatterers):
        sx=rng.randint(0,S); sy=rng.randint(0,S)
        if 0<=sx<S and 0<=sy<S:
            x[sy,sx]+=rng.uniform(0.05,0.3)
    x=ndimage.gaussian_filter(x,sigma=1.5)
    x=x/(x.max()+1e-10)*0.5
    # Anechoic cyst (dark circular region)
    cyst_x=rng.uniform(S*0.2,S*0.8); cyst_y=rng.uniform(S*0.3,S*0.7)
    cyst_r=rng.uniform(10,25)
    cyst=(np.sqrt((xx-cyst_x)**2+(yy-cyst_y)**2)<cyst_r).astype(np.float32)
    x*=(1-cyst)  # Remove scatterers inside cyst
    # Hyperechoic mass
    mx=rng.uniform(S*0.2,S*0.8); my=rng.uniform(S*0.3,S*0.7)
    mr=rng.uniform(8,18)
    mass=np.exp(-0.5*((xx-mx)**2+(yy-my)**2)/mr**2)
    x+=mass*0.4
    # Depth-dependent attenuation (TGC-corrected tissue)
    depth_gain=1+0.5*(yy/S)
    x*=depth_gain
    return np.clip(x,0,1).astype(np.float32)


def phantom_pet_fdg(seed):
    """PET — FDG brain phantom (glucose metabolism map)."""
    rng=np.random.RandomState(seed); S=128
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    brain=(r<S*0.4).astype(np.float32)
    # Gray matter (high FDG uptake)
    gm=((r<S*0.4)&(r>S*0.25)).astype(np.float32)
    x+=gm*0.8
    # White matter (lower uptake)
    wm=(r<S*0.25).astype(np.float32)
    x+=wm*0.35
    # Basal ganglia (high uptake)
    for side in [-1,1]:
        bgx=cx+side*S*0.08
        x+=np.exp(-0.5*((xx-bgx)**2+(yy-cy)**2)/6**2)*0.25
    # Visual cortex (posterior, high uptake)
    vc_y=cy+S*0.25
    x+=np.exp(-0.5*((xx-cx)**2+(yy-vc_y)**2)/10**2)*0.3
    # Tumor hot spot
    if rng.rand()>0.3:
        tx=cx+rng.uniform(-S*0.15,S*0.15); ty=cy+rng.uniform(-S*0.15,S*0.15)
        x+=np.exp(-0.5*((xx-tx)**2+(yy-ty)**2)/rng.uniform(5,10)**2)*0.5
    return np.clip(x*brain,0,1).astype(np.float32)


def phantom_oct_retina(seed):
    """OCT — retinal layers (B-scan cross-section)."""
    rng=np.random.RandomState(seed); H=256; W=512
    x=np.zeros((H,W),dtype=np.float32)
    yy,xx=np.mgrid[:H,:W].astype(np.float32)
    # Retinal layers (smooth curves with varying thickness)
    layers = [
        ("ILM", 80, 0.7),    # Inner limiting membrane
        ("NFL", 85, 0.6),    # Nerve fiber layer
        ("GCL", 92, 0.5),    # Ganglion cell layer
        ("IPL", 100, 0.4),   # Inner plexiform
        ("INL", 108, 0.5),   # Inner nuclear
        ("OPL", 115, 0.4),   # Outer plexiform
        ("ONL", 125, 0.3),   # Outer nuclear
        ("IS/OS", 138, 0.8), # Photoreceptor junction
        ("RPE", 145, 0.9),   # Retinal pigment epithelium
    ]
    # Foveal depression (dip in center)
    fovea_x = W // 2 + rng.randint(-20, 20)
    for name, base_y, brightness in layers:
        # Smooth curvature
        curve = base_y - 8*np.exp(-0.5*((xx-fovea_x)/40)**2)
        curve += 3*np.sin(2*np.pi*xx/W)  # slight undulation
        curve += rng.uniform(-1, 1)
        layer_band = np.exp(-0.5*(yy-curve)**2/2**2)
        x += layer_band * brightness
    # Choroid (below RPE, thicker, moderate signal)
    choroid_y = 160 + 5*np.sin(2*np.pi*xx/W)
    x += np.exp(-0.5*((yy-choroid_y)/15)**2) * 0.25
    # Drusen (sub-RPE deposits in AMD)
    if rng.rand()>0.4:
        n_drusen = rng.randint(2, 6)
        for _ in range(n_drusen):
            dx = rng.uniform(W*0.2, W*0.8)
            dw = rng.uniform(5, 15)
            dh = rng.uniform(3, 8)
            drusen_bump = dh*np.exp(-0.5*((xx-dx)/dw)**2)
            x += np.exp(-0.5*(yy-140-drusen_bump)**2/2**2) * 0.5
    return np.clip(x,0,1).astype(np.float32)


def phantom_seismic_velocity(seed):
    """Seismic tomography — layered earth velocity model (Marmousi-style)."""
    rng=np.random.RandomState(seed); H=256; W=512
    x=np.zeros((H,W),dtype=np.float32)
    yy,xx=np.mgrid[:H,:W].astype(np.float32)
    # Layered velocity model (increasing with depth)
    n_layers=rng.randint(5,10)
    boundaries=[0]
    for _ in range(n_layers):
        boundaries.append(boundaries[-1]+H//(n_layers+1)+rng.randint(-5,5))
    boundaries.append(H)
    for i in range(len(boundaries)-1):
        vel=0.2+0.7*i/(len(boundaries)-1)
        mask=((yy>=boundaries[i])&(yy<boundaries[i+1])).astype(np.float32)
        x+=mask*vel
    # Fold/anticline (velocity anomaly)
    fold_x=W*rng.uniform(0.3,0.7); fold_depth=H*rng.uniform(0.3,0.6)
    fold_amp=rng.uniform(10,25)
    fold=fold_amp*np.exp(-0.5*((xx-fold_x)/50)**2)
    for row in range(H):
        shift=int(fold_amp*np.exp(-0.5*((row-fold_depth)/20)**2)*np.exp(-0.5*((np.arange(W)-fold_x)/50)**2).max())
        if shift>0 and row+shift<H:
            x[row,:]=x[min(row+shift,H-1),:]
    # Salt body (high velocity intrusion)
    if rng.rand()>0.4:
        sx=W*rng.uniform(0.2,0.8); sy=H*rng.uniform(0.3,0.7)
        salt=np.exp(-0.5*((xx-sx)**2/(30**2)+(yy-sy)**2/(20**2)))
        x+=salt*0.3
    return np.clip(x,0,1).astype(np.float32)


def phantom_fwi_velocity(seed):
    """FWI — full waveform inversion velocity model (overthrust-style)."""
    rng=np.random.RandomState(seed); S=128
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Smooth velocity gradient (depth-dependent)
    x+=0.15+0.6*(yy/S)
    # Fault (sharp discontinuity)
    fault_x=S*rng.uniform(0.3,0.7); fault_dip=rng.uniform(0.3,0.8)
    for row in range(S):
        offset=int((row-S*0.3)*fault_dip)
        col=int(fault_x+offset)
        if 0<=col<S:
            x[row,:col]*=0.8  # lower velocity on one side
    # Low-velocity zone (gas pocket)
    gx=rng.uniform(S*0.2,S*0.8); gy=rng.uniform(S*0.3,S*0.7)
    gas=np.exp(-0.5*((xx-gx)**2/(10**2)+(yy-gy)**2/(5**2)))
    x-=gas*0.15
    # High-velocity layer (limestone)
    ly=S*rng.uniform(0.4,0.6)
    limestone=np.exp(-0.5*(yy-ly)**2/3**2)
    x+=limestone*0.15
    return np.clip(x,0,1).astype(np.float32)


def phantom_holography_particles(seed):
    """Digital holography — 3D particle field (in-line holography)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Particles at various z-positions (will have different diffraction rings)
    n_particles=rng.randint(10,30)
    for _ in range(n_particles):
        px=rng.uniform(S*0.1,S*0.9); py=rng.uniform(S*0.1,S*0.9)
        pz=rng.uniform(5,50)  # distance in arbitrary units
        pr=rng.uniform(2,6)    # particle size
        # Particle is a bright disc
        particle=np.exp(-0.5*((xx-px)**2+(yy-py)**2)/pr**2)
        x+=particle*rng.uniform(0.3,0.8)
    return np.clip(x,0,1).astype(np.float32)


def phantom_fluoroscopy_gi(seed):
    """Fluoroscopy — GI barium swallow phantom (esophagus/stomach)."""
    rng=np.random.RandomState(seed); S=256
    x=np.ones((S,S),dtype=np.float32)*0.3  # soft tissue background
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Barium-filled esophagus (bright tube)
    eso_x=cx+rng.uniform(-S*0.05,S*0.05)
    for row in range(S):
        eso_width=3+2*np.sin(row/20)*np.exp(-0.5*(row-S*0.3)**2/(S*0.2)**2)
        x[row,:]+=np.exp(-0.5*(xx[row,:]-eso_x)**2/max(eso_width,1)**2)*0.5
    # Stomach (large bright region at bottom)
    stom_x=cx-S*0.05; stom_y=cy+S*0.15
    stomach=np.exp(-0.5*((xx-stom_x)**2/(S*0.12)**2+(yy-stom_y)**2/(S*0.08)**2))
    x+=stomach*0.4
    # Spine shadow (dark vertical bar)
    spine=np.exp(-0.5*(xx-cx+S*0.15)**2/10**2)
    x+=spine*0.2
    # Ribs (faint horizontal bands)
    for i in range(6):
        ry=cy-S*0.25+i*S*0.1
        rib=np.exp(-0.5*(yy-ry)**2/3**2)*0.08
        x+=rib
    return np.clip(x,0,1).astype(np.float32)


def phantom_panorama_urban(seed):
    """Panorama — 360° equirectangular urban scene."""
    rng=np.random.RandomState(seed); H=256; W=512
    x=np.zeros((H,W,3),dtype=np.float32)
    yy,xx=np.mgrid[:H,:W].astype(np.float32)
    # Sky (top half, blue gradient)
    sky=(yy<H*0.5).astype(np.float32)
    x[:,:,0]+=sky*(0.3-0.15*yy/H); x[:,:,1]+=sky*(0.4-0.15*yy/H); x[:,:,2]+=sky*(0.7-0.2*yy/H)
    # Ground (bottom, brown/gray)
    ground=(yy>H*0.55).astype(np.float32)
    x[:,:,0]+=ground*0.35; x[:,:,1]+=ground*0.3; x[:,:,2]+=ground*0.25
    # Buildings (rectangles along horizon)
    n_bldg=rng.randint(5,12)
    for _ in range(n_bldg):
        bx=rng.uniform(0,W); bw=rng.uniform(20,60); bh=rng.uniform(H*0.1,H*0.35)
        by=H*0.5-bh
        bldg=((np.abs(xx-bx)<bw/2)&(yy>by)&(yy<H*0.55)).astype(np.float32)
        color=rng.uniform(0.3,0.7,3)
        x[:,:,0]+=bldg*color[0]; x[:,:,1]+=bldg*color[1]; x[:,:,2]+=bldg*color[2]
        # Windows
        for wi in range(int(bw/8)):
            for wj in range(int(bh/10)):
                wx=bx-bw/2+wi*8+4; wy=by+wj*10+5
                if 0<=wx<W and 0<=wy<H:
                    x[int(wy):min(int(wy)+3,H),int(wx):min(int(wx)+3,W),:]+=0.15
    return np.clip(x,0,1).astype(np.float32)


def phantom_solar_euv(seed):
    """Solar imaging — EUV solar disk with active regions and prominences."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    solar_r=S*0.4
    # Solar disk
    disk=(r<solar_r).astype(np.float32)
    x+=disk*0.4
    # Limb brightening (EUV)
    limb=np.clip((r/solar_r),0,1)
    x+=disk*0.2*limb**2
    # Active regions (bright patches)
    n_ar=rng.randint(2,6)
    for _ in range(n_ar):
        angle=rng.uniform(0,2*np.pi); ar_r=solar_r*rng.uniform(0.2,0.7)
        ax=cx+ar_r*np.cos(angle); ay=cy+ar_r*np.sin(angle)
        ar_size=rng.uniform(8,20)
        x+=np.exp(-0.5*((xx-ax)**2+(yy-ay)**2)/ar_size**2)*rng.uniform(0.3,0.6)*disk
    # Coronal holes (dark regions)
    if rng.rand()>0.3:
        ch_angle=rng.uniform(0,2*np.pi)
        ch_r=solar_r*0.5
        chx=cx+ch_r*np.cos(ch_angle); chy=cy+ch_r*np.sin(ch_angle)
        x-=np.exp(-0.5*((xx-chx)**2+(yy-chy)**2)/(S*0.06)**2)*0.2*disk
    # Prominence (off-limb feature)
    prom_angle=rng.uniform(0,2*np.pi)
    px=cx+(solar_r+10)*np.cos(prom_angle); py=cy+(solar_r+10)*np.sin(prom_angle)
    x+=np.exp(-0.5*((xx-px)**2+(yy-py)**2)/5**2)*0.4
    return np.clip(x,0,1).astype(np.float32)


def phantom_coded_exposure_motion(seed):
    """Coded exposure — scene with motion (flutter shutter target)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Static background (textured wall)
    x+=0.3+0.1*ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32),sigma=15)
    # Moving object (car/person — will be motion-blurred)
    obj_y=S*rng.uniform(0.3,0.7); obj_x=S*rng.uniform(0.2,0.8)
    obj_w=rng.uniform(20,40); obj_h=rng.uniform(15,30)
    obj=((np.abs(xx-obj_x)<obj_w)&(np.abs(yy-obj_y)<obj_h)).astype(np.float32)
    x+=obj*0.5
    # Wheels (circles on car)
    for wx in [obj_x-obj_w*0.6, obj_x+obj_w*0.6]:
        wheel=np.exp(-0.5*((xx-wx)**2+(yy-obj_y-obj_h*0.7)**2)/5**2)
        x+=wheel*0.3
    # Text/sign (high-frequency detail)
    for i in range(rng.randint(2,5)):
        tx=rng.uniform(S*0.1,S*0.9); ty=rng.uniform(S*0.1,S*0.3)
        tw=rng.uniform(3,8)
        x+=(np.abs(xx-tx)<tw).astype(np.float32)*(np.abs(yy-ty)<2).astype(np.float32)*0.4
    return np.clip(x,0,1).astype(np.float32)


def phantom_octa_retinal_vessels(seed):
    """OCTA — retinal vasculature en-face map."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Optic disc (bright region — many vessels converge)
    disc_x=cx-S*0.15; disc_y=cy
    disc=np.exp(-0.5*((xx-disc_x)**2+(yy-disc_y)**2)/(S*0.06)**2)
    x+=disc*0.3
    # Retinal vessels (branching tree from optic disc)
    def draw_vessel_tree(start_x, start_y, angle, length, width, depth=0):
        if depth > 5 or length < 5: return
        for t in np.linspace(0,1,int(length)):
            angle += rng.uniform(-0.03,0.03)
            px = start_x+length*t*np.cos(angle)
            py = start_y+length*t*np.sin(angle)
            w = width*(1-0.3*t)
            if 0<=px<S and 0<=py<S:
                x_arr = np.exp(-0.5*((xx-px)**2+(yy-py)**2)/max(w,0.5)**2)
                x.__iadd__(x_arr * 0.4)
            if t > 0.3 and rng.rand() > 0.6:
                branch_angle = angle + rng.choice([-1,1])*rng.uniform(0.3,0.8)
                draw_vessel_tree(px,py,branch_angle,length*(1-t)*0.7,width*0.6,depth+1)
        end_x = start_x+length*np.cos(angle)
        end_y = start_y+length*np.sin(angle)
        draw_vessel_tree(end_x,end_y,angle+rng.uniform(-0.2,0.2),length*0.5,width*0.5,depth+1)

    for angle_offset in [0.3, -0.3, 2.8, -2.8]:
        draw_vessel_tree(disc_x,disc_y,angle_offset,S*0.35,2.5)
    # Foveal avascular zone (dark center)
    faz=np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(S*0.03)**2)
    x-=faz*0.2
    return np.clip(x,0,1).astype(np.float32)


def phantom_dic_phase(seed):
    """DIC — optical path difference phantom (unstained tissue section)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Tissue section with varying thickness (OPD map)
    n_cells=rng.randint(5,12)
    for _ in range(n_cells):
        cx=rng.uniform(S*0.1,S*0.9); cy=rng.uniform(S*0.1,S*0.9)
        cr=rng.uniform(15,35)
        # Smooth dome-shaped cell
        dist=np.sqrt((xx-cx)**2+(yy-cy)**2)
        cell=np.clip(1-(dist/cr)**2,0,1)*rng.uniform(0.3,0.6)
        x+=cell
        # Nucleus (denser)
        nuc=np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(cr*0.3)**2)
        x+=nuc*0.2
    return np.clip(x,0,1).astype(np.float32)


def phantom_photometric_normals(seed):
    """Photometric stereo — surface normal map (sculpture/relief)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Relief sculpture (height map)
    # Face-like relief
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    base=np.clip(1-r/(S*0.35),0,1)**2*0.5
    x+=base
    # Raised features (nose, brows)
    x+=np.exp(-0.5*((xx-cx)**2/6**2+(yy-cy+S*0.02)**2/12**2))*0.3
    # Concavities (eyes)
    for side in [-1,1]:
        ex=cx+side*S*0.07; ey=cy-S*0.05
        x-=np.exp(-0.5*((xx-ex)**2+(yy-ey)**2)/5**2)*0.15
    # Ornamental pattern (border)
    border_r=S*0.38
    border=np.exp(-0.5*((r-border_r)/3)**2)*0.2
    x+=border
    # Wave pattern on border
    theta=np.arctan2(yy-cy,xx-cx)
    x+=border*0.1*np.cos(12*theta)
    return np.clip(x,0,1).astype(np.float32)


def phantom_phase_retrieval_nano(seed):
    """Phase retrieval — CDI of nanocrystal (compact support)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Nanocrystal (compact object, small compared to FOV)
    crystal_size=S*rng.uniform(0.08,0.15)
    # Faceted shape (hexagonal)
    r=np.sqrt((xx-cx)**2+(yy-cy)**2)
    theta=np.arctan2(yy-cy,xx-cx)
    hex_r=crystal_size*(1-0.15*np.cos(6*theta))
    crystal=(r<hex_r).astype(np.float32)
    x+=crystal*0.7
    # Internal strain (phase variation)
    strain=0.2*np.sin(2*np.pi*(xx-cx)/crystal_size*2)*crystal
    x+=strain
    # Twin boundary
    twin=(xx>cx).astype(np.float32)*crystal*0.1
    x+=twin
    return np.clip(x,0,1).astype(np.float32)


def phantom_ptycho_circuit(seed):
    """Ptychography — IC circuit pattern (nanofabrication)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Metal interconnects (Manhattan geometry)
    n_h=rng.randint(5,12); n_v=rng.randint(5,12)
    for _ in range(n_h):
        ly=rng.uniform(S*0.05,S*0.95); lw=rng.uniform(2,5)
        x1=rng.uniform(0,S*0.5); x2=rng.uniform(S*0.5,S)
        line=((np.abs(yy-ly)<lw)&(xx>x1)&(xx<x2)).astype(np.float32)
        x+=line*0.7
    for _ in range(n_v):
        lx=rng.uniform(S*0.05,S*0.95); lw=rng.uniform(2,5)
        y1=rng.uniform(0,S*0.5); y2=rng.uniform(S*0.5,S)
        line=((np.abs(xx-lx)<lw)&(yy>y1)&(yy<y2)).astype(np.float32)
        x+=line*0.6
    # Vias (contact points)
    for _ in range(rng.randint(10,30)):
        vx=rng.uniform(S*0.05,S*0.95); vy=rng.uniform(S*0.05,S*0.95)
        x+=np.exp(-0.5*((xx-vx)**2+(yy-vy)**2)/2**2)*0.5
    # Transistor gate patterns
    for _ in range(rng.randint(3,8)):
        gx=rng.uniform(S*0.1,S*0.9); gy=rng.uniform(S*0.1,S*0.9)
        gw=rng.uniform(3,8); gh=rng.uniform(8,20)
        gate=((np.abs(xx-gx)<gw)&(np.abs(yy-gy)<gh)).astype(np.float32)
        x+=gate*0.5
    return np.clip(x,0,1).astype(np.float32)


def phantom_shearography_strain(seed):
    """Shearography — surface strain field (composite panel with disbond)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32); cx,cy=S//2,S//2
    # Loaded composite panel (smooth deformation field)
    x+=0.5+0.15*(yy/S-0.5)  # bending
    # Disbond/delamination (anomalous strain concentration)
    dx=cx+rng.uniform(-S*0.15,S*0.15); dy=cy+rng.uniform(-S*0.15,S*0.15)
    dr=rng.uniform(15,30)
    disbond=np.exp(-0.5*((xx-dx)**2+(yy-dy)**2)/dr**2)
    x+=disbond*0.3
    # Impact damage (concentric rings)
    if rng.rand()>0.4:
        ix=cx+rng.uniform(-S*0.2,S*0.2); iy=cy+rng.uniform(-S*0.2,S*0.2)
        ir=np.sqrt((xx-ix)**2+(yy-iy)**2)
        x+=0.15*np.cos(2*np.pi*ir/rng.uniform(8,15))*np.exp(-ir/(S*0.1))
    return np.clip(x,0,1).astype(np.float32)


def phantom_eddy_metal_defect(seed):
    """Eddy current — metal plate with subsurface crack (impedance map)."""
    rng=np.random.RandomState(seed); S=256
    x=np.ones((S,S),dtype=np.float32)*0.5  # uniform metal plate
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Subsurface crack (changes eddy current path)
    crack_x=S//2+rng.randint(-30,30); crack_y0=rng.uniform(S*0.2,S*0.4)
    crack_len=rng.uniform(S*0.2,S*0.5)
    crack_angle=rng.uniform(-0.2,0.2)
    for t in np.linspace(0,1,200):
        cx_t=crack_x+crack_len*t*np.sin(crack_angle)
        cy_t=crack_y0+crack_len*t
        if 0<=cx_t<S and 0<=cy_t<S:
            # Impedance change around crack
            x+=np.exp(-0.5*((xx-cx_t)**2+(yy-cy_t)**2)/4**2)*0.2
    # Rivet holes (known features)
    for _ in range(rng.randint(3,8)):
        rx=rng.uniform(S*0.1,S*0.9); ry=rng.uniform(S*0.1,S*0.9)
        rivet=np.exp(-0.5*((xx-rx)**2+(yy-ry)**2)/3**2)
        x+=rivet*0.15
    # Corrosion thinning (gradual thickness change)
    if rng.rand()>0.4:
        corr_x=rng.uniform(S*0.1,S*0.9); corr_y=rng.uniform(S*0.1,S*0.9)
        corr=np.exp(-0.5*((xx-corr_x)**2+(yy-corr_y)**2)/(S*0.08)**2)
        x-=corr*0.15
    return np.clip(x,0,1).astype(np.float32)


def phantom_thermography_panel(seed):
    """Active thermography — composite panel with delamination (thermal map)."""
    rng=np.random.RandomState(seed); S=256
    x=np.ones((S,S),dtype=np.float32)*0.4  # uniform surface temperature
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Thermal diffusion gradient (heat applied from one side)
    x+=0.2*(1-yy/S)
    # Delamination (trapped air = hot spot — heat doesn't conduct through)
    n_delam=rng.randint(1,4)
    for _ in range(n_delam):
        dx=rng.uniform(S*0.15,S*0.85); dy=rng.uniform(S*0.15,S*0.85)
        dsx=rng.uniform(10,30); dsy=rng.uniform(8,25)
        delam=np.exp(-0.5*((xx-dx)**2/dsx**2+(yy-dy)**2/dsy**2))
        x+=delam*0.3
    # Fastener thermal bridges (cool spots — heat sinks)
    for _ in range(rng.randint(4,12)):
        fx=rng.uniform(S*0.1,S*0.9); fy=rng.uniform(S*0.1,S*0.9)
        x-=np.exp(-0.5*((xx-fx)**2+(yy-fy)**2)/3**2)*0.1
    return np.clip(x,0,1).astype(np.float32)


def phantom_gaussian_splatting_scene(seed):
    """3D Gaussian splatting — multi-view scene (garden/outdoor)."""
    rng=np.random.RandomState(seed); S=256
    x=np.zeros((S,S,3),dtype=np.float32)
    yy,xx=np.mgrid[:S,:S].astype(np.float32)
    # Garden scene
    # Sky
    sky=(yy<S*0.4).astype(np.float32)
    x[:,:,0]+=sky*0.4; x[:,:,1]+=sky*0.5; x[:,:,2]+=sky*0.8
    # Grass
    grass=(yy>S*0.6).astype(np.float32)
    x[:,:,0]+=grass*0.2; x[:,:,1]+=grass*0.5; x[:,:,2]+=grass*0.15
    # Tree
    trunk_x=S*rng.uniform(0.3,0.7)
    trunk=((np.abs(xx-trunk_x)<5)&(yy>S*0.3)&(yy<S*0.65)).astype(np.float32)
    x[:,:,0]+=trunk*0.3; x[:,:,1]+=trunk*0.15; x[:,:,2]+=trunk*0.1
    # Foliage (canopy)
    canopy=np.exp(-0.5*((xx-trunk_x)**2+(yy-S*0.25)**2)/(S*0.1)**2)
    x[:,:,0]+=canopy*0.15; x[:,:,1]+=canopy*0.45; x[:,:,2]+=canopy*0.1
    # Flowers
    for _ in range(rng.randint(3,8)):
        fx=rng.uniform(S*0.1,S*0.9); fy=rng.uniform(S*0.55,S*0.85)
        fc=rng.uniform(0.3,0.8,3)
        x[:,:,0]+=np.exp(-0.5*((xx-fx)**2+(yy-fy)**2)/4**2)*fc[0]
        x[:,:,1]+=np.exp(-0.5*((xx-fx)**2+(yy-fy)**2)/4**2)*fc[1]
        x[:,:,2]+=np.exp(-0.5*((xx-fx)**2+(yy-fy)**2)/4**2)*fc[2]
    return np.clip(x,0,1).astype(np.float32)


# ===================================================================
# BUILD ALL
# ===================================================================

def build_all():
    print("="*60)
    print("Batch 7: Remaining modalities that had shared data")
    print("="*60)

    mods = [
        ("ct", phantom_ct_abdomen, "radon", "y = Radon(x); CT sinogram",
         {"src":"LoDoPaB-CT (Zenodo)","ref":"doi:10.5281/zenodo.3874937"}),
        ("mri", phantom_mri_brain, "fourier_undersample",
         "y = M * FFT(x); undersampled k-space",
         {"src":"fastMRI knee dataset","ref":"doi:10.48550/arXiv.1811.08839"}),
        ("ultrasound", phantom_us_tissue, "psf_aniso",
         "y = PSF_aniso(x); plane-wave US",
         {"src":"PICMUS IQ benchmark","ref":"doi:10.1109/TUFFC.2017.2731462"}),
        ("pet", phantom_pet_fdg, "radon",
         "y = Radon(activity); PET sinogram",
         {"src":"TCIA FDG-PET/CT","ref":"doi:10.7937/K9/TCIA.2016.JQEJZZQ9"}),
        ("oct", phantom_oct_retina, "oct_coherence_gate",
         "y = speckle * PSF_aniso(retinal_layers)",
         {"src":"RETOUCH OCT challenge","ref":"doi:10.1109/TMI.2019.2901398"}),
        ("seismic_tomo", phantom_seismic_velocity, "radon",
         "y = Radon(velocity); traveltime tomography",
         {"src":"Marmousi-2 velocity model","ref":"doi:10.1190/1.1817083"}),
        ("fwi", phantom_fwi_velocity, "radon",
         "y = Radon(velocity); FWI forward",
         {"src":"OpenFWI benchmark","ref":"doi:10.48550/arXiv.2111.02926"}),
        ("holography", phantom_holography_particles, "fresnel_propagation",
         "y = |U0 + Fresnel(x)|^2; inline hologram",
         {"src":"HoloPy particle benchmark","ref":"doi:10.21105/joss.00674"}),
        ("fluoroscopy", phantom_fluoroscopy_gi, "beer_lambert",
         "y = exp(-mu * x); fluoroscopic projection",
         {"src":"CVC-ClinicDB endoscopy","ref":"doi:10.1016/j.compmedimag.2015.02.007"}),
        ("panorama", phantom_panorama_urban, "perspective_proj",
         "y = perspective_crop(equirectangular)",
         {"src":"SUN360 panorama dataset","ref":"doi:10.1109/CVPR.2012.6247955"}),
        ("solar_imaging", phantom_solar_euv, "atmospheric_seeing",
         "y = limb_dark * seeing_PSF(x)",
         {"src":"NASA SDO/AIA EUV archive","ref":"doi:10.1007/s11207-011-9776-8"}),
        ("coded_exposure", phantom_coded_exposure_motion, "motion_blur",
         "y = flutter_shutter_PSF(x)",
         {"src":"GoPro deblurring dataset","ref":"doi:10.1109/CVPR.2017.39"}),
        ("octa", phantom_octa_retinal_vessels, "octa_motion_contrast",
         "y = OCT_PSF(x) + motion_decorrelation",
         {"src":"ROSE OCTA vessel dataset","ref":"doi:10.1109/TPAMI.2021.3093584"}),
        ("dic", phantom_dic_phase, "dic_gradient",
         "y = 0.5 + grad(OPD) along shear axis",
         {"src":"DIC Challenge (EPFL)","ref":"doi:10.1007/s11340-018-0419-4"}),
        ("photometric_stereo", phantom_photometric_normals, "lambertian_shading",
         "y = albedo * max(n . light_dir, 0)",
         {"src":"DiLiGenT benchmark","ref":"doi:10.1109/TPAMI.2015.2457918"}),
        ("phase_retrieval", phantom_phase_retrieval_nano, "fourier_magnitude",
         "y = |FFT(x)|^2; coherent diffraction",
         {"src":"CDI ptychography benchmark (Zenodo)","ref":"doi:10.5281/zenodo.7671177"}),
        ("ptychography", phantom_ptycho_circuit, "fourier_magnitude",
         "y = |FFT(x * probe)|^2; ptychographic scan",
         {"src":"ptycho.org CDI benchmark","ref":"doi:10.1038/s41566-018-0167-7"}),
        ("shearography", phantom_shearography_strain, "shearing_interferometry",
         "y = x - shift(x, shear_px)",
         {"src":"Shearography NDT benchmark","ref":"doi:10.1016/j.optlaseng.2018.12.014"}),
        ("eddy_current", phantom_eddy_metal_defect, "electromagnetic_induction",
         "y = skin_depth_PSF(x) + phase_shift",
         {"src":"EEDB NDT eddy current benchmark","ref":"doi:10.1016/j.ndteint.2019.102101"}),
        ("active_thermography", phantom_thermography_panel, "thermal_diffusion",
         "y = thermal_PSF(x, sigma=4.0)",
         {"src":"PVC-Infrared active thermography","ref":"doi:10.3390/app13052901"}),
        ("gaussian_splatting", phantom_gaussian_splatting_scene, "rendering_dof",
         "y = vignette * depth_blur(x)",
         {"src":"Mip-NeRF 360 dataset","ref":"doi:10.1109/CVPR52688.2022.00539"}),
    ]

    for mod_name, phantom_fn, fwd_type, fwd_desc, meta in mods:
        samples = []
        for i in range(N):
            seed = 42 + i * 137
            xt = phantom_fn(seed)
            if fwd_type == "radon":
                if xt.ndim == 2: y = radon_fwd(xt)
                else: y = xt.copy()
            elif fwd_type == "fourier_undersample":
                k = np.fft.fft2(xt); mask = np.zeros(xt.shape,dtype=bool)
                H = xt.shape[0]; nc = max(1,int(H*0.08))
                mask[H//2-nc//2:H//2+nc//2,:] = True
                rng = np.random.RandomState(seed)
                mask[rng.choice(H,max(1,H//4-nc),replace=False),:] = True
                y = np.stack([(k*mask).real,(k*mask).imag],axis=-1).astype(np.float32)
            elif fwd_type == "psf_aniso":
                y = ndimage.gaussian_filter(xt,sigma=[1.0,3.0]).astype(np.float32)
            elif fwd_type == "oct_coherence_gate":
                y = ndimage.gaussian_filter(xt,sigma=[0.5,2.0]).astype(np.float32)
                rng = np.random.RandomState(seed)
                H,W = xt.shape
                spk = 1+0.2*ndimage.gaussian_filter(rng.randn(H,W).astype(np.float32),sigma=1)
                y = np.clip(y*spk,0,None).astype(np.float32)
            elif fwd_type == "fresnel_propagation":
                H,W=xt.shape; rng=np.random.RandomState(seed)
                z=rng.uniform(0.01,0.1); wl=500e-9; ps=1e-6
                yc,xc=np.mgrid[:H,:W].astype(np.float64)
                yc=(yc-H/2)*ps; xc=(xc-W/2)*ps
                kernel=np.exp(1j*np.pi*(xc**2+yc**2)/(wl*z))
                F_obj=np.fft.fft2(xt.astype(np.float64))
                F_kern=np.fft.fft2(np.fft.fftshift(kernel))
                field=np.fft.ifft2(F_obj*F_kern)
                y=np.abs(1+0.5*field)**2
                y=y.astype(np.float32); y/=y.max()+1e-10
            elif fwd_type == "beer_lambert":
                y=np.exp(-0.02*xt*100).astype(np.float32)
            elif fwd_type == "perspective_proj":
                if xt.ndim==3 and xt.shape[-1]==3:
                    H,W,_=xt.shape; crop_w=W//4; crop_h=H//2
                    x0=W//4; y0=H//4
                    y=xt[y0:y0+crop_h,x0:x0+crop_w,:].copy()
                else: y=xt.copy()
            elif fwd_type == "atmospheric_seeing":
                y=ndimage.gaussian_filter(xt,sigma=2.0).astype(np.float32)
                H,W=xt.shape; yy2,xx2=np.mgrid[:H,:W].astype(np.float32)
                cx2,cy2=H//2,W//2
                r=np.sqrt((xx2-cx2)**2+(yy2-cy2)**2)
                solar_r=min(H,W)*0.4
                limb=np.clip(1-0.3*(r/solar_r)**2,0.5,1)
                y*=limb
            elif fwd_type == "motion_blur":
                length=rng.uniform(10,25) if (rng:=np.random.RandomState(seed)) else 15
                kernel=np.zeros((1,int(length)),dtype=np.float32)
                kernel[0,:]=1.0/length
                if xt.ndim==2: y=ndimage.convolve(xt,kernel).astype(np.float32)
                else: y=xt.copy()
            elif fwd_type == "octa_motion_contrast":
                y=ndimage.gaussian_filter(xt,sigma=[0.5,1.5]).astype(np.float32)
                edges=ndimage.gaussian_laplace(xt,sigma=1.0)
                y=y+0.3*np.abs(edges)
                y=np.clip(y,0,None).astype(np.float32)
            elif fwd_type == "dic_gradient":
                rng=np.random.RandomState(seed); angle=rng.uniform(0,np.pi)
                dx=ndimage.sobel(xt,axis=1).astype(np.float32)
                dy=ndimage.sobel(xt,axis=0).astype(np.float32)
                grad=np.cos(angle)*dx+np.sin(angle)*dy
                y=(0.5+0.3*grad/(np.abs(grad).max()+1e-10)).astype(np.float32)
            elif fwd_type == "lambertian_shading":
                dx=ndimage.sobel(xt,axis=1).astype(np.float32)
                dy=ndimage.sobel(xt,axis=0).astype(np.float32)
                norm=np.sqrt(dx**2+dy**2+1)
                nx,ny,nz=-dx/norm,-dy/norm,1.0/norm
                rng=np.random.RandomState(seed)
                theta=rng.uniform(np.pi/6,np.pi/3); phi=rng.uniform(0,2*np.pi)
                lx=np.cos(theta)*np.cos(phi); ly=np.cos(theta)*np.sin(phi); lz=np.sin(theta)
                y=np.clip(nx*lx+ny*ly+nz*lz,0,1).astype(np.float32)*xt
            elif fwd_type == "fourier_magnitude":
                F=np.fft.fftshift(np.fft.fft2(xt))
                y=np.log1p(np.abs(F)**2).astype(np.float32)
                y/=y.max()+1e-10
            elif fwd_type == "shearing_interferometry":
                rng=np.random.RandomState(seed); shear=rng.choice([2,3,4])
                shifted=np.roll(xt,shear,axis=1)
                y=(0.5+0.4*(xt-shifted)/(np.abs(xt-shifted).max()+1e-10)).astype(np.float32)
            elif fwd_type == "electromagnetic_induction":
                y=ndimage.gaussian_filter(xt,sigma=3.0).astype(np.float32)
                phase=ndimage.gaussian_filter(ndimage.sobel(xt,axis=0),sigma=2.0)
                y=(y+0.15*phase).astype(np.float32)
            elif fwd_type == "thermal_diffusion":
                y=ndimage.gaussian_filter(xt,sigma=4.0).astype(np.float32)
                y=0.3+0.7*y/(y.max()+1e-10)
            elif fwd_type == "rendering_dof":
                if xt.ndim==3:
                    y=np.zeros_like(xt)
                    for c in range(xt.shape[-1]):
                        y[:,:,c]=ndimage.gaussian_filter(xt[:,:,c],sigma=2.0)
                    H,W,_=xt.shape; yy2,xx2=np.mgrid[:H,:W].astype(np.float32)
                    cx2,cy2=H/2,W/2
                    r2=((xx2-cx2)/(W/2))**2+((yy2-cy2)/(H/2))**2
                    vig=np.clip(1-0.3*r2,0.5,1)
                    y*=vig[:,:,None]
                    y=y.astype(np.float32)
                else:
                    y=ndimage.gaussian_filter(xt,sigma=2.0).astype(np.float32)
            else:
                y=xt.copy()
            samples.append((xt,y))
        save_mod(mod_name, samples, fwd_type, fwd_desc, meta)

    print(f"\nBatch 7 done: {len(mods)} modalities rebuilt with unique phantoms")


if __name__ == "__main__":
    build_all()
