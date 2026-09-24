#!/usr/bin/env python3
"""Analyze second-gradient/nonlinearity metrics in the cached 50x50 ZND manifold.
Reads znd_tp_profiles/*.npz only; does NOT run SDToolbox.
"""
from pathlib import Path
import csv, math, warnings
import numpy as np
import matplotlib.pyplot as plt

PROFILE_DIR=Path("znd_tp_profiles")
OUT=Path("manifold_curvature")
VAL=np.array([[351.5,.84],[351.5,2.54],[351.5,4.64],
              [651.5,.84],[651.5,2.54],[651.5,4.64],
              [951.5,.84],[951.5,2.54],[951.5,4.64]])
WORST=np.array([351.5,4.64])
TIMEOUT=np.array([[400.,3.9],[428.571429,3.2],[485.714286,2.3]])
GRID_T=np.linspace(300.0,1000.0,50)
GRID_P=np.linspace(0.1,5.0,50)
GRID_T_TOL=1e-5
GRID_P_TOL=1e-7
FEATURES=["cj_speed_m_s","x10_T_m","x50_T_m","x90_T_m","ho2_peak","ho2_peak_x_m"]
LABEL={"cj_speed_m_s":"CJ speed","x10_T_m":"x10 temperature-rise distance",
       "x50_T_m":"x50 temperature-rise distance","x90_T_m":"x90 temperature-rise distance",
       "ho2_peak":"Peak HO2 mass fraction","ho2_peak_x_m":"x-location of HO2 peak"}

def scalar(d,k): return float(np.asarray(d[k]).squeeze())
def names(a):
    return [v.decode() if isinstance(v,bytes) else str(v) for v in np.asarray(a).ravel()]

def crossing(x,y,target):
    a=y[:-1]-target; b=y[1:]-target
    ids=np.where((a==0)|(a*b<=0))[0]
    if not len(ids): return np.nan
    i=int(ids[0]); dy=y[i+1]-y[i]
    return float(x[i] if dy==0 else x[i]+(target-y[i])*(x[i+1]-x[i])/dy)

def extract(path):
    with np.load(path,allow_pickle=True) as d:
        need=["T1_K","P1_atm","cj_speed_m_s","x_m","T_K","Y","species_names"]
        if any(k not in d.files for k in need): raise KeyError("missing expected keys")
        T1=scalar(d,"T1_K"); P1=scalar(d,"P1_atm"); cj=scalar(d,"cj_speed_m_s")
        x=np.asarray(d["x_m"],float).squeeze(); T=np.asarray(d["T_K"],float).squeeze()
        Y=np.asarray(d["Y"],float); sp=names(d["species_names"])
    if Y.shape==(len(sp),len(x)): Y=Y.T
    if Y.shape!=(len(x),len(sp)): raise ValueError("unexpected Y shape")
    T0,Tend=float(T[0]),float(T[-1]); dT=Tend-T0
    r={"T1_K":T1,"P1_atm":P1,"cj_speed_m_s":cj,"source_file":path.name}
    for q in (.1,.5,.9):
        r[f"x{int(100*q):02d}_T_m"]=crossing(x,T,T0+q*dT)
    if "HO2" in sp:
        y=Y[:,sp.index("HO2")]; k=int(np.nanargmax(y))
        r["ho2_peak"]=float(y[k]); r["ho2_peak_x_m"]=float(x[k])
    else:
        r["ho2_peak"]=r["ho2_peak_x_m"]=np.nan
    return r

def curvature(F,dT,dP):
    shape=F.shape
    keys=["d2_dT2","d2_dP2","d2_dTdP","C_T","C_P","C_TP","C_combined"]
    o={k:np.full(shape,np.nan) for k in keys}
    finite=np.abs(F[np.isfinite(F)])
    base=float(np.nanmedian(finite)) if finite.size else 1.
    if not np.isfinite(base) or base==0: base=1.
    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            tv=np.array([F[i-1,j],F[i,j],F[i+1,j]])
            pv=np.array([F[i,j-1],F[i,j],F[i,j+1]])
            cv=np.array([F[i+1,j+1],F[i+1,j-1],F[i-1,j+1],F[i-1,j-1]])
            if np.all(np.isfinite(tv)):
                n=F[i+1,j]-2*F[i,j]+F[i-1,j]
                o["d2_dT2"][i,j]=n/dT**2
                o["C_T"][i,j]=abs(n)/max(np.max(np.abs(tv)),base*1e-12)
            if np.all(np.isfinite(pv)):
                n=F[i,j+1]-2*F[i,j]+F[i,j-1]
                o["d2_dP2"][i,j]=n/dP**2
                o["C_P"][i,j]=abs(n)/max(np.max(np.abs(pv)),base*1e-12)
            if np.all(np.isfinite(cv)):
                n=F[i+1,j+1]-F[i+1,j-1]-F[i-1,j+1]+F[i-1,j-1]
                o["d2_dTdP"][i,j]=n/(4*dT*dP)
                o["C_TP"][i,j]=abs(n)/(4*max(np.max(np.abs(cv)),base*1e-12))
            vals=[o["C_T"][i,j],o["C_P"][i,j],o["C_TP"][i,j]]
            if np.all(np.isfinite(vals)):
                o["C_combined"][i,j]=math.sqrt(vals[0]**2+vals[1]**2+2*vals[2]**2)
    return o

def heatmap(T,P,A,feature,kind,cblabel):
    fig,ax=plt.subplots(figsize=(9,6))
    m=ax.pcolormesh(T,P,A.T,shading="auto")
    fig.colorbar(m,ax=ax,label=cblabel)
    ax.scatter(VAL[:,0],VAL[:,1],marker="*",s=90,facecolors="none",edgecolors="black",
               linewidths=1.2,label="Validation states")
    ax.scatter([WORST[0]],[WORST[1]],marker="*",s=190,facecolors="none",
               edgecolors="black",linewidths=2.4,label="Worst validation state")
    ax.scatter(TIMEOUT[:,0],TIMEOUT[:,1],marker="x",s=80,linewidths=2,label="CJspeed timeout")
    ax.set(xlabel="Initial temperature T1 [K]",ylabel="Initial pressure P1 [atm]",
           title=f"{LABEL[feature]}: {cblabel}")
    ax.legend()
    fig.tight_layout()
    p=OUT/f"{kind}_{feature}.png"; fig.savefig(p,dpi=300,bbox_inches="tight"); plt.close(fig)
    return p

def main():
    OUT.mkdir(exist_ok=True)
    if not PROFILE_DIR.exists(): raise FileNotFoundError(f"Missing {PROFILE_DIR.resolve()}")
    rows=[]; skipped=[]
    for p in sorted(PROFILE_DIR.glob("*.npz")):
        try: rows.append(extract(p))
        except Exception as e: skipped.append((p.name,str(e)))
    if not rows: raise RuntimeError("No compatible profiles found.")
    # IMPORTANT: znd_tp_profiles/ also contains independent off-grid truth
    # files used for validation. Do not infer the manifold coordinates from
    # every NPZ in the directory. Use the known 50x50 design grid and keep
    # only states that land on those nodes.
    T=GRID_T.copy()
    P=GRID_P.copy()
    dT=float(T[1]-T[0]); dP=float(P[1]-P[0])

    G={f:np.full((len(T),len(P)),np.nan) for f in FEATURES}
    occ=np.zeros((len(T),len(P)),bool)
    manifold_rows=[]
    offgrid_rows=[]

    for r in rows:
        i=int(np.argmin(np.abs(T-r["T1_K"])))
        j=int(np.argmin(np.abs(P-r["P1_atm"])))
        on_T=abs(T[i]-r["T1_K"]) <= GRID_T_TOL
        on_P=abs(P[j]-r["P1_atm"]) <= GRID_P_TOL
        if not (on_T and on_P):
            offgrid_rows.append(r)
            continue

        # If duplicate cached files exist for one manifold state, keep the
        # first and warn rather than silently overwriting it.
        if occ[i,j]:
            warnings.warn(
                f"Duplicate manifold state near T={T[i]:.9f} K, "
                f"P={P[j]:.9f} atm; ignoring {r['source_file']}"
            )
            continue

        occ[i,j]=True
        manifold_rows.append(r)
        for f in FEATURES:
            G[f][i,j]=r[f]

    rows=manifold_rows
    print(f"Parsed compatible NPZ files: {len(rows)+len(offgrid_rows)}")
    print(f"50x50 manifold states retained: {len(rows)}")
    print(f"Off-grid validation/truth files excluded: {len(offgrid_rows)}")
    print(f"Grid: {len(T)}x{len(P)}; missing manifold states: {occ.size-occ.sum()}")
    print(f"dT={dT:.9f} K, dP={dP:.9f} atm")
    with (OUT/"manifold_scalar_features.csv").open("w",newline="") as fh:
        fields=["T1_K","P1_atm"]+FEATURES+["source_file"]; w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    allc={}
    for f in FEATURES:
        c=curvature(G[f],dT,dP); allc[f]=c
        with (OUT/f"curvature_{f}.csv").open("w",newline="") as fh:
            fields=["T1_K","P1_atm","feature_value"]+list(c); w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
            for i,t in enumerate(T):
                for j,p in enumerate(P):
                    w.writerow({"T1_K":t,"P1_atm":p,"feature_value":G[f][i,j],
                                **{k:c[k][i,j] for k in c}})
    for f in ["ho2_peak","ho2_peak_x_m","x90_T_m","cj_speed_m_s"]:
        heatmap(T,P,allc[f]["C_combined"],f,"combined_nonlinearity",
                "dimensionless combined second-difference indicator")
        heatmap(T,P,np.abs(allc[f]["d2_dP2"]),f,"abs_d2_dP2","|d²f/dP1²|")
        heatmap(T,P,np.abs(allc[f]["d2_dT2"]),f,"abs_d2_dT2","|d²f/dT1²|")
    print("\nNearest grid node to worst validation state (351.5 K, 4.64 atm):")
    i=int(np.argmin(abs(T-WORST[0]))); j=int(np.argmin(abs(P-WORST[1])))
    for f in ["ho2_peak","ho2_peak_x_m","x90_T_m","cj_speed_m_s"]:
        C=allc[f]["C_combined"]; val=C[i,j]; good=C[np.isfinite(C)]
        pct=100*np.sum(good<=val)/len(good) if np.isfinite(val) else np.nan
        print(f"  {f:16s}: node=({T[i]:.3f} K,{P[j]:.3f} atm), C={val:.6e}, percentile={pct:.1f}")
    C=allc["ho2_peak"]["C_combined"]
    inds=np.argwhere(np.isfinite(C)); inds=sorted(inds,key=lambda q:C[tuple(q)],reverse=True)
    print("\nTop 10 nodes by HO2-peak combined nonlinearity:")
    for n,(i,j) in enumerate(inds[:10],1):
        print(f"  {n:2d}. T1={T[i]:.3f} K, P1={P[j]:.3f} atm, C={C[i,j]:.6e}")
    print(f"\nSaved results to {OUT.resolve()}")
    print("Timeout/missing states were NOT filled; affected finite-difference stencils remain NaN.")
    print("C_combined is a visualization/ranking diagnostic, not a universal physical metric.")

if __name__=="__main__":
    main()
