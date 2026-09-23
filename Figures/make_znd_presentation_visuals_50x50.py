#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RESULTS_50=ROOT/"results_50x50"
RESULTS=RESULTS_50 if RESULTS_50.exists() else ROOT/"results"
OUT=ROOT/"presentation_figures_50x50"; OUT.mkdir(exist_ok=True)
VAL=RESULTS/"bilinear_multipoint_validation.csv"
SUM=RESULTS/"bilinear_multipoint_summary.csv"

TGRID=np.linspace(300.,1000.,50)
PGRID=np.linspace(.1,5.,50)
MISSING=[(400.,3.9),(428.571429,3.2),(485.714286,2.3)]
STATES=[(351.5,.84),(351.5,2.54),(351.5,4.64),
        (651.5,.84),(651.5,2.54),(651.5,4.64),
        (951.5,.84),(951.5,2.54),(951.5,4.64)]

JAX=np.array([7.355,7.479,7.311,7.666,7.365,7.422,6.629,7.536,7.485])
SCIPY=np.array([47.661,48.472,47.732,48.495,48.808,48.624,51.026,49.219,51.808])

OLD={"Temperature":.113727,"Pressure":1.669653,"Y_H2":.145745,"Y_O2":.139578,
"Y_H2O":.136586,"Y_OH":.158660,"Y_H":.409724,"Y_O":.291064,
"Y_HO2":.395153,"Y_H2O2":.266157}

def save(fig,name):
    fig.savefig(OUT/f"{name}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

def clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def col(df,*terms):
    for c in df.columns:
        s=str(c).lower()
        if all(t.lower() in s for t in terms): return c
    return None

def qlabel(q):
    return {"Temperature":"Temperature","Pressure":"Pressure","Y_H2":r"$H_2$",
    "Y_O2":r"$O_2$","Y_H2O":r"$H_2O$","Y_OH":"OH","Y_H":"H","Y_O":"O",
    "Y_HO2":r"$HO_2$","Y_H2O2":r"$H_2O_2$","Y_N2":r"$N_2$"}.get(q,q)

def manifold():
    pts=np.array([(T,P) for T in TGRID for P in PGRID
                  if not any(np.isclose(T,a) and np.isclose(P,b) for a,b in MISSING)])
    v=np.array(STATES)
    fig,ax=plt.subplots(figsize=(10.5,6.2))
    ax.scatter(pts[:,0],pts[:,1],s=8,alpha=.42,label="Precomputed manifold state")
    ax.scatter(v[:,0],v[:,1],s=110,marker="*",zorder=4,label="Off-grid validation state")
    for k,(T,P) in enumerate(MISSING):
        ax.scatter([T],[P],s=85,marker="x",linewidths=2.2,zorder=5,
                   label="Timed-out state" if k==0 else None)
    ax.set(title="50 × 50 ZND manifold and independent validation states",
           xlabel=r"Initial temperature, $T_1$ [K]",ylabel=r"Initial pressure, $P_1$ [atm]")
    ax.text(.02,.97,"2,497 generated manifold states\n9 off-grid validation states\n3 timed-out states",
            transform=ax.transAxes,va="top",fontsize=11)
    ax.legend(frameon=False); clean(ax); fig.tight_layout(); save(fig,"01_manifold_validation_map_50x50")

def geometry():
    Tq,Pq=351.5,.84
    iT=np.searchsorted(TGRID,Tq); iP=np.searchsorted(PGRID,Pq)
    T0,T1=TGRID[iT-1],TGRID[iT]; P0,P1=PGRID[iP-1],PGRID[iP]
    a=(Tq-T0)/(T1-T0); b=(Pq-P0)/(P1-P0)
    corners=np.array([[T0,P0],[T1,P0],[T0,P1],[T1,P1]])
    weights=np.array([(1-a)*(1-b),a*(1-b),(1-a)*b,a*b])
    nearest=corners[np.argmax(weights)]
    fig,ax=plt.subplots(figsize=(9.5,6.2))
    ax.scatter(corners[:,0],corners[:,1],s=150,label="50 × 50 manifold corners")
    ax.scatter([Tq],[Pq],s=190,marker="*",zorder=5,label="Query: 351.5 K, 0.84 atm")
    ax.plot([T0,T1,T1,T0,T0],[P0,P0,P1,P1,P0],lw=1.5)
    ax.annotate("Nearest grid point",xy=nearest,xytext=(Tq+1,P0+.012),
                arrowprops=dict(arrowstyle="->",lw=1.5))
    for (T,P),w in zip(corners,weights): ax.text(T-2.2,P+.006,f"w={w:.3f}",fontsize=10)
    ax.set(title="Bilinear reconstruction on the refined 50 × 50 grid",
           xlabel=r"Initial temperature, $T_1$ [K]",ylabel=r"Initial pressure, $P_1$ [atm]",
           xlim=(T0-5,T1+5),ylim=(P0-.025,P1+.025))
    ax.legend(frameon=False,loc="lower left"); clean(ax); fig.tight_layout(); save(fig,"02_interpolation_geometry_50x50")

def validation_df():
    if not VAL.exists(): raise FileNotFoundError(f"Missing {VAL}")
    return pd.read_csv(VAL)

def refinement():
    if not SUM.exists(): raise FileNotFoundError(f"Missing {SUM}")
    d=pd.read_csv(SUM); qc=col(d,"quantity"); mc=col(d,"mean","nrmse")
    new=dict(zip(d[qc].astype(str),d[mc].astype(float)))
    qs=list(OLD); old=np.array([OLD[q] for q in qs]); cur=np.array([new[q] for q in qs])
    order=np.argsort(old); qs=[qs[i] for i in order]; old=old[order]; cur=cur[order]
    y=np.arange(len(qs)); h=.36
    fig,ax=plt.subplots(figsize=(10,6.5))
    ax.barh(y+h/2,old,height=h,label="8 × 50 baseline")
    ax.barh(y-h/2,cur,height=h,label="50 × 50 refined")
    ax.set_yticks(y,[qlabel(q) for q in qs]); ax.set_xscale("log")
    ax.set(xlabel="Mean profile NRMSE across 9 states [%]",
           title="Refining temperature sampling substantially reduces interpolation error")
    ax.grid(axis="x",which="both",alpha=.18); ax.legend(frameon=False)
    red=100*(1-new["Pressure"]/OLD["Pressure"])
    ax.text(.98,.05,f"Pressure mean NRMSE reduction: {red:.1f}%",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=10)
    clean(ax); fig.tight_layout(); save(fig,"03_8x50_vs_50x50_mean_nrmse")

def heatmap(df,q,name):
    qc=col(df,"quantity"); ec=col(df,"nrmse"); sub=df[df[qc].astype(str)==q].copy()
    tc=col(sub,"t_k") or col(sub,"temperature"); pc=col(sub,"p_atm") or col(sub,"pressure")
    cc=col(sub,"case")
    if tc is None or pc is None:
        if cc is not None:
            mp={i+1:s for i,s in enumerate(STATES)}
            sub["_T"]=[mp[int(x)][0] for x in sub[cc]]; sub["_P"]=[mp[int(x)][1] for x in sub[cc]]
        elif len(sub)==9:
            sub["_T"]=[s[0] for s in STATES]; sub["_P"]=[s[1] for s in STATES]
        else: raise ValueError("Cannot recover validation T/P coordinates.")
        tc,pc="_T","_P"
    temps=[351.5,651.5,951.5]; press=[.84,2.54,4.64]; Z=np.full((3,3),np.nan)
    for i,T in enumerate(temps):
        for j,P in enumerate(press):
            r=sub[np.isclose(sub[tc].astype(float),T)&np.isclose(sub[pc].astype(float),P)]
            if len(r): Z[i,j]=float(r.iloc[0][ec])
    fig,ax=plt.subplots(figsize=(7.5,5.7)); im=ax.imshow(Z,origin="lower",aspect="auto")
    ax.set_xticks(range(3),[f"{p:.2f}" for p in press]); ax.set_yticks(range(3),[f"{t:.1f}" for t in temps])
    ax.set(xlabel=r"Initial pressure, $P_1$ [atm]",ylabel=r"Initial temperature, $T_1$ [K]",
           title=f"{qlabel(q)} profile error — 50 × 50 manifold")
    for i in range(3):
        for j in range(3): ax.text(j,i,f"{Z[i,j]:.3f}%",ha="center",va="center",weight="bold")
    cb=fig.colorbar(im,ax=ax); cb.set_label("NRMSE [%]"); fig.tight_layout(); save(fig,name)

def aggregate():
    d=pd.read_csv(SUM); qc=col(d,"quantity"); mc=col(d,"mean","nrmse"); xc=col(d,"max","nrmse")
    d=d[d[qc].astype(str)!="Y_N2"].sort_values(mc); y=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(9.5,6.4))
    ax.barh(y,d[mc].astype(float),alpha=.75,label="Mean NRMSE")
    ax.scatter(d[xc].astype(float),y,s=65,marker="D",zorder=4,label="Maximum NRMSE")
    ax.set_yticks(y,[qlabel(q) for q in d[qc]])
    ax.set(xlabel="NRMSE [%]",title="50 × 50 bilinear reconstruction accuracy across 9 off-grid states")
    ax.legend(frameon=False); clean(ax); fig.tight_layout(); save(fig,"07_aggregate_nrmse_50x50")

def speed():
    means=[SCIPY.mean(),JAX.mean()]; std=[SCIPY.std(ddof=1),JAX.std(ddof=1)]
    fig,ax=plt.subplots(figsize=(7.5,5.8)); bars=ax.bar(["SciPy","JAX"],means,yerr=std,capsize=6)
    ax.set(ylabel="Warmed lookup time [μs/query]",title="50 × 50 warmed interpolation-only lookup on CPU")
    for b,v in zip(bars,means): ax.text(b.get_x()+b.get_width()/2,v+1,f"{v:.2f} μs",ha="center",weight="bold")
    ax.text(.98,.92,f"Measured mean speedup: {SCIPY.mean()/JAX.mean():.2f}×",transform=ax.transAxes,ha="right")
    ax.text(.98,.84,"Disk I/O and physical-x resampling excluded",transform=ax.transAxes,ha="right",fontsize=9)
    clean(ax); fig.tight_layout(); save(fig,"08_jax_scipy_speed_50x50")

def main():
    print(f"Using validation results from: {RESULTS}")
    manifold(); geometry(); refinement(); d=validation_df()
    heatmap(d,"Temperature","04_validation_heatmap_temperature_50x50")
    heatmap(d,"Pressure","05_validation_heatmap_pressure_50x50")
    heatmap(d,"Y_HO2","06_validation_heatmap_HO2_50x50")
    aggregate(); speed()
    print(f"Done. PNG + SVG figures saved to {OUT}")

if __name__=="__main__": main()
