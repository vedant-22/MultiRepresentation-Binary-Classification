"""Generate accuracy-vs-training-size plots for the report (Task 1 & Task 2).
Author: Vedant Tiwari (221184)."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import common as C
import textseq_final as TS

FRACS=[0.2,0.4,0.6,0.8,1.0]; SEED=42

def sweep(Xt, yt, Xv, yv, mk):
    a=[]
    for f in FRACS:
        n=int(round(len(yt)*f)); m=mk(); m.fit(Xt[:n], yt[:n]); a.append(accuracy_score(yv, m.predict(Xv))*100)
    return a

# emoticon
Xe,ye=C.load_emoticon("train"); Xev,yev=C.load_emoticon("valid")
enc=C.EmoticonEncoder().fit(Xe)
emo=sweep(enc.transform(Xe),ye,enc.transform(Xev),yev,lambda:LogisticRegression(C=5,max_iter=4000,random_state=SEED))
# deep
Xd,yd=C.load_deepfeat("train"); Xdv,ydv=C.load_deepfeat("valid")
deep=sweep(C.deep_flatten(Xd),yd,C.deep_flatten(Xdv),ydv,lambda:LogisticRegression(C=0.3,max_iter=4000,random_state=SEED))
# seq
Xs,ys=C.load_textseq("train"); Xsv,ysv=C.load_textseq("valid")
fe=TS.SeqFeaturizer().fit(Xs)
seq=sweep(fe.transform(Xs),ys,fe.transform(Xsv),ysv,lambda:LogisticRegression(C=1,max_iter=5000,random_state=SEED))
# combined (stacking) sweep
def comb_sweep():
    specs=[(enc.transform(Xe),enc.transform(Xev),lambda:LogisticRegression(C=5,max_iter=4000,random_state=SEED)),
           (C.deep_flatten(Xd),C.deep_flatten(Xdv),lambda:LogisticRegression(C=0.3,max_iter=4000,random_state=SEED)),
           (fe.transform(Xs),fe.transform(Xsv),lambda:LogisticRegression(C=1,max_iter=5000,random_state=SEED))]
    out=[]
    for f in FRACS:
        n=int(round(len(ye)*f)); yy=ye[:n]
        oof=[]; vap=[]
        for Xt,Xv,mk in specs:
            oof.append(cross_val_predict(mk(),Xt[:n],yy,cv=5,method="predict_proba")[:,1])
            vap.append(mk().fit(Xt[:n],yy).predict_proba(Xv)[:,1])
        meta=LogisticRegression(max_iter=2000,random_state=SEED).fit(np.column_stack(oof),yy)
        out.append(accuracy_score(yev,meta.predict(np.column_stack(vap)))*100)
    return out
comb=comb_sweep()

xs=[20,40,60,80,100]
np.savez("plots/sweep_data.npz",fracs=xs,emo=emo,deep=deep,seq=seq,comb=comb)

# Individual Task-1 figure
plt.figure(figsize=(7,5))
for name,ys_,mk in [("Emoticon",emo,'o'),("Deep Features",deep,'s'),("Text Sequence",seq,'^')]:
    plt.plot(xs,ys_,marker=mk,label=name)
plt.xlabel("% of training data used"); plt.ylabel("Validation accuracy (%)")
plt.title("Task 1: validation accuracy vs training-set size")
plt.grid(alpha=.3); plt.legend(); plt.ylim(50,100); plt.tight_layout()
plt.savefig("plots/task1_sweep.png",dpi=150)

# Combined figure
plt.figure(figsize=(7,5))
plt.plot(xs,deep,marker='s',label="Deep (best single)")
plt.plot(xs,comb,marker='D',label="Combined (stacking)")
plt.xlabel("% of training data used"); plt.ylabel("Validation accuracy (%)")
plt.title("Task 2: combined vs best single model")
plt.grid(alpha=.3); plt.legend(); plt.ylim(90,100); plt.tight_layout()
plt.savefig("plots/task2_sweep.png",dpi=150)

print("emo ",[round(x,2) for x in emo])
print("deep",[round(x,2) for x in deep])
print("seq ",[round(x,2) for x in seq])
print("comb",[round(x,2) for x in comb])
print("saved plots/task1_sweep.png, plots/task2_sweep.png")
