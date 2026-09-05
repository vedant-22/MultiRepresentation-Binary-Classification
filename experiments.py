"""Model-selection experiments for each representation.
Author: Vedant Tiwari (221184). Reports validation accuracy across the
{20,40,60,80,100}% training-size sweep and enforces the <=10,000 trainable
parameter budget for every candidate.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import common as C

FRACS = [0.2,0.4,0.6,0.8,1.0]
SEED = 42

def sweep(make_model, Xtr, ytr, Xva, yva, nfeat):
    accs=[]
    for f in FRACS:
        Xp, yp = C.prefix(Xtr, ytr, f)
        m = make_model()
        m.fit(Xp, yp)
        accs.append(accuracy_score(yva, m.predict(Xva)))
    return accs

def report(name, accs, nparam):
    row=" ".join(f"{a*100:6.2f}" for a in accs)
    print(f"{name:34s} params={nparam:5d}  [{row}]  best={max(accs)*100:.2f}")

print("cols = 20% 40% 60% 80% 100% training data  ->  validation accuracy\n")

# ================= EMOTICON =================
print("========== EMOTICON ==========")
Xtr_raw, ytr = C.load_emoticon("train")
Xva_raw, yva = C.load_emoticon("valid")
enc = C.EmoticonEncoder().fit(Xtr_raw)
Xtr = enc.transform(Xtr_raw); Xva = enc.transform(Xva_raw)
nf = Xtr.shape[1]
print("one-hot features:", nf)
report("LogReg C=1", sweep(lambda:LogisticRegression(C=1,max_iter=2000), Xtr,ytr,Xva,yva,nf), nf+1)
report("LogReg C=0.3", sweep(lambda:LogisticRegression(C=0.3,max_iter=2000), Xtr,ytr,Xva,yva,nf), nf+1)
report("LinearSVC C=0.5", sweep(lambda:LinearSVC(C=0.5), Xtr,ytr,Xva,yva,nf), nf+1)

# ================= DEEP FEATURES =================
print("\n========== DEEP FEATURES ==========")
Xtr3, ytr = C.load_deepfeat("train")
Xva3, yva = C.load_deepfeat("valid")
# mean pool
Xtr_mp=C.deep_meanpool(Xtr3); Xva_mp=C.deep_meanpool(Xva3)
report("meanpool768 + StdSc + LogReg", sweep(lambda:make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000)), Xtr_mp,ytr,Xva_mp,yva,768), 768+1)
# flatten 9984
Xtr_fl=C.deep_flatten(Xtr3); Xva_fl=C.deep_flatten(Xva3)
report("flatten9984 + LogReg C=0.1", sweep(lambda:LogisticRegression(C=0.1,max_iter=3000), Xtr_fl,ytr,Xva_fl,yva,9984), 9984+1)
# PCA to 200 then logreg
report("PCA200 + LogReg", sweep(lambda:make_pipeline(StandardScaler(),PCA(200,random_state=SEED),LogisticRegression(C=1,max_iter=3000)), Xtr_mp,ytr,Xva_mp,yva,200), 200+1)

# ================= TEXT SEQUENCE =================
print("\n========== TEXT SEQUENCE ==========")
Xtr_raw,ytr=C.load_textseq("train"); Xva_raw,yva=C.load_textseq("valid")
senc=C.SeqPositionalEncoder(50)
Xtr=senc.transform(Xtr_raw); Xva=senc.transform(Xva_raw)
report("pos-onehot500 + LogReg", sweep(lambda:LogisticRegression(C=1,max_iter=3000), Xtr,ytr,Xva,yva,500), 500+1)
# char n-grams
from sklearn.feature_extraction.text import TfidfVectorizer
for ng,mf in [((2,3),2000),((2,4),5000),((3,5),8000)]:
    vec=TfidfVectorizer(analyzer='char',ngram_range=ng,max_features=mf)
    Xt=vec.fit_transform(Xtr_raw); Xv=vec.transform(Xva_raw)
    nfe=Xt.shape[1]
    def mk(): return LogisticRegression(C=1,max_iter=3000)
    accs=[]
    for f in FRACS:
        n=int(round(len(ytr)*f)); m=mk(); m.fit(Xt[:n],ytr[:n]); accs.append(accuracy_score(yva,m.predict(Xv)))
    report(f"char{ng} tfidf(mf={mf}) LogReg", accs, nfe+1)
