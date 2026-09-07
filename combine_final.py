"""Combined model  (CS771 Mini-Project 1, Task 2).
Author: Vedant Tiwari (221184).

Since the 3 datasets encode the SAME inputs (identical label ordering), we
combine at the decision level by *stacking*: each base model emits P(y=1),
and a tiny logistic meta-learner (3 inputs + bias = 4 trainable params) is
trained on 5-fold out-of-fold base probabilities to avoid leakage.

Finding: stacking learns to put almost all weight on the deep-features view
and matches it (~98.6%); the weaker emoticon/sequence views do not add
complementary signal, so combining does not beat the best single model.
Meta-learner trainable params = 4  (base models are the per-dataset models,
each already <=10000 params).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import common as C
import emoticon_final as EM
import deepfeat_final as DF
import textseq_final as TS

SEED = 42

def _base_specs():
    # returns list of (name, Xtrain, Xvalid, Xtest, make_estimator)
    Xe_tr, ytr = C.load_emoticon("train"); Xe_va, yva = C.load_emoticon("valid"); Xe_te, _ = C.load_emoticon("test")
    enc = C.EmoticonEncoder().fit(Xe_tr)
    emo = ("emo", enc.transform(Xe_tr), enc.transform(Xe_va), enc.transform(Xe_te),
           lambda: LogisticRegression(C=5, max_iter=4000, random_state=SEED))

    Xd_tr, _ = C.load_deepfeat("train"); Xd_va, _ = C.load_deepfeat("valid"); Xd_te, _ = C.load_deepfeat("test")
    deep = ("deep", C.deep_flatten(Xd_tr), C.deep_flatten(Xd_va), C.deep_flatten(Xd_te),
            lambda: LogisticRegression(C=0.3, max_iter=4000, random_state=SEED))

    Xs_tr, _ = C.load_textseq("train"); Xs_va, _ = C.load_textseq("valid"); Xs_te, _ = C.load_textseq("test")
    fe = TS.SeqFeaturizer().fit(Xs_tr)
    seq = ("seq", fe.transform(Xs_tr), fe.transform(Xs_va), fe.transform(Xs_te),
           lambda: LogisticRegression(C=1, max_iter=5000, random_state=SEED))
    return ytr, yva, [emo, deep, seq]

def build():
    ytr, yva, specs = _base_specs()
    oof, va_p, te_p = [], [], []
    for name, Xt, Xv, Xte, mk in specs:
        oof.append(cross_val_predict(mk(), Xt, ytr, cv=5, method="predict_proba")[:, 1])
        m = mk().fit(Xt, ytr)
        va_p.append(m.predict_proba(Xv)[:, 1])
        te_p.append(m.predict_proba(Xte)[:, 1])
    Z_tr = np.column_stack(oof); Z_va = np.column_stack(va_p); Z_te = np.column_stack(te_p)
    meta = LogisticRegression(max_iter=2000, random_state=SEED).fit(Z_tr, ytr)
    return meta, Z_va, yva, Z_te

def predict_test():
    meta, _, _, Z_te = build()
    return meta.predict(Z_te).astype(int)

if __name__ == "__main__":
    meta, Z_va, yva, Z_te = build()
    from sklearn.metrics import accuracy_score
    print(f"[combined] meta params=4  valid acc={accuracy_score(yva, meta.predict(Z_va))*100:.2f}%")
    print("meta weights (emo,deep,seq):", np.round(meta.coef_.ravel(), 3))
    np.savetxt("pred_combined.txt", meta.predict(Z_te).astype(int), fmt="%d")
    print("wrote pred_combined.txt")
