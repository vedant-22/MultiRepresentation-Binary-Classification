"""Emoticon dataset model  (CS771 Mini-Project 1, Task 1).
Author: Vedant Tiwari (221184).

Representation: each input is 13 emojis. EDA showed 7 emojis are constant
"padding" present in every row, and the label depends on the *arrangement*
of tokens (bag-of-content-emojis is at chance ~52%, positional encoding ~91%).
Model: positional one-hot (13 slots x |vocab|) -> L2 Logistic Regression.
Trainable params = 2782 + 1 = 2783  (<= 10000).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import common as C

SEED = 42

def build():
    Xtr, ytr = C.load_emoticon("train")
    enc = C.EmoticonEncoder().fit(Xtr)
    clf = LogisticRegression(C=5, max_iter=4000, random_state=SEED)
    clf.fit(enc.transform(Xtr), ytr)
    n_params = enc.transform(Xtr[:1]).shape[1] + 1
    return enc, clf, n_params

def predict_test():
    enc, clf, _ = build()
    Xte, _ = C.load_emoticon("test")
    return clf.predict(enc.transform(Xte)).astype(int)

if __name__ == "__main__":
    enc, clf, n = build()
    Xva, yva = C.load_emoticon("valid")
    from sklearn.metrics import accuracy_score
    print(f"[emoticon] params={n}  valid acc={accuracy_score(yva, clf.predict(enc.transform(Xva)))*100:.2f}%")
    np.savetxt("pred_emoticon.txt", predict_test(), fmt="%d")
    print("wrote pred_emoticon.txt")
