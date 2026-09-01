"""Deep-features dataset model  (CS771 Mini-Project 1, Task 1).
Author: Vedant Tiwari (221184).

Representation: 13 x 768 embedding matrix per input. Mean-pooling collapses
to chance (~50%) because it averages out the positional signal, so we FLATTEN
to 9984 features (preserving position) and fit L2 Logistic Regression.
Trainable params = 9984 + 1 = 9985  (<= 10000).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import common as C

SEED = 42

def build():
    X, y = C.load_deepfeat("train")
    clf = LogisticRegression(C=0.3, max_iter=4000, random_state=SEED)
    clf.fit(C.deep_flatten(X), y)
    return clf, C.deep_flatten(X).shape[1] + 1

def predict_test():
    clf, _ = build()
    Xte, _ = C.load_deepfeat("test")
    return clf.predict(C.deep_flatten(Xte)).astype(int)

if __name__ == "__main__":
    clf, n = build()
    Xva, yva = C.load_deepfeat("valid")
    from sklearn.metrics import accuracy_score
    print(f"[deepfeat] params={n}  valid acc={accuracy_score(yva, clf.predict(C.deep_flatten(Xva)))*100:.2f}%")
    np.savetxt("pred_deepfeat.txt", predict_test(), fmt="%d")
    print("wrote pred_deepfeat.txt")
