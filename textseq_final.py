"""Text-sequence dataset model  (CS771 Mini-Project 1, Task 1).
Author: Vedant Tiwari (221184).

Representation: a shuffled concatenation of variable-length per-emoji digit
codes (EDA: '15436' occurs once in every row, '1596' ~twice, at varying
offsets). Clean tokenisation back to 13 ordered codes is ambiguous, and the
rules forbid using the other datasets to decode it. Best linear features that
stay <10k params: length-50 positional digit one-hot (500) + char n-gram
(2-5) presence counts capped at 9000 -> L2 Logistic Regression.
Trainable params = 9500 + 1 = 9501  (<= 10000).
"""
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import common as C

SEED = 42

class SeqFeaturizer:
    def __init__(self):
        self.pos = C.SeqPositionalEncoder(50)
        self.vec = CountVectorizer(analyzer='char', ngram_range=(2, 5), max_features=9000)
    def fit(self, X):
        self.vec.fit(X); return self
    def transform(self, X):
        P = csr_matrix(self.pos.transform(X))
        return hstack([P, self.vec.transform(X)]).tocsr()
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def build():
    Xtr, ytr = C.load_textseq("train")
    fe = SeqFeaturizer().fit(Xtr)
    Xt = fe.transform(Xtr)
    clf = LogisticRegression(C=1, max_iter=5000, random_state=SEED)
    clf.fit(Xt, ytr)
    return fe, clf, Xt.shape[1] + 1

def predict_test():
    fe, clf, _ = build()
    Xte, _ = C.load_textseq("test")
    return clf.predict(fe.transform(Xte)).astype(int)

if __name__ == "__main__":
    fe, clf, n = build()
    Xva, yva = C.load_textseq("valid")
    from sklearn.metrics import accuracy_score
    print(f"[textseq] params={n}  valid acc={accuracy_score(yva, clf.predict(fe.transform(Xva)))*100:.2f}%")
    np.savetxt("pred_textseq.txt", predict_test(), fmt="%d")
    print("wrote pred_textseq.txt")
