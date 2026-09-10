"""CS771 Mini-Project 1 - main entry point.
Author: Vedant Tiwari (221184).

Running this file freshly trains all four models (fixed seeds; a few seconds
each) and writes the four required prediction files into the current folder:
    pred_emoticon.txt   pred_deepfeat.txt   pred_textseq.txt   pred_combined.txt
Each file has one predicted label (0/1) per test input, one per line.

NOTE: the course expects this file to be named <your group no>.py at
submission time; copy/rename this file accordingly.
"""
import numpy as np
from sklearn.metrics import accuracy_score
import common as C
import emoticon_final as EM
import deepfeat_final as DF
import textseq_final as TS
import combine_final as CB

def _acc_emo():
    enc, clf, n = EM.build(); Xva, yva = C.load_emoticon("valid")
    return accuracy_score(yva, clf.predict(enc.transform(Xva))), n
def _acc_deep():
    clf, n = DF.build(); Xva, yva = C.load_deepfeat("valid")
    return accuracy_score(yva, clf.predict(C.deep_flatten(Xva))), n
def _acc_seq():
    fe, clf, n = TS.build(); Xva, yva = C.load_textseq("valid")
    return accuracy_score(yva, clf.predict(fe.transform(Xva))), n

def main():
    print("Training models and writing predictions...\n")

    np.savetxt("pred_emoticon.txt", EM.predict_test(), fmt="%d")
    a, n = _acc_emo();  print(f"  emoticon : valid={a*100:5.2f}%  params={n}")

    np.savetxt("pred_deepfeat.txt", DF.predict_test(), fmt="%d")
    a, n = _acc_deep(); print(f"  deepfeat : valid={a*100:5.2f}%  params={n}")

    np.savetxt("pred_textseq.txt", TS.predict_test(), fmt="%d")
    a, n = _acc_seq();  print(f"  textseq  : valid={a*100:5.2f}%  params={n}")

    meta, Z_va, yva, Z_te = CB.build()
    np.savetxt("pred_combined.txt", meta.predict(Z_te).astype(int), fmt="%d")
    print(f"  combined : valid={accuracy_score(yva, meta.predict(Z_va))*100:5.2f}%  meta-params=4")

    print("\nWrote: pred_emoticon.txt, pred_deepfeat.txt, pred_textseq.txt, pred_combined.txt")

if __name__ == "__main__":
    main()
