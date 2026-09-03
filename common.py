"""Shared data loading and feature-transformation utilities.
CS771 Mini-Project 1  |  Author: Vedant Tiwari (221184)

All feature transforms are deterministic and stateless-per-fit so that the
{20,40,60,80,100}% training-size sweep and final refit use identical logic.
"""
import numpy as np
import pandas as pd

DATA = "datasets/"

# ---------- raw loaders ----------
def load_emoticon(split):
    df = pd.read_csv(f"{DATA}{split}/{split}_emoticon.csv")
    X = df['input_emoticon'].astype(str).tolist()
    y = df['label'].to_numpy() if 'label' in df.columns else None
    return X, y

def load_textseq(split):
    df = pd.read_csv(f"{DATA}{split}/{split}_text_seq.csv")
    X = df['input_str'].astype(str).tolist()
    y = df['label'].to_numpy() if 'label' in df.columns else None
    return X, y

def load_deepfeat(split):
    d = np.load(f"{DATA}{split}/{split}_feature.npz", allow_pickle=True)
    X = d['features'].astype(np.float32)          # (N,13,768)
    y = d['label'].astype(int) if 'label' in d.files else None
    return X, y

# ---------- emoticon feature transform ----------
class EmoticonEncoder:
    """Positional one-hot over the 13 emoji slots.
    Vocabulary is learned from the training split only.
    """
    def __init__(self):
        self.vocab = None            # emoji -> index
        self.n_pos = 13

    def fit(self, X):
        chars = set()
        for s in X:
            chars.update(list(s))
        self.vocab = {c: i for i, c in enumerate(sorted(chars))}
        return self

    def transform(self, X):
        V = len(self.vocab)
        out = np.zeros((len(X), self.n_pos * V), dtype=np.float32)
        for r, s in enumerate(X):
            for p, ch in enumerate(list(s)[:self.n_pos]):
                j = self.vocab.get(ch)
                if j is not None:
                    out[r, p * V + j] = 1.0
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ---------- text-sequence feature transform ----------
class SeqPositionalEncoder:
    """Left-pad/truncate each digit string to a fixed length, then one-hot
    each position over digits 0-9. 50*10 = 500 features."""
    def __init__(self, length=50):
        self.length = length
    def fit(self, X):
        return self
    def _pad(self, s):
        s = s[:self.length]
        return s.rjust(self.length, '0')
    def transform(self, X):
        out = np.zeros((len(X), self.length * 10), dtype=np.float32)
        for r, s in enumerate(X):
            s = self._pad(s)
            for p, ch in enumerate(s):
                if ch.isdigit():
                    out[r, p * 10 + int(ch)] = 1.0
        return out
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ---------- deep-feature transforms ----------
def deep_flatten(X):
    return X.reshape(X.shape[0], -1)          # (N, 9984)

def deep_meanpool(X):
    return X.mean(axis=1)                      # (N, 768)

# ---------- helpers ----------
def prefix(X, y, frac):
    """First `frac` fraction of the data (rows are already in fixed order)."""
    n = int(round(len(y) * frac))
    if isinstance(X, np.ndarray):
        return X[:n], y[:n]
    return X[:n], y[:n]

def count_linear_params(clf, n_features):
    """Trainable params of a linear classifier: weights + bias."""
    return n_features + 1
