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

# ---------- helpers ----------
def prefix(X, y, frac):
    """First `frac` fraction of the data (rows are already in fixed order)."""
    n = int(round(len(y) * frac))
    if isinstance(X, np.ndarray):
        return X[:n], y[:n]
    return X[:n], y[:n]
