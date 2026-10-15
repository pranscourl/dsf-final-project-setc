"""
synth_utils.py
Helper functions for notebooks 01_prepare_synthetic and 02_train.

Creates synthetic "diabetes" respondents with a SMOTE-style method that works for survey data
(mostly 0/1 answers plus a few counts and scales). Only numpy, pandas and scikit-learn are needed.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

TARGET = "Diabetes_binary"

# Answers that are Yes/No (0/1) and answers that are counts or ordered scales
BINARY = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
          "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
          "NoDocbcCost", "DiffWalk", "Sex"]
NUMERIC = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
FEATURES = BINARY + NUMERIC


def _row_hashes(df, cols):
    """One 64-bit fingerprint per row, used to find identical rows quickly."""
    return pd.util.hash_pandas_object(df[cols].astype("int64"), index=False).to_numpy()


def generate_synthetic(minority, n_new, k=5, seed=42, avoid=None, oversample=1.4):
    """
    Create `n_new` synthetic rows from the real rows in `minority` (a DataFrame with FEATURES).

    How one synthetic person is made (SMOTE idea, adapted to survey data):
      1. pick a real person at random (the base);
      2. pick one of the base's k nearest real neighbours (the neighbour);
      3. draw a random fraction `gap` between 0 and 1;
      4. numeric answers (BMI, GenHlth, ...): base + gap * (neighbour - base), rounded to a whole number;
      5. Yes/No answers: copied from the base with probability 1 - gap, otherwise from the neighbour.

    Rows that are identical to a row in `avoid` (real rows, the test set) or to another synthetic row
    are thrown away, so every synthetic row is new. Returns a DataFrame with exactly `n_new` rows.
    """
    rng = np.random.default_rng(seed)
    m = minority[FEATURES].reset_index(drop=True)
    num = m[NUMERIC].to_numpy(float)
    bina = m[BINARY].to_numpy(float)

    # Distance between people: numeric answers standardized, Yes/No answers as 0/1
    z = (num - num.mean(axis=0)) / num.std(axis=0)
    space = np.hstack([z, bina])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(space)
    neighbours = nn.kneighbors(space, return_distance=False)[:, 1:]       # drop the person themselves

    lo, hi = num.min(axis=0), num.max(axis=0)

    # Fingerprints of rows that a synthetic row must NOT copy
    banned = set()
    if avoid is not None:
        banned |= set(_row_hashes(avoid, FEATURES).tolist())
    banned |= set(_row_hashes(m, FEATURES).tolist())

    kept, kept_hashes, have = [], set(), 0
    for _ in range(10):                                   # a few rounds are enough; normally 1
        n_cand = int((n_new - have) * oversample) + 10
        base = rng.integers(0, len(m), n_cand)
        nbr = neighbours[base, rng.integers(0, k, n_cand)]
        gap = rng.random((n_cand, 1))

        new_num = np.clip(np.rint(num[base] + gap * (num[nbr] - num[base])), lo, hi)
        take_nbr = rng.random((n_cand, len(BINARY))) < gap
        new_bin = np.where(take_nbr, bina[nbr], bina[base])

        cand = pd.DataFrame(np.hstack([new_bin, new_num]).astype("int64"), columns=BINARY + NUMERIC)[FEATURES]
        h = _row_hashes(cand, FEATURES)
        ok = np.array([(x not in banned) and (x not in kept_hashes) for x in h])
        cand, h = cand[ok], h[ok]
        first = ~pd.Series(h).duplicated().to_numpy()      # drop duplicates inside this round
        cand, h = cand[first], h[first]
        kept.append(cand)
        kept_hashes |= set(h.tolist())
        have += len(cand)
        if have >= n_new:
            break
    out = pd.concat(kept, ignore_index=True).iloc[:n_new]
    if len(out) < n_new:
        raise RuntimeError(f"Could only create {len(out)} unique synthetic rows (needed {n_new}).")
    return out
