"""
Task 3 - Data pre-processing for CICIDS2017 (Friday afternoon, DDoS)

Steps:
  1. Load the CSV and clean the column names
  2. Analyse the labels
  3. Drop identifier columns and the duplicated header column
  4. Handle Infinity and NaN values
  5. Drop constant (zero-variance) columns
  6. Scale every feature to [0, 1] with MinMaxScaler
  7. Rank features with SelectKBest and keep the best 20
  8. Save the reduced dataset and the full feature ranking

Usage:  python preprocess.py [path/to/csv]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler

INPUT = sys.argv[1] if len(sys.argv) > 1 else "Friday-WorkingHours-Afternoon-DDos_pcap_ISCX.csv"
OUTPUT = "ddos_top20_scaled.csv"
SCORES_OUTPUT = "feature_scores.csv"
K = 20

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv(INPUT, low_memory=False)
# CICIDS2017 headers have leading spaces (" Label", " Flow Duration"), so strip them
df.columns = df.columns.str.strip()
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------- 2. Labels
df["Label"] = df["Label"].str.strip()
print("\nLabel distribution:")
counts = df["Label"].value_counts()
for label, n in counts.items():
    print(f"  {label:<10} {n:>8,}  ({n / len(df):.1%})")

# ---------------------------------------------------------------- 3. Identifiers
# These identify a specific flow or host in the testbed rather than describe its
# behaviour. Source IP alone almost perfectly separates the classes here (the
# attacker is 172.16.0.1), so leaving them in would teach a model the lab setup.
ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]
df = df.drop(columns=ID_COLS, errors="ignore")

X = df.drop(columns="Label")
y = (df["Label"] != "BENIGN").astype(int)  # BENIGN -> 0, attack -> 1

# ---------------------------------------------------------------- 4. Infinity / NaN
# Flow Bytes/s and Flow Packets/s divide by flow duration; when the duration is 0
# CICFlowMeter writes Infinity (or NaN for 0/0). Convert Inf to NaN so both are
# handled in one place, then drop the affected rows.
X = X.replace([np.inf, -np.inf], np.nan)
bad_per_col = X.isna().sum()
bad_rows = X.isna().any(axis=1)
print("\nColumns with Infinity/NaN values:")
for col, n in bad_per_col[bad_per_col > 0].items():
    print(f"  {col:<20} {n}")
print(f"Dropping {bad_rows.sum()} rows ({bad_rows.mean():.3%} of the data)")
X, y = X[~bad_rows], y[~bad_rows]

# ---------------------------------------------------------------- 5. Redundant columns
# A column with one value everywhere carries no information, and SelectKBest
# can't score it (it returns NaN), so remove it before selection.
constant = X.columns[X.nunique() <= 1]
print(f"\nDropping {len(constant)} constant columns: {list(constant)}")
X = X.drop(columns=constant)

# Several CICFlowMeter columns are copies of another column (e.g. "Subflow Fwd Bytes"
# = "Total Length of Fwd Packets"). SelectKBest scores each column on its own, so it
# would pick both copies and waste slots in the top 20. Compare after rounding to
# 4 decimals, because a few copies differ only by float rounding in the CSV.
rounded = X.round(4)
duplicates = rounded.columns[rounded.T.duplicated()]
print(f"\nDropping {len(duplicates)} duplicated columns:")
for col in duplicates:
    twin = next(c for c in rounded.columns if c != col and rounded[c].equals(rounded[col]))
    print(f"  {col:<24} (same as {twin})")
X = X.drop(columns=duplicates)

# ---------------------------------------------------------------- 6. Min-max scaling
# x_scaled = (x - min) / (max - min), computed per column -> every feature in [0, 1]
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
print(f"\nScaled {X_scaled.shape[1]} features to "
      f"[{X_scaled.min().min():.1f}, {X_scaled.max().max():.1f}]")

# ---------------------------------------------------------------- 7. SelectKBest
# f_classif = ANOVA F-test: how far apart the class means of a feature are,
# relative to the spread within each class. Higher score = better separation.
# chi2 is a common alternative; it needs non-negative inputs, which the
# min-max scaling above guarantees.
selector = SelectKBest(score_func=f_classif, k=K)
selector.fit(X_scaled, y)

scores = (
    pd.DataFrame({"feature": X_scaled.columns, "score": selector.scores_})
    .sort_values("score", ascending=False)
    .reset_index(drop=True)
)
scores.index += 1
top_features = scores["feature"].head(K).tolist()

print(f"\nTop {K} features by ANOVA F-score:")
for rank, row in scores.head(K).iterrows():
    print(f"  {rank:>2}. {row['feature']:<28} {row['score']:>12,.1f}")

# ---------------------------------------------------------------- 8. Save
result = X_scaled[top_features].copy()
result["Label"] = y
result.to_csv(OUTPUT, index=False)
scores.to_csv(SCORES_OUTPUT, index_label="rank")
print(f"\nSaved {result.shape[0]:,} rows x {result.shape[1]} columns to {OUTPUT}")
print(f"Saved the full feature ranking to {SCORES_OUTPUT}")
