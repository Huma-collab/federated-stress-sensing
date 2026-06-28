"""
All-classifiers comparison script.

Uses data/dataset_combined.py's Data class -- confirmed to produce
paper-consistent labels. Per-channel TSFEL feature extraction, then
trains every classifier from the paper's stated Methodology
(Section 3.1) on the same correctly-built features, with fair
class-imbalance handling where each model supports it.
"""
import os
import numpy as np
import pandas as pd
import tsfel
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier

os.environ["TQDM_DISABLE"] = "1"

from config import settings
from data.dataset_combined import Data

random_state = 42


def to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


channel_keys = [
    "hourly_act", "hourly_loc", "hourly_unlock",
    "daily_loc_mov", "daily_loc_still", "daily_unlock", "daily_sleep",
]

train_channel_samples = {k: [] for k in channel_keys}
test_channel_samples = {k: [] for k in channel_keys}
train_labels = []
test_labels = []
skipped_users = []

total_users = len(settings.USERS)
print(f"Loading data for {total_users} users (per-user chronological split)...")

for i, user in enumerate(settings.USERS):
    try:
        print(f"[{i+1}/{total_users}] {user[:8]}...")
        dataset = Data([user], settings.ALL_HEADINGS_W_HR)
        n_samples = len(dataset.stress)
        if n_samples == 0:
            skipped_users.append(user)
            continue

        split_idx = int(0.6 * n_samples)
        user_indices = list(range(n_samples))
        train_user_idx = set(user_indices[:split_idx])
        test_user_idx = set(user_indices[split_idx:])

        for s in range(n_samples):
            try:
                label_tensor = dataset.stress[s]
                label = float(to_np(label_tensor))
                if label < 0:
                    continue

                act = to_np(dataset.hourly_act_features[s]).flatten()
                loc = to_np(dataset.hourly_loc_features[s]).flatten()
                unlock = to_np(dataset.hourly_unlock_features[s]).flatten()
                dmov = to_np(dataset.daily_loc_mov_features[s]).flatten()
                dstill = to_np(dataset.daily_loc_still_features[s]).flatten()
                dunlock = to_np(dataset.daily_unlock_features[s]).flatten()
                dsleep = to_np(dataset.daily_sleep_features[s]).flatten()

                target_samples = train_channel_samples if s in train_user_idx else test_channel_samples
                target_samples["hourly_act"].append(act)
                target_samples["hourly_loc"].append(loc)
                target_samples["hourly_unlock"].append(unlock)
                target_samples["daily_loc_mov"].append(dmov)
                target_samples["daily_loc_still"].append(dstill)
                target_samples["daily_unlock"].append(dunlock)
                target_samples["daily_sleep"].append(dsleep)

                if s in train_user_idx:
                    train_labels.append(int(label))
                else:
                    test_labels.append(int(label))
            except Exception as e:
                print(f"  sample {s} skipped: {e}")
                continue
    except Exception as e:
        print(f"  USER {user[:8]} FAILED: {e}")
        skipped_users.append(user)
        continue

n_train = len(train_labels)
n_test = len(test_labels)
print(f"\nTrain samples: {n_train}, Test samples: {n_test}")
print(f"Skipped users: {len(skipped_users)}")

if n_train == 0 or n_test == 0:
    raise RuntimeError("No samples collected in train or test split.")

y_train = np.array(train_labels)
y_test = np.array(test_labels)
print(f"Train label distribution: {np.bincount(y_train)}")
print(f"Train class balance: {np.bincount(y_train) / len(y_train)}")
print(f"Test label distribution: {np.bincount(y_test)}")
print(f"Test class balance: {np.bincount(y_test) / len(y_test)}")

print("\nPadding per-channel sample arrays to consistent length...")
train_channel_matrices = {}
test_channel_matrices = {}
for ck in channel_keys:
    train_samples = train_channel_samples[ck]
    test_samples = test_channel_samples[ck]
    max_len = max(
        max(len(s) for s in train_samples),
        max(len(s) for s in test_samples),
    )

    def pad_to(s, L):
        if len(s) < L:
            return np.pad(s, (0, L - len(s)), mode="constant", constant_values=0)
        return s[:L]

    train_matrix = np.array([pad_to(s, max_len) for s in train_samples])
    test_matrix = np.array([pad_to(s, max_len) for s in test_samples])
    train_channel_matrices[ck] = train_matrix
    test_channel_matrices[ck] = test_matrix
    print(f"  {ck}: train shape {train_matrix.shape}, test shape {test_matrix.shape}")

cfg_file = tsfel.get_features_by_domain(["statistical", "temporal"])

train_feat_blocks = []
test_feat_blocks = []
for ck in channel_keys:
    print(f"Extracting TSFEL features for channel: {ck}")
    train_signal = train_channel_matrices[ck]
    test_signal = test_channel_matrices[ck]

    train_rows = []
    for row in train_signal:
        f = tsfel.time_series_features_extractor(cfg_file, row, fs=1, verbose=0)
        train_rows.append(f)
    train_feats = pd.concat(train_rows, axis=0, ignore_index=True)

    test_rows = []
    for row in test_signal:
        f = tsfel.time_series_features_extractor(cfg_file, row, fs=1, verbose=0)
        test_rows.append(f)
    test_feats = pd.concat(test_rows, axis=0, ignore_index=True)

    train_feats.columns = [f"{ck}_{c}" for c in train_feats.columns]
    test_feats.columns = [f"{ck}_{c}" for c in test_feats.columns]
    train_feats.fillna(0, inplace=True)
    test_feats.fillna(0, inplace=True)

    train_feat_blocks.append(train_feats)
    test_feat_blocks.append(test_feats)
    print(f"  {ck} done: train shape {train_feats.shape}")

X_train_feats = pd.concat(train_feat_blocks, axis=1)
X_test_feats = pd.concat(test_feat_blocks, axis=1)
print(f"\nCombined feature matrix shape (train): {X_train_feats.shape}")

corr_features, X_train_feats = tsfel.correlated_features(X_train_feats, drop_correlated=True)
X_test_feats.drop(corr_features, axis=1, inplace=True, errors="ignore")

selector = VarianceThreshold()
X_train_np = selector.fit_transform(X_train_feats)
X_test_np = selector.transform(X_test_feats)

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

print(f"Final feature count after filtering: {X_train_scaled.shape[1]}")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier

n_neg = int(np.sum(y_train == 0))
n_pos = int(np.sum(y_train == 1))
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
print(f"\nTrain class counts: class0={n_neg}, class1={n_pos}")
print(f"scale_pos_weight / class weight ratio: {scale_pos_weight:.4f}")

results = {}

def evaluate(name, model):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    results[name] = (acc, f1)
    print(f"\n{'='*60}")
    print(f"{name} RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))

evaluate("KNN (k=250, Manhattan)", KNeighborsClassifier(
    n_neighbors=250, metric="manhattan", weights="distance"))

evaluate("Decision Tree (depth=10, class_weight=balanced)", DecisionTreeClassifier(
    max_depth=10, class_weight="balanced", random_state=random_state))

evaluate("Random Forest (500 trees, class_weight=balanced)", RandomForestClassifier(
    n_estimators=500, class_weight="balanced", random_state=random_state, n_jobs=-1))

evaluate("AdaBoost", AdaBoostClassifier(random_state=random_state))

evaluate("XGBoost (scale_pos_weight)", XGBClassifier(
    n_estimators=500, random_state=random_state, scale_pos_weight=scale_pos_weight))

evaluate("Ridge Classifier (class_weight=balanced)", RidgeClassifier(
    class_weight="balanced", random_state=random_state))

print(f"\n\n{'='*60}")
print("SUMMARY: ALL MODELS, SAME CORRECT PIPELINE")
print(f"{'='*60}")
print(f"{'Model':<45} {'Accuracy':>10} {'Wtd. F1':>10}")
for name, (acc, f1) in sorted(results.items(), key=lambda x: -x[1][1]):
    print(f"{name:<45} {acc:>10.4f} {f1:>10.4f}")
best_model = max(results.items(), key=lambda x: x[1][1])
print(f"\nBEST MODEL BY WEIGHTED F1: {best_model[0]} "
      f"(Accuracy={best_model[1][0]:.4f}, F1={best_model[1][1]:.4f})")
