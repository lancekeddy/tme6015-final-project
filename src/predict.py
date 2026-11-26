"""
predict.py - classify one building's load-shape archetype from 12 monthly bills.

Loads a trained Keras model (dense or 1-D CNN) and models/preprocessing.json, rebuilds the
55 input features exactly as make_features() in notebooks/01_models.ipynb does, standardizes
them with the TRAINING mean and std, and prints:
  - the probability of each of the 4 archetypes
  - the predicted archetype (most probable class)
  - p_shaveable = P(daytime_weekday) + P(weekday_moderate)
  - the shaveable decision at the model's tuned threshold stored in preprocessing.json

Usage (from the repository root):
    python src/predict.py examples/example_building.json
    python src/predict.py examples/example_building.json --model models/cnn_tuned.keras

Input JSON keys:
    kwh             list of 12 monthly energy values, kWh, January..December
    peak_kw_billed  list of 12 monthly billed peak demand values, kW
    sq_ft           floor area, square feet
    lat             latitude, degrees
    resolution_min  meter interval the bills were built from (15 or 60); 60 sets hourly_source = 1
    building_type   one of the training types, e.g. "office_commercial" (see preprocessing.json)
    region          one of the training regions, e.g. "US_Mountain" (see preprocessing.json)

From Python:
    from predict import load, predict_building
    model, prep = load("models/cnn_tuned.keras", "models/preprocessing.json")
    result = predict_building(building_dict, model, prep, "cnn_tuned")
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = range(1, 13)
REPO = Path(__file__).resolve().parent.parent
REQUIRED = ["kwh", "peak_kw_billed", "sq_ft", "lat", "resolution_min", "building_type", "region"]


def make_features(d):
    """Same feature block as make_features() in notebooks/01_models.ipynb.

    d: DataFrame with kwh_m01..m12, peak_kw_billed_m01..m12, sq_ft, lat, resolution_min,
    building_type, region. Returns one row of features per row of d (before standardization).
    """
    kwh = d[[f"kwh_m{m:02d}" for m in MONTHS]].values
    pk = d[[f"peak_kw_billed_m{m:02d}" for m in MONTHS]].values
    f = pd.DataFrame({
        **{f"kwh_share_m{m:02d}": kwh[:, i] / kwh.sum(1) * 12 for i, m in enumerate(MONTHS)},   # seasonality
        **{f"peak_rel_m{m:02d}": pk[:, i] / pk.max(1) for i, m in enumerate(MONTHS)},          # peak pattern
        **{f"lf_m{m:02d}": kwh[:, i] / (pk[:, i] * 730) for i, m in enumerate(MONTHS)},        # monthly load factor
        "log_annual_kwh": np.log10(kwh.sum(1)),
        "log_sq_ft": np.log10(d.sq_ft.clip(lower=100)).values,
        "lat": d.lat.values,
        "hourly_source": (d.resolution_min == 60).astype(float).values,
    })
    # one 0/1 column per building type and per region (one-hot encoding)
    f = pd.concat([f, pd.get_dummies(d.building_type, prefix="type").astype(float).reset_index(drop=True),
                   pd.get_dummies(d.region, prefix="region").astype(float).reset_index(drop=True)], axis=1)
    return f.fillna(0)


def building_to_row(b):
    """Turn the input dict (lists of 12 values + metadata) into the one-row wide table the notebook uses."""
    missing = [k for k in REQUIRED if k not in b]
    if missing:
        raise ValueError(f"input is missing keys: {missing}")
    if len(b["kwh"]) != 12 or len(b["peak_kw_billed"]) != 12:
        raise ValueError("kwh and peak_kw_billed must each have 12 monthly values")
    return pd.DataFrame([{**{f"kwh_m{m:02d}": float(b["kwh"][m - 1]) for m in MONTHS},
                          **{f"peak_kw_billed_m{m:02d}": float(b["peak_kw_billed"][m - 1]) for m in MONTHS},
                          **{k: b[k] for k in ["sq_ft", "lat", "resolution_min", "building_type", "region"]}}])


def load(model_path, prep_path):
    """Load a saved .keras model and the preprocessing file written by the notebook."""
    import keras   # imported here so the feature code can be used without TensorFlow installed
    with open(prep_path) as fh:
        prep = json.load(fh)
    return keras.models.load_model(model_path), prep


def predict_building(b, model, prep, model_name):
    """Return class probabilities, predicted archetype, p_shaveable and the shaveable decision for one building."""
    cols = prep["feature_columns"]
    row = building_to_row(b)

    # Warn if a category was not seen in training: its one-hot column does not exist, so all its 0/1 columns are 0
    for prefix, key in [("type", "building_type"), ("region", "region")]:
        if f"{prefix}_{b[key]}" not in cols:
            known = sorted(c[len(prefix) + 1:] for c in cols if c.startswith(prefix + "_"))
            print(f"warning: {key} '{b[key]}' was not in the training data; known values: {known}", file=sys.stderr)

    # Same column order as training; one-hot columns for categories this building does not have are 0
    f = make_features(row).reindex(columns=cols, fill_value=0)
    # Standardize with the TRAINING mean and std saved by the notebook
    xz = ((f - np.array(prep["train_mean"])) / np.array(prep["train_std"])).values.astype("float32")

    if len(model.inputs) == 2:
        # 1-D CNN: input A = (1, 12 months, 2 channels: kWh share, relative peak), input B = all other features
        col = {c: i for i, c in enumerate(cols)}
        share = xz[:, [col[f"kwh_share_m{m:02d}"] for m in MONTHS]]
        peak = xz[:, [col[f"peak_rel_m{m:02d}"] for m in MONTHS]]
        seq = np.stack([share, peak], axis=-1)
        rest = xz[:, [col[c] for c in prep["rest_cols"]]]
        proba = model.predict([seq, rest], verbose=0)[0]
    else:
        # dense net: all 55 standardized features in one vector
        proba = model.predict(xz, verbose=0)[0]

    classes = prep["classes"]
    p_shave = float(sum(proba[classes.index(c)] for c in prep["shaveable"]))
    threshold = float(prep["thresholds"][model_name])
    return {
        "model": model_name,
        "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
        "predicted_archetype": classes[int(np.argmax(proba))],
        "p_shaveable": round(p_shave, 3),
        "threshold": round(threshold, 3),
        "shaveable": p_shave >= threshold,
    }


def main():
    ap = argparse.ArgumentParser(description="Predict a building's load-shape archetype from 12 monthly bills.")
    ap.add_argument("building", help="path to a building JSON file, e.g. examples/example_building.json")
    ap.add_argument("--model", default=str(REPO / "models/cnn_tuned.keras"),
                    help="path to a .keras model (default: models/cnn_tuned.keras)")
    ap.add_argument("--prep", default=str(REPO / "models/preprocessing.json"),
                    help="path to preprocessing.json (default: models/preprocessing.json)")
    args = ap.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow start-up messages
    with open(args.building) as fh:
        building = json.load(fh)
    model_name = Path(args.model).stem                    # "dense_tuned" or "cnn_tuned", the key in thresholds
    model, prep = load(args.model, args.prep)
    r = predict_building(building, model, prep, model_name)

    print("model:", r["model"])
    print("class probabilities:")
    for c, p in r["probabilities"].items():
        print(f"  {c:<18} {p:.3f}")
    print("predicted archetype:", r["predicted_archetype"])
    print(f"p_shaveable = {r['p_shaveable']:.3f} -> {'shaveable' if r['shaveable'] else 'not shaveable'} "
          f"at the tuned threshold {r['threshold']:.3f}")


if __name__ == "__main__":
    main()
