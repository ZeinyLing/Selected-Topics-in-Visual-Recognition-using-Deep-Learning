import pandas as pd
import numpy as np

ens = pd.read_csv("./ensemble_clean0.50_gap0.50.csv")

params = {
    "pred_week1": {"a": 0.7, "b": 0.5, "t": 0.5},
    "pred_week2": {"a": 0.8, "b": 0.4, "t": 0.5},
    #"pred_week1": {"a": 0.65, "b": 0.45, "t": 0.5},
    #"pred_week1": {"a": 0.7706107571, "b": 0.5396658440, "t": 0.555},
    #"pred_week2": {"a": 0.8089307278, "b": 0.4456145607, "t": 0.530},
    #"pred_week3": {"a": 0.8195200042, "b": 0.3840128595, "t": 0.750},
    #"pred_week4": {"a": 0.8466602880, "b": 0.3576618927, "t": 0.810},
    #"pred_week5": {"a": 0.8595354424, "b": 0.3133853840, "t": 0.765},
}

out = ens.copy()

for col, p in params.items():
    pred = out[col].values
    pred = p["a"] * pred + p["b"]
    pred = np.clip(pred, 0, 5)
    pred[pred < p["t"]] = 0
    out[col] = pred

out.to_csv("ensemble_calibrated_affine_zero_w1_1.csv", index=False)