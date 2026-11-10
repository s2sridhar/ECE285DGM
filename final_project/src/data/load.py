from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data/processed/radius_200"


def load_data(split, device):
    metadata = pd.read_parquet(data_dir / "metadata.parquet")
    arrays = np.load(data_dir / "sequences.npz")
    training = metadata.split.eq("train").to_numpy()
    context_mean = arrays["c"][training].mean(0)
    context_std = arrays["c"][training].std(0).clip(0.01)
    selected = metadata.split.eq(split).to_numpy()
    telemetry = arrays["x"][selected].copy()
    telemetry[:, :, :2] /= 100
    context = (arrays["c"][selected] - context_mean) / context_std
    return (
        torch.tensor(telemetry).to(device),
        torch.tensor(context).to(device),
        metadata.loc[selected].reset_index(drop=True),
        context_mean,
        context_std,
    )
