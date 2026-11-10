import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from final_project.src.data.load import load_data
from final_project.src.eval.metrics import evaluate
from final_project.src.train.model import CVAE

project_root = Path(__file__).resolve().parents[2]
MODEL_NAME = "entry_relative_stronger_kl"
CHECKPOINT = project_root / "checkpoints/generation_diagnosis" / f"{MODEL_NAME}.pt"
OUTPUT_DIR = project_root / "rerun/evaluation"


def main():
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x, context, metadata, _, _ = load_data("test", device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    mean, std = checkpoint["context_mean"], checkpoint["context_std"]
    model = CVAE(float(mean[0]), float(std[0]), checkpoint["anchor_entry"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, samples = evaluate(model, x, context, metadata, noise_seed=39285)
    rows = [
        {"name": MODEL_NAME, "group": "balanced_all", "n": len(metadata), **metrics}
    ]
    groups = {
        "Q": metadata.session.eq("Q"),
        "R": metadata.session.eq("R"),
        "reserved_corners": metadata.heldout_corner,
        "familiar_corners": ~metadata.heldout_corner,
        "Bahrain": metadata.event.eq("Bahrain"),
        "Spain": metadata.event.eq("Spain"),
        "Italy": metadata.event.eq("Italy"),
    }
    for group, mask in groups.items():
        selected = mask.to_numpy()
        metrics, _ = evaluate(
            model, x[selected], context[selected], metadata.loc[mask], noise_seed=39285
        )
        rows.append(
            {"name": MODEL_NAME, "group": group, "n": int(mask.sum()), **metrics}
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(rows, indent=2))
    metadata.to_parquet(OUTPUT_DIR / "test_metadata.parquet", index=False)
    np.savez_compressed(
        OUTPUT_DIR / "test_predictions.npz",
        actual=x.cpu().numpy(),
        generated=samples.numpy(),
    )
    print(
        pd.DataFrame(rows)[
            [
                "group",
                "mean_speed_mae",
                "mean_throttle_mae",
                "brake_brier",
                "energy_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
