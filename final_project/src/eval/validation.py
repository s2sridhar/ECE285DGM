from pathlib import Path

import pandas as pd
import torch

from final_project.src.data.load import load_data
from final_project.src.eval.metrics import evaluate
from final_project.src.train.model import CVAE

project_root = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = project_root / "checkpoints/generation_diagnosis"
OUTPUT_DIR = project_root / "rerun/evaluation"


def main():
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x, context, metadata, mean, std = load_data("validation", device)
    rows = []
    for name in [
        "extended",
        "stronger_kl",
        "entry_relative",
        "entry_relative_stronger_kl",
    ]:
        checkpoint = torch.load(
            CHECKPOINT_DIR / f"{name}.pt", map_location=device, weights_only=False
        )
        model = CVAE(float(mean[0]), float(std[0]), checkpoint["anchor_entry"]).to(
            device
        )
        model.load_state_dict(checkpoint["state_dict"])
        metrics, _ = evaluate(model, x, context, metadata)
        rows.append({"name": name, "epoch": checkpoint["epoch"], **metrics})
    results = pd.DataFrame(rows).sort_values("energy_score")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    print(results[["name", "epoch", "energy_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
