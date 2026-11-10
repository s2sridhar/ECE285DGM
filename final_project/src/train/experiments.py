import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from final_project.src.data.load import load_data
from final_project.src.eval.metrics import evaluate
from final_project.src.train.model import CVAE, reconstruction_loss

project_root = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = project_root / "rerun"
INITIAL_CHECKPOINT = project_root / "checkpoints/initial_Q_R/cvae.pt"
EPOCHS = 1000
SEED = 271
BATCH_SIZE = 128


def main():
    run_dir = OUTPUT_ROOT / "runs/generation_diagnosis"
    checkpoint_dir = OUTPUT_ROOT / "checkpoints/generation_diagnosis"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for name, beta, predict_changes in [
        ("extended", 1, False),
        ("stronger_kl", 4, False),
        ("entry_relative", 1, True),
        ("entry_relative_stronger_kl", 4, True),
    ]:
        if (checkpoint_dir / f"{name}.pt").exists():
            print("Checkpoint already exists:", name)
            continue
        torch.manual_seed(SEED)
        telemetry, context, metadata, context_mean, context_std = load_data(
            "train", device
        )
        val_x, val_c, val_metadata, _, _ = load_data("validation", device)
        model = CVAE(float(context_mean[0]), float(context_std[0]), predict_changes).to(
            device
        )
        if not predict_changes:
            initial = torch.load(
                INITIAL_CHECKPOINT, map_location=device, weights_only=False
            )
            np.testing.assert_array_equal(initial["context_mean"], context_mean)
            np.testing.assert_array_equal(initial["context_std"], context_std)
            model.load_state_dict(initial["state_dict"])

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        group_sizes = metadata.groupby(["session", "driver"]).size()
        driver_counts = metadata.groupby("session").driver.nunique()
        weights = [
            1 / (group_sizes.loc[(session, driver)] * driver_counts[session])
            for session, driver in zip(metadata.session, metadata.driver)
        ]
        sampler_generator = torch.Generator().manual_seed(SEED)
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=sampler_generator
        )
        loader = DataLoader(
            TensorDataset(telemetry.cpu(), context.cpu()),
            batch_size=BATCH_SIZE,
            sampler=sampler,
        )
        history = []
        best_score = float("inf")
        best_checkpoint = None
        start = time.time()
        for epoch in range(EPOCHS + 1):
            if epoch % 10 == 0:
                metrics, _ = evaluate(model, val_x, val_c, val_metadata)
                history.append({"epoch": epoch, **metrics})
                if metrics["energy_score"] < best_score:
                    best_score = metrics["energy_score"]
                    best_checkpoint = {
                        "state_dict": copy.deepcopy(model.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                        "context_mean": context_mean,
                        "context_std": context_std,
                        "name": name,
                        "epoch": epoch,
                        "beta": beta,
                        "anchor_entry": predict_changes,
                        "validation": metrics,
                        "seed": SEED,
                    }
                if epoch % 100 == 0:
                    print(name, epoch, json.dumps(metrics), flush=True)
            if epoch == EPOCHS:
                break
            model.train()
            for batch_x, batch_c in loader:
                batch_x, batch_c = batch_x.to(device), batch_c.to(device)
                output, kl = model(batch_x, batch_c)
                continuous, brake = reconstruction_loss(output, batch_x)
                loss = (continuous + brake + beta * kl / batch_x.size(1)).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
        assert best_checkpoint is not None
        best_checkpoint["training_seconds"] = time.time() - start
        torch.save(best_checkpoint, checkpoint_dir / f"{name}.pt")
        pd.DataFrame(history).to_csv(run_dir / f"{name}_history.csv", index=False)
        model.load_state_dict(best_checkpoint["state_dict"])
        metrics, _ = evaluate(model, val_x, val_c, val_metadata, noise_seed=19285)
        summary = {
            "name": name,
            "selected_epoch": best_checkpoint["epoch"],
            "seconds": best_checkpoint["training_seconds"],
            **metrics,
        }
        (run_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
