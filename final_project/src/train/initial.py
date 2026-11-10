import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from final_project.src.train.model import CVAE, reconstruction_loss

project_root = Path(__file__).resolve().parents[2]
SEED = 271
EPOCHS = 40
BATCH_SIZE = 128
OUTPUT_ROOT = project_root / "rerun"


def main():
    run_dir = OUTPUT_ROOT / "runs/initial_Q_R"
    checkpoint_dir = OUTPUT_ROOT / "checkpoints/initial_Q_R"
    if (checkpoint_dir / "cvae.pt").exists():
        print("Checkpoint already exists:", checkpoint_dir / "cvae.pt")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = project_root / "data/processed/radius_200"
    metadata = pd.read_parquet(data / "metadata.parquet")
    arrays = np.load(data / "sequences.npz")
    x = arrays["x"].copy()
    x[:, :, :2] /= 100
    context = arrays["c"]
    train_mask = metadata.split.eq("train").to_numpy()
    validation_mask = metadata.split.eq("validation").to_numpy()
    mean, std = context[train_mask].mean(0), context[train_mask].std(0).clip(0.01)
    normalized_context = (context - mean) / std
    telemetry_tensor = torch.tensor(x)
    context_tensor = torch.tensor(normalized_context)
    model = CVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_metadata = metadata[train_mask]
    group_sizes = train_metadata.groupby(["session", "driver"]).size()
    drivers_per_session = train_metadata.groupby("session").driver.nunique()
    weights = [
        1 / (group_sizes.loc[(session, driver)] * drivers_per_session[session])
        for session, driver in zip(train_metadata.session, train_metadata.driver)
    ]
    train_loader = DataLoader(
        TensorDataset(telemetry_tensor[train_mask], context_tensor[train_mask]),
        batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
    )
    history = []
    best_validation_loss = float("inf")
    best_state = None
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_total = 0.0
        epoch_continuous = 0.0
        epoch_brake = 0.0
        epoch_kld = 0.0
        num_examples = 0
        for telemetry_batch, context_batch in train_loader:
            telemetry_batch, context_batch = (
                telemetry_batch.to(device),
                context_batch.to(device),
            )
            prediction, kl = model(telemetry_batch, context_batch)
            continuous, braking = reconstruction_loss(prediction, telemetry_batch)
            recon, brake = continuous.mean(), braking.mean()
            kl = kl.mean()
            loss = recon + brake + kl / 64
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            batch_size = telemetry_batch.size(0)
            epoch_total += loss.item() * batch_size
            epoch_continuous += recon.item() * batch_size
            epoch_brake += brake.item() * batch_size
            epoch_kld += kl.item() * batch_size
            num_examples += batch_size
        model.eval()
        with torch.no_grad():
            validation_telemetry, validation_context = (
                telemetry_tensor[validation_mask].to(device),
                context_tensor[validation_mask].to(device),
            )
            mu, log_var = model.encode(validation_telemetry, validation_context)
            prediction = model.decode(mu, validation_context)
            kl = -0.5 * (1 + log_var - mu.square() - log_var.exp()).sum(1).mean()
            continuous, braking = reconstruction_loss(prediction, validation_telemetry)

            scores = []
            validation_metadata = metadata[validation_mask]
            for kind in ["Q", "R"]:
                selection = torch.tensor(
                    validation_metadata.session.eq(kind).to_numpy(), device=device
                )
                if selection.any():
                    scores.append(
                        (
                            continuous[selection].mean()
                            + braking[selection].mean()
                            + kl / 64
                        ).item()
                    )
            score = float(np.mean(scores))
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=epoch_total / num_examples,
                continuous=epoch_continuous / num_examples,
                brake=epoch_brake / num_examples,
                kl=epoch_kld / num_examples,
                validation=score,
            )
        )
        if score < best_validation_loss:
            best_validation_loss = score
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0:
            print(history[-1], flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(
        dict(
            state_dict=best_state,
            context_mean=mean,
            context_std=std,
            best_validation=best_validation_loss,
            training_seconds=time.time() - start,
        ),
        checkpoint_dir / "cvae.pt",
    )
    pd.DataFrame(history).to_csv(run_dir / "cvae_history.csv", index=False)


if __name__ == "__main__":
    main()
