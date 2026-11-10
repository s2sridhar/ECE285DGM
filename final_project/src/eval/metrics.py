import numpy as np
import torch

NUM_SAMPLES = 32


def balanced_mean(values, metadata):
    session_means = []
    for session in ["Q", "R"]:
        mask = torch.tensor(
            metadata.session.eq(session).to_numpy(), device=values.device
        )
        if mask.any():
            session_means.append(values[mask].mean().item())
    return float(np.mean(session_means))


def predictive_samples(model, context, noise_seed):
    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    latent_noise = torch.randn(NUM_SAMPLES, len(context), 8, generator=generator).to(
        context.device
    )
    observation_noise = torch.randn(
        NUM_SAMPLES, len(context), 64, 2, generator=generator
    ).to(context.device)
    brake_noise = torch.rand(NUM_SAMPLES, len(context), 64, generator=generator).to(
        context.device
    )
    means = []
    for sample in range(NUM_SAMPLES):
        output = model.decode(latent_noise[sample], context)
        output[:, :, 2] = output[:, :, 2].sigmoid()
        means.append(output)
    means = torch.stack(means)
    draws = means.clone()
    draws[:, :, :, :2] += 0.1 * observation_noise
    draws[:, :, :, 2] = (brake_noise < means[:, :, :, 2]).float()
    return means, draws


def evaluate(model, telemetry, context, metadata, noise_seed=9285):
    model.eval()
    with torch.inference_mode():
        means, draws = predictive_samples(model, context, noise_seed)

        sample_vectors = draws.permute(1, 0, 2, 3).flatten(2)
        target_vectors = telemetry.flatten(1)
        target_distance = torch.linalg.vector_norm(
            sample_vectors - target_vectors[:, None], dim=2
        ).mean(1)
        pair_distance = torch.cdist(sample_vectors, sample_vectors).sum((1, 2))
        pair_distance /= NUM_SAMPLES * (NUM_SAMPLES - 1)
        energy = (target_distance - 0.5 * pair_distance) / np.sqrt(
            target_vectors.size(1)
        )

        mean_prediction = means.mean(0)
        absolute_error = (mean_prediction - telemetry).abs()
        per_draw_error = (means - telemetry[None]).abs()
        sample_sd = means.std(0)
        lower = torch.quantile(draws[:, :, :, :2], 0.05, dim=0)
        upper = torch.quantile(draws[:, :, :, :2], 0.95, dim=0)
        covered = (
            (telemetry[:, :, :2] >= lower) & (telemetry[:, :, :2] <= upper)
        ).float()
        result = {
            "energy_score": balanced_mean(energy, metadata),
            "mean_speed_mae": balanced_mean(absolute_error[:, :, 0].mean(1), metadata)
            * 100,
            "mean_throttle_mae": balanced_mean(
                absolute_error[:, :, 1].mean(1), metadata
            )
            * 100,
            "draw_speed_mae": balanced_mean(
                per_draw_error[:, :, :, 0].mean((0, 2)), metadata
            )
            * 100,
            "brake_brier": balanced_mean(
                (mean_prediction[:, :, 2] - telemetry[:, :, 2]).square().mean(1),
                metadata,
            ),
            "speed_latent_sd": balanced_mean(sample_sd[:, :, 0].mean(1), metadata)
            * 100,
            "throttle_latent_sd": balanced_mean(sample_sd[:, :, 1].mean(1), metadata)
            * 100,
            "speed_90_coverage": balanced_mean(covered[:, :, 0].mean(1), metadata),
            "speed_90_width": balanced_mean((upper - lower)[:, :, 0].mean(1), metadata)
            * 100,
            "throttle_invalid_draw_fraction": float(
                ((draws[:, :, :, 1] < 0) | (draws[:, :, :, 1] > 1)).float().mean()
            ),
        }
        mu, log_var = model.encode(telemetry, context)
        kl = -0.5 * (1 + log_var - mu.square() - log_var.exp()).sum(1)
        reconstruction = model.decode(mu, context)
        result["kl_nats"] = balanced_mean(kl, metadata)
        result["reconstruction_speed_mae"] = 100 * balanced_mean(
            (reconstruction[:, :, 0] - telemetry[:, :, 0]).abs().mean(1), metadata
        )
        return result, means.cpu()
