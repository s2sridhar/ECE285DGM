import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial import KDTree

project_root = Path(__file__).resolve().parents[2]
SEASON = 2024
EVENTS = ["Bahrain", "Spain", "Italy"]
SESSIONS = ["Q", "R"]
SEED = 271
GRID_POINTS = 64
MAX_SAMPLE_GAP_S = 0.8
TRAFFIC_RADIUS_M = 200
MAX_POSITION_AGE_S = 0.6
MAX_LAP_RATIO = 1.07
OUTPUT_DIR = project_root / "data/processed/rebuilt"
BASE_FEATURES = [
    "entry_speed",
    "tyre_age",
    "fresh_at_stint_start",
    "race",
    "race_progress",
    "air_temperature",
    "track_temperature",
    "soft",
    "medium",
    "hard",
    "red_bull",
    "ferrari",
    "mclaren",
    "mercedes",
]
GEOMETRY_FEATURES = [
    "window_length",
    "previous_corner_gap",
    "next_corner_gap",
    "turn_angle",
    "absolute_turn",
    *[f"curvature_{i}" for i in range(8)],
]


def read_records(path, name):
    return pd.DataFrame(json.loads((path / f"{name}.json").read_text()))


def timestamps_to_seconds(values):
    return (
        pd.to_datetime(values, utc=True, format="mixed").astype("int64").to_numpy()
        / 1e9
    )


def assign_split(key):
    value = int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "validation"
    return "test"


def nearest_indices(t, query):
    right = np.searchsorted(t, query).clip(0, len(t) - 1)
    left = (right - 1).clip(0)
    return np.where(abs(t[left] - query) < abs(t[right] - query), left, right)


def load_stream(path, name):
    frame = read_records(path, name).sort_values("date").drop_duplicates("date")
    frame["t"] = timestamps_to_seconds(frame.date)
    return frame


def active_intervals(laps):
    result = {}
    for driver, group in laps.groupby("driver_number"):
        driver_laps = group.dropna(subset=["date_start"]).sort_values("lap_number")
        starts = timestamps_to_seconds(driver_laps.date_start)
        ends = starts + driver_laps.lap_duration.fillna(180).to_numpy()

        result[int(driver)] = np.column_stack([starts, ends])
    return result


def curate_session(path):
    manifest = json.loads((path / "manifest.json").read_text())
    kind = manifest["session"]
    laps = read_records(path, "laps")
    stints = read_records(path, "stints")
    driver_info = (
        read_records(path, "drivers")
        .drop_duplicates("driver_number")
        .set_index("driver_number")
    )
    weather = load_stream(path, "weather")
    race_control = load_stream(path, "race_control")
    pit = read_records(path, "pit")
    positions = {
        int(f.stem.split("_")[-1]): load_stream(path, f.stem)
        for f in path.glob("location_*.json")
    }
    active = active_intervals(laps)
    candidates, rejections = [], []
    for _, lap in laps[
        laps.driver_number.isin(manifest["selected_drivers"])
    ].iterrows():
        driver, number = int(lap.driver_number), int(lap.lap_number)
        key = f"{path.name}/{driver}/{number}"
        metadata = {
            "key": key,
            "event": manifest["event"],
            "session": kind,
            "driver": driver,
            "lap": number,
            "split": assign_split(key),
            "season": manifest["year"],
            "session_key": manifest["session_key"],
            "team": str(driver_info.loc[driver, "team_name"]),
        }
        reasons = []
        if pd.isna(lap.date_start) or pd.isna(lap.lap_duration):
            reasons.append("missing_lap_timing")
        if (
            lap.is_pit_out_lap
            or ((pit.driver_number == driver) & (pit.lap_number == number)).any()
        ):
            reasons.append("pit_lap")
        if kind == "R" and number <= 2:
            reasons.append("opening_laps")
        if reasons:
            rejections.append(
                {**metadata, "reasons": "|".join(reasons), "level": "lap"}
            )
            continue
        t0 = pd.to_datetime(lap.date_start, utc=True).value / 1e9
        t1 = t0 + lap.lap_duration
        stint = stints[
            (stints.driver_number == driver)
            & (stints.lap_start <= number)
            & (stints.lap_end >= number)
        ]
        if len(stint) != 1 or stint.iloc[0].compound not in ["SOFT", "MEDIUM", "HARD"]:
            reasons.append("unknown_or_wet_tyre")

        adverse = race_control[
            race_control.flag.isin(["YELLOW", "DOUBLE YELLOW", "RED"])
            | race_control.category.eq("SafetyCar")
            | race_control.message.str.contains(
                "SAFETY CAR|VIRTUAL SAFETY", case=False, na=False
            )
        ]
        if ((adverse.t >= t0 - 180) & (adverse.t <= t1)).any():
            reasons.append("flag_or_restart_window")
        if race_control.message.str.contains(
            rf"CAR {driver} .*TIME.*DELETED", case=False, na=False
        ).any():
            metadata["driver_has_deletion_message"] = True
        deleted = race_control[
            race_control.message.str.contains("DELETED", case=False, na=False)
            & (
                race_control.driver_number.eq(driver)
                | race_control.message.str.contains(rf"CAR {driver} ", na=False)
            )
        ]
        if (
            (deleted.lap_number.eq(number))
            | ((deleted.t >= t0) & (deleted.t <= t1 + 60))
        ).any():
            reasons.append("deleted_or_uncertain_lap")

        safety_start = None
        for control_time, message in race_control[["t", "message"]].itertuples(
            index=False, name=None
        ):
            message = str(message).upper()
            if "DEPLOYED" in message and "SAFETY CAR" in message:
                safety_start = control_time
            if safety_start is not None and (
                "ENDING" in message or "IN THIS LAP" in message
            ):
                if t0 <= control_time + 180 and t1 >= safety_start:
                    reasons.append("safety_car_interval")
                safety_start = None
        if safety_start is not None and t1 >= safety_start:
            reasons.append("safety_car_interval")
        same = laps[(laps.driver_number == driver) & ~laps.is_pit_out_lap]
        best = same.lap_duration.min()
        if lap.lap_duration > MAX_LAP_RATIO * best:
            reasons.append("slow_lap")
        weather_indices = (
            np.searchsorted(weather.t.to_numpy(), [t0, t1]).clip(1, len(weather)) - 1
        )
        lap_weather = weather.iloc[weather_indices[0] : weather_indices[1] + 1]
        if (
            lap_weather.empty
            or lap_weather.rainfall.isna().any()
            or lap_weather.rainfall.ne(0).any()
            or t0 - lap_weather.iloc[0].t > 120
        ):
            reasons.append("wet_or_unknown_weather")
        if reasons:
            rejections.append(
                {**metadata, "reasons": "|".join(reasons), "level": "lap"}
            )
            continue
        st = stint.iloc[0]
        metadata.update(
            t0=t0,
            t1=t1,
            lap_duration=float(lap.lap_duration),
            compound=st.compound,
            tyre_age=float(st.tyre_age_at_start + number - st.lap_start),
            fresh_at_stint_start=float(st.tyre_age_at_start == 0),
            stint=int(st.stint_number),
            air_temperature=float(lap_weather.iloc[0].air_temperature),
            track_temperature=float(lap_weather.iloc[0].track_temperature),
            race_progress=float(number / laps.lap_number.max()) if kind == "R" else 0.0,
        )
        candidates.append(metadata)
    references = [m for m in candidates if m["split"] == "train"]

    reference_attempts = []
    good = False
    for reference_lap in sorted(references, key=lambda m: m["key"]):
        pos = positions[reference_lap["driver"]]
        reference_positions = pd.DataFrame(
            pos.loc[(pos.t >= reference_lap["t0"]) & (pos.t <= reference_lap["t1"])]
        ).copy()
        map_coordinates = reference_positions[["x", "y"]].to_numpy() / 10.0
        valid = np.r_[
            True, np.linalg.norm(np.diff(map_coordinates, axis=0), axis=1) > 0.1
        ]
        map_coordinates = map_coordinates[valid]
        good = (
            len(map_coordinates) >= 100
            and np.max(np.linalg.norm(np.diff(map_coordinates, axis=0), axis=1)) <= 100
            and np.linalg.norm(map_coordinates[-1] - map_coordinates[0]) <= 100
        )
        reference_attempts.append({"key": reference_lap["key"], "accepted": bool(good)})
        if good:
            break
    assert good
    map_coordinates = np.vstack([map_coordinates, map_coordinates[0]])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(map_coordinates, axis=0), axis=1))]
    grid = np.arange(0, arc[-1], 2.0)
    mapping = np.column_stack(
        [np.interp(grid, arc, map_coordinates[:, i]) for i in range(2)]
    )
    smooth = np.asarray(savgol_filter(mapping, 31, 3, axis=0, mode="wrap"))
    heading = np.unwrap(
        np.arctan2(np.gradient(smooth[:, 1]), np.gradient(smooth[:, 0]))
    )
    curvature = np.gradient(heading, grid)
    tree = KDTree(mapping, leafsize=16)
    markers = pd.read_parquet(path / "corners.parquet")
    markers["s"] = grid[tree.query(markers[["X", "Y"]].to_numpy() / 10)[1]]
    markers = markers.sort_values("s").reset_index(drop=True)
    cars = {d: load_stream(path, f"car_data_{d}") for d in manifest["selected_drivers"]}
    accepted, sequences, contexts, masks = [], [], [], []
    for metadata in candidates:
        car = cars[metadata["driver"]]
        car = pd.DataFrame(
            car.loc[(car.t >= metadata["t0"] - 0.5) & (car.t <= metadata["t1"] + 0.5)]
        ).copy()
        pos = positions[metadata["driver"]]
        position_indices = nearest_indices(pos.t.to_numpy(), car.t.to_numpy())
        driver_coordinates = pos[["x", "y"]].to_numpy()[position_indices] / 10
        distance_to_map, projection = tree.query(driver_coordinates)
        s = grid[projection]

        mid = (car.t > metadata["t0"] + 2) & (car.t < metadata["t1"] - 2)
        if np.any(np.diff(s[mid]) < -30):
            rejections.append(
                {**metadata, "level": "lap", "reasons": "nonmonotone_position"}
            )
            continue
        for j, (_, corner) in enumerate(markers.iterrows()):
            center = corner.s
            previous_gap = center - markers.iloc[j - 1].s if j else center
            next_gap = (
                markers.iloc[j + 1].s - center
                if j + 1 < len(markers)
                else arc[-1] - center
            )
            window_start, window_end = (
                max(0, center - min(250, 0.48 * previous_gap)),
                min(arc[-1], center + min(150, 0.48 * next_gap)),
            )
            selection = (
                (s >= window_start - 10) & (s <= window_end + 10) & mid.to_numpy()
            )
            rows = pd.DataFrame(car.loc[selection])
            record = {
                **metadata,
                "corner": f"{int(corner.Number)}{corner.Letter}",
                "corner_index": int(j),
                "window_start_m": window_start,
                "window_end_m": window_end,
                "reference_key": reference_lap["key"],
                "heldout_corner": bool(j == len(markers) // 2),
            }
            reasons = []
            if len(rows) < 8 or window_end - window_start < 80:
                reasons.append("short_or_sparse_window")
            if reasons:
                rejections.append(
                    {**record, "level": "corner", "reasons": "|".join(reasons)}
                )
                continue
            t = rows.t.to_numpy()
            sample_distances = s[selection]
            if np.diff(t).max() > MAX_SAMPLE_GAP_S:
                reasons.append("car_sample_gap")
            if (
                np.max(abs(pos.t.to_numpy()[position_indices[selection]] - t))
                > MAX_POSITION_AGE_S
            ):
                reasons.append("own_position_gap")
            if (
                float(np.max(distance_to_map[selection])) > 30
                or sample_distances.min() > window_start + 10
                or sample_distances.max() < window_end - 10
            ):
                reasons.append("map_alignment")
            values = rows[["speed", "throttle", "brake"]].to_numpy(dtype=float)
            if (
                not np.isfinite(values).all()
                or np.any((values[:, 0] < 20) | (values[:, 0] > 380))
                or np.any((values[:, 1] < 0) | (values[:, 1] > 100))
                or not np.isin(values[:, 2], [0, 100]).all()
            ):
                reasons.append("invalid_channel")

            nearest = float("inf")
            for rival, rival_positions in positions.items():
                if rival == metadata["driver"]:
                    continue
                spans = active.get(rival, np.empty((0, 2)))
                is_active = (
                    (t[:, None] >= spans[:, 0]) & (t[:, None] <= spans[:, 1])
                ).any(axis=1)
                if not is_active.any():
                    continue
                rival_indices = nearest_indices(
                    rival_positions.t.to_numpy(), t[is_active]
                )
                if (
                    np.max(
                        abs(rival_positions.t.to_numpy()[rival_indices] - t[is_active])
                    )
                    > MAX_POSITION_AGE_S
                ):
                    reasons.append("unknown_rival_position")
                    break
                rival_coordinates = (
                    rival_positions[["x", "y"]].to_numpy()[rival_indices] / 10
                )
                if np.any(np.linalg.norm(rival_coordinates, axis=1) == 0):
                    reasons.append("unknown_rival_position")
                    break
                _, rival_index = tree.query(rival_coordinates)

                delta = abs(s[selection][is_active] - grid[rival_index])
                separation = np.minimum(delta, arc[-1] - delta)
                nearest = min(nearest, separation.min())
            record["nearest_rival_m"] = nearest if np.isfinite(nearest) else None
            record["max_sample_gap_s"] = float(np.diff(t).max())
            record["native_samples"] = len(rows)
            if nearest < TRAFFIC_RADIUS_M:
                reasons.append("nearby_car")
            if record["heldout_corner"] and record["split"] != "test":
                reasons.append("reserved_corner")
            if reasons:
                rejections.append(
                    {
                        **record,
                        "level": "corner",
                        "reasons": "|".join(sorted(set(reasons))),
                    }
                )
                continue

            order = np.argsort(sample_distances)
            sample_distances, values, t = (
                sample_distances[order],
                values[order],
                t[order],
            )
            unique, indices = np.unique(sample_distances, return_index=True)
            values, t = values[indices], t[indices]
            target = np.linspace(
                max(window_start, unique[0]),
                min(window_end, unique[-1]),
                GRID_POINTS,
            )
            out = np.column_stack(
                [np.interp(target, unique, values[:, k]) for k in [0, 1]]
                + [values[nearest_indices(unique, target), 2] / 100]
            )
            record["duration_s"] = float(
                np.interp(target[-1], unique, t) - np.interp(target[0], unique, t)
            )
            record["entry_speed"] = float(out[0, 0])
            features = [
                out[0, 0],
                metadata["tyre_age"],
                metadata["fresh_at_stint_start"],
                float(kind == "R"),
                metadata["race_progress"],
                metadata["air_temperature"],
                metadata["track_temperature"],
                *[float(metadata["compound"] == c) for c in ["SOFT", "MEDIUM", "HARD"]],
            ]
            features.extend(
                float(team in metadata["team"].lower())
                for team in ["red bull", "ferrari", "mclaren", "mercedes"]
            )
            geometry_mask = (grid >= window_start) & (grid <= window_end)
            geometry = [
                window_end - window_start,
                previous_gap,
                next_gap,
                np.trapezoid(curvature[geometry_mask], grid[geometry_mask]),
                np.trapezoid(abs(curvature[geometry_mask]), grid[geometry_mask]),
                *np.interp(np.linspace(window_start, window_end, 8), grid, curvature),
            ]
            if not np.isfinite(features + geometry).all():
                rejections.append(
                    {**record, "level": "corner", "reasons": "missing_context"}
                )
                continue
            mask = np.zeros(len(target), dtype=bool)
            mask[
                nearest_indices(
                    target, unique[(unique >= target[0]) & (unique <= target[-1])]
                )
            ] = True
            sequences.append(out)
            contexts.append(features + geometry)
            masks.append(mask)
            accepted.append(record)
    stats = dict(
        session=path.name,
        reference_key=reference_lap["key"],
        reference_attempts=reference_attempts,
        track_length_m=float(arc[-1]),
        accepted=len(accepted),
        rejected=len(rejections),
        qualifying_or_race=kind,
    )
    return accepted, rejections, sequences, contexts, masks, stats


def main():
    paths = sorted((project_root / "data/raw/openf1").glob("*/manifest.json"))
    assert len(paths) == len(EVENTS) * len(SESSIONS)
    output = OUTPUT_DIR
    if (output / "sequences.npz").exists():
        print("Dataset already exists:", output)
        return
    output.mkdir(parents=True, exist_ok=True)
    all_metadata = []
    all_rejections = []
    all_sequences = []
    all_context = []
    all_masks = []
    session_stats = []
    for path in paths:
        (
            metadata,
            rejections,
            sequences,
            contexts,
            masks,
            stats,
        ) = curate_session(path.parent)
        all_metadata.extend(metadata)
        all_rejections.extend(rejections)
        all_sequences.extend(sequences)
        all_context.extend(contexts)
        all_masks.extend(masks)
        session_stats.append(stats)
        print(stats, flush=True)
    pd.DataFrame(all_metadata).to_parquet(output / "metadata.parquet", index=False)
    pd.DataFrame(all_rejections).to_parquet(output / "rejections.parquet", index=False)
    np.savez_compressed(
        output / "sequences.npz",
        x=np.asarray(all_sequences, dtype=np.float32),
        c=np.asarray(all_context, dtype=np.float32),
        native_mask=np.asarray(all_masks),
    )
    (output / "schema.json").write_text(
        json.dumps(
            {
                "features": BASE_FEATURES + GEOMETRY_FEATURES,
                "base_feature_count": len(BASE_FEATURES),
                "outputs": ["speed_kmh", "throttle_percent", "brake"],
                "config": {
                    "season": SEASON,
                    "events": EVENTS,
                    "sessions": SESSIONS,
                    "seed": SEED,
                    "grid_points": GRID_POINTS,
                    "max_sample_gap_s": MAX_SAMPLE_GAP_S,
                    "traffic_radius_m": TRAFFIC_RADIUS_M,
                    "max_position_age_s": MAX_POSITION_AGE_S,
                    "max_lap_ratio": MAX_LAP_RATIO,
                },
                "traffic_radius_m": TRAFFIC_RADIUS_M,
                "sessions": session_stats,
            },
            indent=2,
        )
    )
    if all_metadata:
        frame = pd.DataFrame(all_metadata)
        counts = frame.groupby(["event", "session", "split"]).size().to_string()
        (output / "counts.txt").write_text(counts)
        print(counts)


if __name__ == "__main__":
    main()
