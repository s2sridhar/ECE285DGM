import json
import time
from datetime import datetime, timezone
from pathlib import Path

import fastf1.mvapi
import pandas as pd
import requests

project_root = Path(__file__).resolve().parents[2]
SEASON = 2024
EVENTS = ["Bahrain", "Spain", "Italy"]
SESSIONS = ["Q", "R"]
SELECTED = [1, 11, 16, 55, 4, 81, 44, 63]
BASE = "https://api.openf1.org/v1/"


def fetch(endpoint, params, path):
    if path.exists():
        return json.loads(path.read_text())
    time.sleep(0.6)
    response = requests.get(BASE + endpoint, params=params, timeout=90)
    response.raise_for_status()
    data = response.json()
    path.write_text(json.dumps(data))
    return data


def main():
    base = project_root / "data/raw/openf1"
    base.mkdir(parents=True, exist_ok=True)
    sessions = fetch("sessions", {"year": SEASON}, base / "sessions.json")
    for event in EVENTS:
        for kind in SESSIONS:
            name = {"Q": "Qualifying", "R": "Race"}[kind]
            candidates = [
                s
                for s in sessions
                if s["country_name"] == event and s["session_name"] == name
            ]
            if event == "Italy":
                candidates = [
                    s for s in candidates if s["circuit_short_name"] == "Monza"
                ]
            (session,) = candidates
            out = base / f"{SEASON}_{event}_{kind}"
            out.mkdir(exist_ok=True)
            params = {"session_key": session["session_key"]}
            for endpoint in [
                "drivers",
                "laps",
                "stints",
                "weather",
                "race_control",
                "pit",
            ]:
                fetch(endpoint, params, out / f"{endpoint}.json")
            drivers = json.loads((out / "drivers.json").read_text())
            for driver in sorted({d["driver_number"] for d in drivers}):
                for endpoint in (
                    ["location", "car_data"] if driver in SELECTED else ["location"]
                ):
                    path = out / f"{endpoint}_{driver}.json"
                    fetch(endpoint, {**params, "driver_number": driver}, path)
                print(f"{out.name}: driver {driver}", flush=True)
            if not (out / "corners.parquet").exists():
                circuit = fastf1.mvapi.get_circuit_info(
                    year=SEASON, circuit_key=session["circuit_key"]
                )
                assert circuit is not None
                pd.DataFrame(circuit.corners).to_parquet(
                    out / "corners.parquet", index=False
                )
            manifest = {
                **session,
                "event": event,
                "session": kind,
                "selected_drivers": SELECTED,
                "provider": "OpenF1",
                "acquired_utc": datetime.now(timezone.utc).isoformat(),
            }
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"COMPLETE {out.name}", flush=True)


if __name__ == "__main__":
    main()
