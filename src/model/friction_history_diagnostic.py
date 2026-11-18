"""Test whether causal velocity history explains synthetic LuGre friction.

Fits jointwise ridge models on bare condition T1-T11/T15 only and evaluates
T12. The LuGre labels are generated identically across conditions for a given
trajectory/trial, so using all seven would repeat the same friction examples.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .collect_friction import DEFAULT_OUTPUT as DEFAULT_DATA, sha256
from .combined import ROOT
from .networks import SplineBasis


DEFAULT_OUTPUT = ROOT / "result/friction_history_diagnostic_v2"
LAGS = (1, 2, 5, 10, 20)
WARMUP_ROWS = max(LAGS)


def _design(v: np.ndarray, history: bool) -> np.ndarray:
    indices = np.arange(len(v))
    parts = [np.ones(len(v)), np.tanh(v / 0.005), v]
    if history:
        for lag in LAGS:
            past = v[np.maximum(indices - lag, 0)]
            parts.extend((np.tanh(past / 0.005), past))
    return np.column_stack(parts).astype(np.float64)


def _spline_design(v: np.ndarray, mean: float, scale: float,
                   basis: SplineBasis) -> np.ndarray:
    with torch.inference_mode():
        shape = basis(torch.from_numpy(((v - mean) / scale).astype(np.float32)[:, None]))
    return np.column_stack((np.ones(len(v)), shape[:, 0, :].numpy())).astype(np.float64)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(data_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"use a new result folder: {output}")
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("training_ready") or len(manifest["trials"]) != 336:
        raise ValueError("full verified dataset required")
    records = [record for record in manifest["trials"]
               if record["condition"] == "bare_p0" and record["split"] in ("train", "validation")]
    if len(records) != 39:
        raise ValueError("expected 36 train and 3 validation bare-condition chunks")
    gram = {kind: np.zeros((6, 3 if kind == "static" else 13,
                            3 if kind == "static" else 13), dtype=np.float64)
            for kind in ("static", "lagged", "static_spline13")}
    rhs = {kind: np.zeros((6, gram[kind].shape[1], 3), dtype=np.float64)
           for kind in gram}
    validation = []
    train_chunks = []
    for record in records:
        chunk = data_root / record["relative_npz"]
        if sha256(chunk) != record["npz_sha256"]:
            raise ValueError(f"chunk hash mismatch: {chunk}")
        with np.load(chunk, allow_pickle=False) as saved:
            velocity = saved["features"][:, 6:12].astype(np.float64)
            target = saved["delta_friction_nm"][:, 12:15, :].astype(np.float64)
        if record["split"] == "validation":
            validation.append((record, velocity, target))
            continue
        train_chunks.append((velocity, target))
        for joint in range(6):
            y = target[WARMUP_ROWS:, :, joint]
            for kind in ("static", "lagged"):
                x = _design(velocity[:, joint], kind == "lagged")[WARMUP_ROWS:]
                gram[kind][joint] += x.T @ x
                rhs[kind][joint] += x.T @ y
    train_velocity = np.concatenate([chunk[0] for chunk in train_chunks])
    velocity_mean = train_velocity.mean(axis=0)
    velocity_scale = np.maximum(train_velocity.std(axis=0), 0.01)
    basis = SplineBasis(intervals=9)
    if basis.n_basis + 1 != 13:
        raise ValueError("spline parameter count must match lagged regression")
    for velocity, target in train_chunks:
        for joint in range(6):
            y = target[WARMUP_ROWS:, :, joint]
            x = _spline_design(velocity[:, joint], velocity_mean[joint],
                               velocity_scale[joint], basis)[WARMUP_ROWS:]
            gram["static_spline13"][joint] += x.T @ x
            rhs["static_spline13"][joint] += x.T @ y
    coefficients = {}
    for kind in gram:
        coefficients[kind] = np.empty_like(rhs[kind])
        for joint in range(6):
            ridge = np.eye(gram[kind].shape[1]) * 1e-4
            ridge[0, 0] = 0
            coefficients[kind][joint] = np.linalg.solve(gram[kind][joint] + ridge,
                                                         rhs[kind][joint])
    output.mkdir(parents=True)
    np.savez_compressed(output / "fits.npz", static=coefficients["static"],
                        lagged=coefficients["lagged"],
                        static_spline13=coefficients["static_spline13"],
                        velocity_mean=velocity_mean, velocity_scale=velocity_scale)
    scores = []
    by_trial = []
    for kind in gram:
        all_squared = []
        for record, velocity, target in validation:
            squared = np.empty((len(velocity) - WARMUP_ROWS, 3, 6), dtype=np.float64)
            for joint in range(6):
                x = (_spline_design(velocity[:, joint], velocity_mean[joint],
                                    velocity_scale[joint], basis)[WARMUP_ROWS:]
                     if kind == "static_spline13" else
                     _design(velocity[:, joint], kind == "lagged")[WARMUP_ROWS:])
                predicted = x @ coefficients[kind][joint]
                squared[:, :, joint] = (predicted - target[WARMUP_ROWS:, :, joint]) ** 2
            all_squared.append(squared)
            by_trial.append({"model": kind, "trial": record["trial"],
                             "rows": len(squared), "rmse_nm": float(np.sqrt(squared.mean()))})
        merged = np.concatenate(all_squared)
        descriptor = ("current velocity" if kind == "static" else
                      "current velocity spline grid9" if kind == "static_spline13" else
                      "current and causal velocity lags 0.1, 0.2, 0.5, 1, 2 seconds")
        scores.append({"model": kind, "input": descriptor,
                       "parameters": int(coefficients[kind].size),
                       "train_trajectories": 12, "validation_trajectory": "T12",
                       "validation_trials": 3, "validation_state_rows": len(merged),
                       "validation_rmse_nm": float(np.sqrt(merged.mean()))})
        for joint in range(6):
            scores.append({"model": kind, "input": f"joint_{joint+1}",
                           "parameters": int(coefficients[kind].shape[1] * 3),
                           "train_trajectories": 12, "validation_trajectory": "T12",
                           "validation_trials": 3, "validation_state_rows": len(merged),
                           "validation_rmse_nm": float(np.sqrt(merged[:, :, joint].mean()))})
    _write_csv(output / "validation.csv", scores)
    _write_csv(output / "by_trial.csv", by_trial)
    joint_static = [float(row["validation_rmse_nm"]) for row in scores
                    if row["model"] == "static" and row["input"].startswith("joint_")]
    joint_lagged = [float(row["validation_rmse_nm"]) for row in scores
                    if row["model"] == "lagged" and row["input"].startswith("joint_")]
    joint_spline = [float(row["validation_rmse_nm"]) for row in scores
                    if row["model"] == "static_spline13" and row["input"].startswith("joint_")]
    fig, axis = plt.subplots(figsize=(8, 4.2))
    positions = np.arange(6)
    axis.bar(positions - 0.24, joint_static, width=0.24, label="current velocity")
    axis.bar(positions, joint_spline, width=0.24, label="current spline (same parameters)")
    axis.bar(positions + 0.24, joint_lagged, width=0.24, label="current + past velocity")
    axis.set_xticks(positions, [f"J{i}" for i in range(1, 7)])
    axis.set(ylabel="T12 LuGre validation RMSE (Nm)",
             title="Causal velocity history diagnostic · bare condition")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "validation_by_joint.png", dpi=220)
    fig.savefig(output / "validation_by_joint.svg")
    plt.close(fig)
    overall = {row["model"]: row["validation_rmse_nm"] for row in scores
               if row["input"] in ("current velocity", "current velocity spline grid9",
                                   "current and causal velocity lags 0.1, 0.2, 0.5, 1, 2 seconds")}
    summary = {"dataset_manifest_sha256": sha256(manifest_path),
               "target": "synthetic LuGre-style friction delta, levels low/nominal/high",
               "split": "bare condition train T1-T11/T15; validation T12 only; no Test access",
               "lag_steps_at_10hz": list(LAGS), "warmup_rows_excluded_per_chunk": WARMUP_ROWS,
               "overall_validation_rmse_nm": overall,
               "relative_rmse_reduction_percent": 100 * (1 - overall["lagged"] / overall["static"])}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.data_root.resolve(), args.output.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
