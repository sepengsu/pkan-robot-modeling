"""Validation-only KAN grid, width, learning-rate, and epoch diagnostic."""

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
from .combined import CONDITIONS, ROOT
from .friction_validate import (_fit_neural, _predict_neural, _scenario_features,
                                SharedResidual)
from .networks import parameter_count


DEFAULT_RESULT = ROOT / "result/friction_model_validation_full_v1"
DEFAULT_OUTPUT = ROOT / "result/friction_kan_sensitivity_v1"
CONFIG_PATH = ROOT / "src/model/friction_validate_config.json"
VARIANTS = (
    ("grid9_width8", 9, 8, 25, 0.001),
    ("grid5_width16", 5, 16, 25, 0.001),
    ("grid5_width8_epoch50", 5, 8, 50, 0.001),
    ("grid5_width8_lr003", 5, 8, 25, 0.003),
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_train_validation(data_root: Path, config: dict) -> tuple[dict, dict, dict]:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("training_ready") or manifest.get("trial_chunks") != 336:
        raise ValueError("verified full dataset required")
    rng = np.random.default_rng(config["selection_seed"])
    parts = {name: {"features": [], "delta": [], "condition": []}
             for name in ("train", "validation")}
    counts = {"train": 0, "validation": 0}
    for record in manifest["trials"]:
        split = record["split"]
        if split == "test":
            continue  # Test files and labels are never opened by this diagnostic.
        if split not in parts or record["trajectory"] not in manifest["config"]["split_policy"][split]:
            raise ValueError("unexpected train/validation split")
        chunk = data_root / record["relative_npz"]
        if sha256(chunk) != record["npz_sha256"]:
            raise ValueError(f"chunk hash mismatch: {chunk}")
        with np.load(chunk, allow_pickle=False) as arrays:
            x = arrays["features"]
            y = arrays["delta_friction_nm"]
        if x.shape != (3921, 22) or y.shape != (3921, 15, 6):
            raise ValueError(f"invalid chunk shape: {chunk}")
        chosen = (np.arange(len(x)) if split == "train" else
                  np.sort(rng.choice(len(x), size=config["validation_rows_per_chunk"], replace=False)))
        parts[split]["features"].append(x[chosen])
        parts[split]["delta"].append(y[chosen])
        parts[split]["condition"].append(np.full(len(chosen), CONDITIONS.index(record["condition"]), dtype=np.int8))
        counts[split] += 1
    if counts != {"train": 252, "validation": 21}:
        raise ValueError(f"unexpected chunk counts: {counts}")
    data = {split: {field: np.concatenate(chunks) for field, chunks in fields.items()}
            for split, fields in parts.items()}
    return manifest, data, {"manifest_sha256": sha256(manifest_path),
                            "train_rows": len(data["train"]["features"]),
                            "validation_rows": len(data["validation"]["features"])}


def run(data_root: Path, prior: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"use a new result folder: {output}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest, data, provenance = _load_train_validation(data_root, config)
    scenario_features = _scenario_features(manifest)
    torch.set_num_threads(config["torch_threads"])
    saved = torch.load(prior / "KAN_seed_11.pt", map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    baseline = SharedResidual("KAN", state["x_mean"].numpy(), state["x_scale"].numpy(),
                              state["y_mean"].numpy(), state["y_scale"].numpy(), 5)
    baseline.load_state_dict(state)
    prediction = _predict_neural(baseline, data["validation"]["features"],
                                 scenario_features, config["inference_batch_size"])
    baseline_rmse = float(np.sqrt(np.mean((prediction.astype(np.float64) -
                                            data["validation"]["delta"]) ** 2)))
    with (prior / "overall.csv").open(newline="", encoding="utf-8") as stream:
        original = [row for row in csv.DictReader(stream)
                    if row["model"] == "KAN" and row["seed"] == "11" and row["split"] == "validation"][0]
    if not np.isclose(baseline_rmse, float(original["torque_rmse_nm"]), atol=1e-8):
        raise ValueError("validation selection does not match original experiment")
    output.mkdir(parents=True)
    summaries = [{"variant": "original_grid5_width8_epoch25_lr001", "seed": 11,
                  "grid_intervals": 5, "kan_width": 8, "max_epochs": 25,
                  "learning_rate": 0.001, "parameters": parameter_count(baseline),
                  "best_epoch": int(original["best_epoch"]),
                  "validation_rmse_nm": baseline_rmse,
                  "training_s": float(original["training_s"])}]
    histories = []
    with (prior / "history.csv").open(newline="", encoding="utf-8") as stream:
        histories.extend({"variant": summaries[0]["variant"], "epoch": int(row["epoch"]),
                          "validation_rmse_nm": float(row["validation_rmse_nm"])}
                         for row in csv.DictReader(stream)
                         if row["model"] == "KAN" and row["seed"] == "11")
    for name, intervals, width, epochs, lr in VARIANTS:
        variant = {**config, "spline_intervals": intervals, "kan_width": width,
                   "epochs": epochs, "learning_rate": lr}
        model, history, best_epoch, training_s = _fit_neural(
            "KAN", 11, data["train"], data["validation"], scenario_features, variant)
        best_rmse = float(history[best_epoch - 1]["validation_rmse_nm"])
        summaries.append({"variant": name, "seed": 11, "grid_intervals": intervals,
                          "kan_width": width, "max_epochs": epochs,
                          "learning_rate": lr, "parameters": parameter_count(model),
                          "best_epoch": best_epoch, "validation_rmse_nm": best_rmse,
                          "training_s": training_s})
        histories.extend({"variant": name, "epoch": int(row["epoch"]),
                          "validation_rmse_nm": float(row["validation_rmse_nm"])}
                         for row in history)
        torch.save({"variant": name, "state_dict": model.state_dict(),
                    "config": variant, "best_epoch": best_epoch,
                    "dataset_manifest_sha256": provenance["manifest_sha256"]},
                   output / f"{name}_seed11.pt")
        _write_csv(output / "validation_sensitivity.csv", summaries)
        _write_csv(output / "history.csv", histories)
        print(f"Completed {name}: T12 RMSE {best_rmse:.6f} Nm", flush=True)
    fig, axis = plt.subplots(figsize=(8.5, 4.7))
    for name in [row["variant"] for row in summaries]:
        selected = [row for row in histories if row["variant"] == name]
        axis.plot([row["epoch"] for row in selected],
                  [row["validation_rmse_nm"] for row in selected], label=name)
    axis.set(xlabel="Epoch", ylabel="T12 validation RMSE (Nm)",
             title="KAN sensitivity · full Train · seed 11")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "validation_learning_curves.png", dpi=220)
    fig.savefig(output / "validation_learning_curves.svg")
    plt.close(fig)
    result = {"source_result": str(prior), "dataset": provenance,
              "protocol": "Train T1-T11/T15 all rows; T12 fixed 2688 states for selection; original Test NPZ not opened and variants not evaluated on Test",
              "variants": summaries,
              "limitations": "one seed only; variant parameter counts and epoch budgets differ; no Test evaluation"}
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prior", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.data_root.resolve(), args.prior.resolve(), args.output.resolve())
    print(json.dumps(result["variants"], indent=2))


if __name__ == "__main__":
    main()
