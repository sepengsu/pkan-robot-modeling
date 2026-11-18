"""Evaluate both simulated physical settings and export reproducible artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import load_prepared
from .train import load_branch


def metrics(actual: np.ndarray, predicted: np.ndarray, baseline_rmse: float | None = None) -> dict:
    error = predicted.astype(np.float64) - actual.astype(np.float64)
    result = {"samples": len(error), "rmse_nm": float(np.sqrt(np.mean(error ** 2))),
              "mae_nm": float(np.mean(np.abs(error))),
              "per_joint_rmse_nm": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
              "max_abs_error_nm": float(np.max(np.abs(error)))}
    if baseline_rmse is not None:
        result["relative_improvement_over_M1_percent"] = 100 * (1 - result["rmse_nm"] / baseline_rmse) if baseline_rmse > 0 else None
    return result


def _predict(model, features: torch.Tensor, batch_size: int = 1024) -> tuple[np.ndarray, float]:
    with torch.inference_mode():
        _ = model(features[:min(1024, len(features))])
        start = time.perf_counter()
        prediction = torch.cat([model(batch) for batch in features.split(batch_size)]).numpy()
        duration = time.perf_counter() - start
    return prediction, duration / len(features) * 1e6


def evaluate(data_root: Path, model_root: Path, output: Path) -> dict:
    manifest, arrays = load_prepared(data_root)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    features = torch.from_numpy(arrays["features"].astype(np.float32, copy=False))
    target_a, target_b = arrays["tau_reference"], arrays["tau_friction"]
    m0, m1 = arrays["tau_nominal"], target_a
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(config.get("threads", 1))
    predictions = {"M0": m0, "M1": m1}
    metadata = {"M0": {"parameters": 0, "training_s": 0, "inference_us_per_sample": None},
                "M1": {"parameters": 0, "training_s": 0, "inference_us_per_sample": None}}
    for seed in config["seeds"]:
        seed_dir = model_root / f"seed_{seed}"
        for kind in ("M2", "M3", "M4", "M5"):
            model, ckpt = load_branch(seed_dir / f"{kind}.pt")
            residual, latency = _predict(model, features)
            joint = np.zeros_like(residual)
            coupled = np.zeros_like(residual)
            if kind == "M5":
                joint_model, joint_ckpt = load_branch(seed_dir / "M3.pt")
                joint, joint_latency = _predict(joint_model, features)
                coupled = residual
                residual = joint + coupled
                latency += joint_latency
                parameters = ckpt["total_parameter_count"]
                duration = ckpt["training_s"] + joint_ckpt["training_s"]
            else:
                parameters = ckpt["parameter_count"]
                duration = ckpt["training_s"]
            name = f"{kind}_seed_{seed}"
            predictions[name] = m1 + residual
            metadata[name] = {"parameters": parameters, "training_s": duration,
                              "inference_us_per_sample": latency, "best_epoch": ckpt["best_epoch"]}
            if kind == "M5" and seed == config["seeds"][0]:
                np.savez_compressed(output / "branch_contributions.npz", jointwise_nm=joint,
                                    coupled_nm=coupled, target_residual_nm=target_b - m1,
                                    features=arrays["features"], split=arrays["split"],
                                    condition=arrays["condition"], trajectory=arrays["trajectory"])
            if kind == "M3":
                basis = model.network.basis
                spline = {"coefficients": model.network.coefficients.detach().numpy().tolist(),
                          "bias": model.network.bias.detach().numpy().tolist(),
                          "knots": basis.knots.numpy().tolist(), "degree": basis.degree,
                          "input_mean": model.x_mean.numpy().tolist(),
                          "input_scale": model.x_scale.numpy().tolist(),
                          "output_mean": model.y_mean.numpy().tolist(),
                          "output_scale": model.y_scale.numpy().tolist()}
                (output / f"jointwise_spline_seed_{seed}.json").write_text(json.dumps(spline, indent=2) + "\n", encoding="utf-8")
    # Save one seed's predictions for figure regeneration, not all rows per seed.
    first = config["seeds"][0]
    np.savez_compressed(output / "predictions_seed_first.npz", target_A=target_a, target_B=target_b,
                        **{name: value for name, value in predictions.items() if name in ("M0", "M1") or name.endswith(f"seed_{first}")},
                        features=arrays["features"], split=arrays["split"],
                        condition=arrays["condition"], trajectory=arrays["trajectory"])
    records = []
    split_codes = {k: v for k, v in manifest["split_codes"].items() if k != "diagnostic"}
    for scenario, target in (("A", target_a), ("B", target_b)):
        names = ("M0", "M1") if scenario == "A" else tuple(predictions)
        for split, code in split_codes.items():
            split_mask = arrays["split"] == code
            scopes = [("all", "all", split_mask)]
            scopes.extend(("condition", name, split_mask & (arrays["condition"] == ci)) for ci, name in enumerate(manifest["conditions"]))
            scopes.extend(("trajectory", f"T{tid}", split_mask & (arrays["trajectory"] == tid)) for tid in sorted(set(arrays["trajectory"][split_mask])))
            for scope, group, mask in scopes:
                if not np.any(mask):
                    continue
                base = metrics(target[mask], m1[mask])["rmse_nm"]
                for name in names:
                    score = metrics(target[mask], predictions[name][mask], base)
                    records.append({"scenario": scenario, "split": split, "scope": scope, "group": group,
                                    "model": name, **score, **metadata[name]})
    fields = list(dict.fromkeys(k for row in records for k in row if k != "per_joint_rmse_nm")) + [f"joint_{i}_rmse_nm" for i in range(1, 7)]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({**{k: v for k, v in row.items() if k != "per_joint_rmse_nm"},
                             **{f"joint_{i}_rmse_nm": value for i, value in enumerate(row["per_joint_rmse_nm"], 1)}})
    summary = {"label": "MuJoCo qfrc_inverse and modified-scene qfrc_inverse; no measured motor torque",
               "models": {"M0": "nominal RBD, tool/payload inertial scaled to 80%",
                          "M1": "exact saved-scene RBD (simulator consistency reference)",
                          "M2": "M1 + residual MLP on q,dq",
                          "M3": "M1 + six independent velocity splines",
                          "M4": "M1 + coupled spline KAN [12,8,6]",
                          "M5": "M1 + frozen M3 + coupled KAN trained on remaining residual"},
               "config": config, "records": records}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    from .plots import make_figures
    make_figures(output, manifest, records, first)
    return summary
