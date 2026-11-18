"""Fit friction residual models and evaluate unseen Wide16 trajectories.

The reference inverse-dynamics torque is known for each sample. Every model
predicts only the synthetic friction delta; torque error equals delta error.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .collect_friction import DEFAULT_OUTPUT as DEFAULT_DATA, sha256
from .combined import CONDITIONS, ROOT
from .networks import KANLayer, ResidualMLP, SplineBasis, parameter_count


CONFIG_PATH = ROOT / "src/model/friction_validate_config.json"
DEFAULT_OUTPUT = ROOT / "result/friction_model_validation_full_v1"
SCENARIOS = 15
JOINTS = 6


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scenario_features(manifest: dict) -> np.ndarray:
    models = ["coulomb", "viscous", "stribeck", "asymmetric", "lugre"]
    rows = []
    for scenario in manifest["scenario_order"]:
        onehot = [float(scenario["model"] == name) for name in models]
        rows.append(onehot + [float(scenario["scale"])])
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (SCENARIOS, 6):
        raise ValueError("expected 15 scenarios across five friction laws")
    return result


def _read_data(data_root: Path, config: dict, smoke: bool) -> tuple[dict, dict, dict]:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not manifest.get("training_ready") or manifest.get("partial_collection")
            or manifest.get("trial_chunks") != 336 or manifest.get("state_rows") != 1317456
            or manifest.get("exact_feature_split_overlap") !=
            {"train_validation": 0, "train_test": 0, "validation_test": 0}):
        raise ValueError("full friction dataset validation did not pass")
    _scenario_features(manifest)
    by_split = {name: {"features": [], "delta": [], "condition": []}
                for name in ("train", "validation", "test")}
    rng = np.random.default_rng(config["selection_seed"])
    seen = set()
    for record in manifest["trials"]:
        chunk = data_root / record["relative_npz"]
        if sha256(chunk) != record["npz_sha256"]:
            raise ValueError(f"chunk hash mismatch: {chunk}")
        with np.load(chunk, allow_pickle=False) as arrays:
            x = arrays["features"]
            y = arrays["delta_friction_nm"]
        if (x.shape != (record["rows"], 22) or y.shape != (record["rows"], SCENARIOS, JOINTS)
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError(f"invalid chunk arrays: {chunk}")
        group = (record["condition"], record["trajectory"], record["trial"])
        if group in seen:
            raise ValueError(f"repeated chunk identity: {group}")
        seen.add(group)
        split = record["split"]
        if split not in by_split or record["trajectory"] not in manifest["config"]["split_policy"][split]:
            raise ValueError(f"split mismatch: {chunk}")
        limit = (config["train_rows_per_chunk"] if split == "train" else
                 config["validation_rows_per_chunk"] if split == "validation" else
                 16 if smoke else len(x))
        chosen = (np.arange(len(x)) if limit >= len(x) else
                  np.sort(rng.choice(len(x), size=limit, replace=False)))
        target = by_split[split]
        target["features"].append(x[chosen])
        target["delta"].append(y[chosen])
        target["condition"].append(np.full(len(chosen), CONDITIONS.index(record["condition"]), dtype=np.int8))
    if len(seen) != 336:
        raise ValueError("missing trial chunks")
    data = {split: {field: np.concatenate(parts) for field, parts in fields.items()}
            for split, fields in by_split.items()}
    provenance = {"dataset_manifest_sha256": sha256(manifest_path),
                  "dataset_config_sha256": manifest["config_sha256"],
                  "training_state_rows_used": len(data["train"]["features"]),
                  "validation_state_rows_used": len(data["validation"]["features"]),
                  "test_state_rows_used": len(data["test"]["features"]),
                  "full_training_state_rows_available": 988092,
                  "full_validation_state_rows_available": 82341,
                  "full_test_state_rows_available": 247023,
                  "smoke": smoke}
    return manifest, data, provenance


def _linear_design(velocity: np.ndarray) -> np.ndarray:
    return np.stack((np.ones_like(velocity), np.tanh(velocity / 0.005), velocity), axis=1)


def _spline_design(velocity: np.ndarray, mean: float, scale: float,
                   basis: SplineBasis) -> np.ndarray:
    with torch.inference_mode():
        spline = basis(torch.from_numpy(((velocity - mean) / scale).astype(np.float32)[:, None]))
    return np.concatenate((np.ones((len(velocity), 1)), spline[:, 0, :].numpy()), axis=1)


def _fit_closed_form(data: dict, config: dict) -> tuple[dict, dict, dict[str, float]]:
    x, y = data["train"]["features"], data["train"]["delta"].astype(np.float64)
    lam = float(config["ridge_lambda"])
    basis = SplineBasis(config["spline_intervals"])
    vel_mean = x[:, 6:12].mean(axis=0).astype(np.float64)
    vel_scale = np.maximum(x[:, 6:12].std(axis=0).astype(np.float64), 0.01)
    linear = np.empty((JOINTS, 3, SCENARIOS), dtype=np.float64)
    spline = np.empty((JOINTS, basis.n_basis + 1, SCENARIOS), dtype=np.float64)
    timings = {"Linear_friction": 0.0, "Jointwise_spline": 0.0}
    for joint in range(JOINTS):
        velocity = x[:, 6 + joint].astype(np.float64)
        for name, output in (("Linear_friction", linear), ("Jointwise_spline", spline)):
            start = time.perf_counter()
            design = (_linear_design(velocity) if name == "Linear_friction" else
                      _spline_design(velocity, vel_mean[joint], vel_scale[joint], basis))
            gram = design.T @ design
            ridge = np.eye(design.shape[1]) * lam
            ridge[0, 0] = 0.0
            output[joint] = np.linalg.solve(gram + ridge, design.T @ y[:, :, joint])
            timings[name] += time.perf_counter() - start
    return ({"coefficient": linear},
            {"coefficient": spline, "velocity_mean": vel_mean,
             "velocity_scale": vel_scale, "intervals": config["spline_intervals"]}, timings)


def _predict_closed(kind: str, fit: dict, x: np.ndarray) -> np.ndarray:
    if kind == "RBD_zero_friction":
        return np.zeros((len(x), SCENARIOS, JOINTS), dtype=np.float32)
    result = np.empty((len(x), SCENARIOS, JOINTS), dtype=np.float32)
    basis = SplineBasis(fit["intervals"]) if kind == "Jointwise_spline" else None
    for joint in range(JOINTS):
        velocity = x[:, 6 + joint].astype(np.float64)
        design = (_spline_design(velocity, fit["velocity_mean"][joint],
                                 fit["velocity_scale"][joint], basis)
                  if basis is not None else _linear_design(velocity))
        result[:, :, joint] = design @ fit["coefficient"][joint]
    return result


class SharedResidual(nn.Module):
    def __init__(self, kind: str, x_mean: np.ndarray, x_scale: np.ndarray,
                 y_mean: np.ndarray, y_scale: np.ndarray, intervals: int):
        super().__init__()
        self.kind = kind
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_scale", torch.tensor(y_scale, dtype=torch.float32))
        if kind == "MLP":
            self.network = ResidualMLP(28, width=32)
        elif kind == "KAN":
            self.network = nn.Sequential(KANLayer(28, 8, intervals), KANLayer(8, 6, intervals))
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network((x - self.x_mean) / self.x_scale) * self.y_scale + self.y_mean


def _model_stats(train: dict, scenario_features: np.ndarray) -> tuple[np.ndarray, ...]:
    state = train["features"]
    x_mean = np.r_[state.mean(0), scenario_features.mean(0)]
    x_scale = np.maximum(np.r_[state.std(0), scenario_features.std(0)], 0.01)
    y = train["delta"]
    y_mean = y.mean(axis=(0, 1))
    y_scale = np.maximum(y.std(axis=(0, 1)), 0.02)
    return x_mean, x_scale, y_mean, y_scale


def _predict_neural(model: SharedResidual, x: np.ndarray, scenario_features: np.ndarray,
                    batch_size: int) -> np.ndarray:
    result = np.empty((len(x), SCENARIOS, JOINTS), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), max(1, batch_size // SCENARIOS)):
            stop = min(len(x), start + max(1, batch_size // SCENARIOS))
            repeated = np.repeat(x[start:stop], SCENARIOS, axis=0)
            scenarios = np.tile(scenario_features, (stop - start, 1))
            inputs = torch.from_numpy(np.concatenate((repeated, scenarios), axis=1))
            result[start:stop] = model(inputs).numpy().reshape(stop - start, SCENARIOS, JOINTS)
    return result


def _fit_neural(kind: str, seed: int, train: dict, validation: dict,
                scenario_features: np.ndarray, config: dict) -> tuple[SharedResidual, list[dict], int, float]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = SharedResidual(kind, *_model_stats(train, scenario_features), config["spline_intervals"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    states = torch.from_numpy(train["features"])
    targets = torch.from_numpy(train["delta"])
    scenario_tensor = torch.from_numpy(scenario_features)
    best_rmse, best_epoch, best_state = float("inf"), 0, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        indices = np_rng.permutation(len(states))
        loss_sum = 0.0
        for subset in np.array_split(indices, max(1, int(np.ceil(len(indices) / config["batch_size"])))):
            ids = torch.from_numpy(subset)
            scenario_ids = (ids + epoch - 1) % SCENARIOS
            batch = torch.cat((states[ids], scenario_tensor[scenario_ids]), dim=1)
            target = targets[ids, scenario_ids]
            optimizer.zero_grad(set_to_none=True)
            error = (model(batch) - target) / model.y_scale
            loss = error.square().mean()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(subset)
        predicted = _predict_neural(model, validation["features"], scenario_features,
                                    config["inference_batch_size"])
        rmse = float(np.sqrt(np.mean((predicted.astype(np.float64) - validation["delta"]) ** 2)))
        elapsed = time.perf_counter() - start
        history.append({"epoch": epoch, "train_normalized_mse": loss_sum / len(states),
                        "validation_rmse_nm": rmse, "elapsed_s": elapsed})
        print(f"{kind} seed={seed} epoch={epoch} val_rmse={rmse:.6f} Nm ({elapsed:.1f}s)", flush=True)
        if not np.isfinite(rmse):
            raise ValueError("nonfinite validation RMSE")
        if rmse < best_rmse - 1e-7:
            best_rmse, best_epoch, best_state = rmse, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch, time.perf_counter() - start


def _scores(kind: str, seed: str, split: str, data: dict, prediction: np.ndarray,
            scenario_order: list[dict], parameters: int, training_s: float,
            best_epoch: int) -> tuple[dict, list[dict], list[dict], list[dict]]:
    truth = data["delta"].astype(np.float64)
    error = prediction.astype(np.float64) - truth
    sq = error ** 2
    base = {"model": kind, "seed": seed, "split": split}
    overall = {**base, "state_rows": len(truth), "scenario_labels": len(truth) * SCENARIOS,
               "torque_rmse_nm": float(np.sqrt(sq.mean())),
               "parameters": parameters, "training_s": training_s, "best_epoch": best_epoch}
    scenarios = [{**base, "scenario": scenario_order[i]["name"],
                  "model_family": scenario_order[i]["model"], "level": scenario_order[i]["level"],
                  "state_rows": len(truth), "torque_rmse_nm": float(np.sqrt(sq[:, i].mean()))}
                 for i in range(SCENARIOS)]
    conditions = [{**base, "condition": condition, "state_rows": int(np.sum(data["condition"] == ci)),
                   "torque_rmse_nm": float(np.sqrt(sq[data["condition"] == ci].mean()))}
                  for ci, condition in enumerate(CONDITIONS)]
    joints = [{**base, "joint": i + 1, "torque_rmse_nm": float(np.sqrt(sq[:, :, i].mean()))}
              for i in range(JOINTS)]
    return overall, scenarios, conditions, joints


def run(data_root: Path, output: Path, config: dict, smoke: bool = False) -> dict:
    torch.set_num_threads(config["torch_threads"])
    manifest, data, provenance = _read_data(data_root, config, smoke)
    if output.exists():
        raise FileExistsError(f"use a new result folder: {output}")
    output.mkdir(parents=True)
    selected_validation = {key: value for key, value in data["validation"].items()}
    scenario_features = _scenario_features(manifest)
    if smoke:
        config = {**config, "seeds": config["seeds"][:1], "epochs": 2,
                  "train_rows_per_chunk": 16, "validation_rows_per_chunk": 8}
    all_overall, all_scenarios, all_conditions, all_joints, histories = [], [], [], [], []
    linear, spline, fit_times = _fit_closed_form(data, config)
    for kind, fit in (("RBD_zero_friction", {}), ("Linear_friction", linear),
                      ("Jointwise_spline", spline)):
        for split in ("validation", "test"):
            prediction = _predict_closed(kind, fit, data[split]["features"])
            score = _scores(kind, "closed_form", split, data[split], prediction,
                            manifest["scenario_order"], 0 if not fit else fit["coefficient"].size,
                            fit_times.get(kind, 0.0), 0)
            for target, values in zip((all_overall, all_scenarios, all_conditions, all_joints), score):
                target.extend(values if isinstance(values, list) else [values])
        if fit:
            np.savez_compressed(output / f"{kind}_fit.npz", **fit)
        print(f"{kind} test RMSE {all_overall[-1]['torque_rmse_nm']:.6f} Nm", flush=True)
    for kind in ("MLP", "KAN"):
        for seed in config["seeds"]:
            model, history, best_epoch, training_s = _fit_neural(
                kind, seed, data["train"], selected_validation, scenario_features, config)
            torch.save({"kind": kind, "seed": seed, "state_dict": model.state_dict(),
                        "config": config, "best_epoch": best_epoch,
                        "dataset_manifest_sha256": provenance["dataset_manifest_sha256"]},
                       output / f"{kind}_seed_{seed}.pt")
            histories.extend({"model": kind, "seed": seed, **row} for row in history)
            for split in ("validation", "test"):
                prediction = _predict_neural(model, data[split]["features"], scenario_features,
                                             config["inference_batch_size"])
                score = _scores(kind, str(seed), split, data[split], prediction,
                                manifest["scenario_order"], parameter_count(model), training_s, best_epoch)
                for target, values in zip((all_overall, all_scenarios, all_conditions, all_joints), score):
                    target.extend(values if isinstance(values, list) else [values])
            print(f"{kind} seed={seed} full test RMSE {all_overall[-1]['torque_rmse_nm']:.6f} Nm", flush=True)
    for filename, rows in (("overall.csv", all_overall), ("by_scenario.csv", all_scenarios),
                           ("by_condition.csv", all_conditions), ("by_joint.csv", all_joints),
                           ("history.csv", histories)):
        _csv(output / filename, rows)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "experiment": config["experiment"], "dataset_kind": manifest["dataset_kind"],
              "target": "synthetic friction delta; total torque RMSE is identical because reference RBD is added exactly",
              "protocol": "Train T1-T11/T15; validation T12; full test T13/T14/T16; 15 scenarios; all 7 conditions",
              "memoryless_models": True,
              "lugre_hidden_state_not_in_28_inputs": True,
              "closed_form_fit_times_s": fit_times,
              "config": config, "provenance": provenance,
              "test_summary": [{"model": kind, "seeds": [r["seed"] for r in all_overall
                                                  if r["model"] == kind and r["split"] == "test"],
                                "mean_rmse_nm": float(np.mean([r["torque_rmse_nm"] for r in all_overall
                                                               if r["model"] == kind and r["split"] == "test"])),
                                "sd_rmse_nm": float(np.std([r["torque_rmse_nm"] for r in all_overall
                                                              if r["model"] == kind and r["split"] == "test"], ddof=1))
                                if kind in ("MLP", "KAN") and len(config["seeds"]) > 1 else None}
                               for kind in ("RBD_zero_friction", "Linear_friction",
                                            "Jointwise_spline", "MLP", "KAN")]}
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = {**config, "train_rows_per_chunk": 16, "validation_rows_per_chunk": 8,
                  "epochs": 2, "seeds": config["seeds"][:1]}
    result = run(args.data_root.resolve(), args.output.resolve(), config, args.smoke)
    print(json.dumps({"test_summary": result["test_summary"], "provenance": result["provenance"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
