"""Diagnose fitted synthetic-friction models using saved checkpoints.

The audit evaluates held-in training states and pre-existing validation/test
scores. New tuning experiments in this module use validation only.
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

from .collect_friction import DEFAULT_OUTPUT as DEFAULT_DATA, _asymmetric_friction, _smooth_friction
from .friction_validate import (_predict_closed, _predict_neural, _read_data,
                                _scenario_features, _scores, SharedResidual)
from .networks import SplineBasis
from .combined import ROOT


DEFAULT_RESULT = ROOT / "result/friction_model_validation_full_v1"
DEFAULT_OUTPUT = ROOT / "result/friction_model_diagnostics_v2"
FAMILIES = ("coulomb", "viscous", "stribeck", "asymmetric", "lugre")
MODELS = ("RBD_zero_friction", "Linear_friction", "Jointwise_spline", "MLP", "KAN")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _load_neural(result: Path, kind: str, seed: int, intervals: int) -> SharedResidual:
    saved = torch.load(result / f"{kind}_seed_{seed}.pt", map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    model = SharedResidual(kind, state["x_mean"].numpy(), state["x_scale"].numpy(),
                           state["y_mean"].numpy(), state["y_scale"].numpy(), intervals)
    model.load_state_dict(state)
    model.eval()
    return model


def _pooled_family(scenario_rows: list[dict], split: str) -> list[dict]:
    result = []
    for model in MODELS:
        seeds = sorted({row["seed"] for row in scenario_rows if row["model"] == model and row["split"] == split})
        for seed in seeds:
            for family in (*FAMILIES, "excluding_lugre"):
                rows = [row for row in scenario_rows if row["model"] == model and
                        row["seed"] == seed and row["split"] == split and
                        (row["model_family"] == family if family != "excluding_lugre"
                         else row["model_family"] != "lugre")]
                if len(rows) != (12 if family == "excluding_lugre" else 3):
                    raise ValueError(f"scenario coverage mismatch: {model}/{seed}/{family}")
                values = np.asarray([float(row["torque_rmse_nm"]) for row in rows])
                result.append({"model": model, "seed": seed, "split": split,
                               "family": family, "rmse_nm": float(np.sqrt(np.mean(values ** 2)))})
    return result


def _plot_edge(model: SharedResidual, velocity: np.ndarray, output: Path) -> list[dict]:
    mean = float(model.x_mean[6])
    scale = float(model.x_scale[6])
    grid = np.linspace(float(velocity.min()), float(velocity.max()), 241, dtype=np.float32)
    basis = SplineBasis(model.network[0].basis.intervals)
    with torch.inference_mode():
        values = basis(torch.from_numpy(((grid - mean) / scale)[:, None])).numpy()[:, 0, :]
    coefficients = model.network[0].coefficients.detach().numpy()[:, 6, :]
    curves = values @ coefficients.T
    rows = [{"dq1_rad_s": float(v), **{f"hidden_{i+1}_edge_value": float(y[i]) for i in range(8)}}
            for v, y in zip(grid, curves)]
    _write_csv(output / "kan_j1_first_layer_edges.csv", rows)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for i in range(8):
        axes[0].plot(grid, curves[:, i], label=f"hidden {i+1}", linewidth=1.2)
    axes[0].set(ylabel="First-layer edge contribution", title="KAN seed 11 · J1 velocity edges")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].hist(velocity, bins=70, color="#75849b")
    axes[1].set(xlabel="J1 velocity (rad/s)", ylabel="Train count")
    fig.tight_layout()
    fig.savefig(output / "kan_j1_first_layer_edges.png", dpi=220)
    fig.savefig(output / "kan_j1_first_layer_edges.svg")
    plt.close(fig)
    return rows


def _plot_partial(models: dict, data: dict, scenarios: np.ndarray,
                  generator_config: dict, output: Path) -> None:
    train = data["train"]["features"]
    contexts = train[np.flatnonzero(train[:, 18] == 1)[:20]].copy()
    if len(contexts) != 20:
        raise ValueError("not enough bare contexts")
    velocities = np.linspace(float(np.percentile(train[:, 6], 0.5)),
                             float(np.percentile(train[:, 6], 99.5)), 201)
    fc = np.asarray(generator_config["friction_base_nm"])
    b = np.asarray(generator_config["viscous_base_nm_s_rad"])
    selected = (("viscous", 4), ("stribeck", 7), ("asymmetric", 10))
    all_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=False)
    for axis, (family, scenario_id) in zip(axes, selected):
        x = np.repeat(contexts[None, :, :], len(velocities), axis=0)
        x[:, :, 6] = velocities[:, None]
        x = x.reshape(-1, 22)
        inputs = np.concatenate((x, np.tile(scenarios[scenario_id], (len(x), 1))), axis=1)
        curves = {}
        for kind in ("MLP", "KAN"):
            predictions = []
            for model in models[kind]:
                with torch.inference_mode():
                    values = model(torch.from_numpy(inputs.astype(np.float32))).numpy()[:, 0]
                predictions.append(values.reshape(len(velocities), len(contexts)).mean(axis=1))
            curves[kind] = np.asarray(predictions)
        if family == "viscous":
            truth = b[0] * velocities
        elif family == "stribeck":
            truth = _smooth_friction(velocities[:, None], fc[:1], b[:1], generator_config)[:, 0]
        else:
            truth = _asymmetric_friction(velocities[:, None], fc[:1], b[:1], generator_config)[:, 0]
        axis.plot(velocities, truth, color="black", label="generator", linewidth=2)
        for kind, color in (("MLP", "#d29759"), ("KAN", "#816bb0")):
            mean = curves[kind].mean(axis=0)
            sd = curves[kind].std(axis=0, ddof=1)
            axis.plot(velocities, mean, color=color, label=kind)
            axis.fill_between(velocities, mean - sd, mean + sd, color=color, alpha=0.12)
        axis.set(title=family.title(), xlabel="J1 velocity (rad/s)")
        axis.grid(alpha=0.25)
        for index, velocity in enumerate(velocities):
            all_rows.append({"family": family, "dq1_rad_s": float(velocity),
                             "generator_nm": float(truth[index]),
                             "mlp_mean_nm": float(curves["MLP"][:, index].mean()),
                             "kan_mean_nm": float(curves["KAN"][:, index].mean())})
    axes[0].set_ylabel("J1 synthetic friction delta (Nm)")
    axes[-1].legend()
    fig.suptitle("Partial dependence at 20 training contexts · nominal severity")
    fig.tight_layout()
    fig.savefig(output / "j1_partial_dependence.png", dpi=220)
    fig.savefig(output / "j1_partial_dependence.svg")
    plt.close(fig)
    _write_csv(output / "j1_partial_dependence.csv", all_rows)


def _zero_speed_scores(models: dict, validation: dict,
                       scenarios: np.ndarray, output: Path) -> list[dict]:
    x = validation["features"]
    truth = validation["delta"].astype(np.float64)
    velocity = np.abs(x[:, 6:12])
    bins = (("near_zero", velocity < 0.005),
            ("transition", (velocity >= 0.005) & (velocity < 0.02)),
            ("faster", velocity >= 0.02))
    rows = []
    for kind, fitted in models.items():
        for seed, model in zip((11, 29, 47), fitted):
            predicted = _predict_neural(model, x, scenarios, 8192).astype(np.float64)
            error = (predicted - truth) ** 2
            for family_index, family in enumerate(FAMILIES):
                family_error = error[:, family_index * 3:(family_index + 1) * 3, :]
                for label, mask in bins:
                    selected = np.broadcast_to(mask[:, None, :], family_error.shape)
                    rows.append({"model": kind, "seed": seed, "family": family,
                                 "speed_bin": label, "velocity_range_rad_s":
                                 "[0,0.005)" if label == "near_zero" else
                                 "[0.005,0.02)" if label == "transition" else "[0.02,inf)",
                                 "joint_scenario_elements": int(selected.sum()),
                                 "rmse_nm": float(np.sqrt(family_error[selected].mean()))})
    _write_csv(output / "validation_by_speed_bin.csv", rows)
    return rows


def audit(data_root: Path, result: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"use a new diagnostic folder: {output}")
    config = json.loads((ROOT / "src/model/friction_validate_config.json").read_text(encoding="utf-8"))
    probe_config = {**config, "train_rows_per_chunk": 64, "validation_rows_per_chunk": 128}
    manifest, data, provenance = _read_data(data_root, probe_config, smoke=True)
    torch.set_num_threads(config["torch_threads"])
    output.mkdir(parents=True)
    scenarios = _scenario_features(manifest)
    original = _read_csv(result / "overall.csv")
    original_scenarios = _read_csv(result / "by_scenario.csv")
    train_overall, train_scenarios, train_joints = [], [], []
    neural_models = {"MLP": [], "KAN": []}
    for kind in MODELS:
        seeds = config["seeds"] if kind in neural_models else ["closed_form"]
        for seed in seeds:
            if kind == "RBD_zero_friction":
                prediction = np.zeros_like(data["train"]["delta"])
            elif kind in ("Linear_friction", "Jointwise_spline"):
                with np.load(result / f"{kind}_fit.npz", allow_pickle=False) as saved:
                    fit = {key: saved[key] for key in saved.files}
                if "intervals" in fit:
                    fit["intervals"] = int(fit["intervals"])
                prediction = _predict_closed(kind, fit, data["train"]["features"])
            else:
                model = _load_neural(result, kind, seed, config["spline_intervals"])
                neural_models[kind].append(model)
                prediction = _predict_neural(model, data["train"]["features"], scenarios,
                                             config["inference_batch_size"])
            score = _scores(kind, str(seed), "train_sample", data["train"], prediction,
                            manifest["scenario_order"], 0, 0.0, 0)
            train_overall.append(score[0])
            train_scenarios.extend(score[1])
            train_joints.extend(score[3])
            print(f"Train sample {kind}/{seed}: {score[0]['torque_rmse_nm']:.6f} Nm", flush=True)
    _write_csv(output / "train_sample_overall.csv", train_overall)
    _write_csv(output / "train_sample_by_scenario.csv", train_scenarios)
    _write_csv(output / "train_sample_by_joint.csv", train_joints)
    comparison = [{"model": row["model"], "seed": row["seed"], "split": row["split"],
                   "rmse_nm": row["torque_rmse_nm"], "state_rows": row["state_rows"]}
                  for row in original]
    comparison += [{"model": row["model"], "seed": row["seed"], "split": "train_sample",
                    "rmse_nm": row["torque_rmse_nm"], "state_rows": row["state_rows"]}
                   for row in train_overall]
    _write_csv(output / "train_validation_test.csv", comparison)
    family = _pooled_family(train_scenarios, "train_sample") + _pooled_family(original_scenarios, "test")
    _write_csv(output / "family_rmse.csv", family)
    model = neural_models["KAN"][0]
    x = data["train"]["features"]
    norm = (x - model.x_mean[:22].numpy()) / model.x_scale[:22].numpy()
    grid = [{"input": f"dq{joint+1}", "normalized_p01": float(np.percentile(norm[:, 6+joint], 1)),
             "normalized_p99": float(np.percentile(norm[:, 6+joint], 99)),
             "fraction_abs_z_gt_2": float(np.mean(np.abs(norm[:, 6+joint]) > 2)),
             "fraction_abs_z_gt_3": float(np.mean(np.abs(norm[:, 6+joint]) > 3))}
            for joint in range(6)]
    _write_csv(output / "kan_velocity_grid_coverage.csv", grid)
    _plot_edge(model, x[:, 6], output)
    _plot_partial(neural_models, data, scenarios, manifest["config"], output)
    speed_rows = _zero_speed_scores(neural_models, data["validation"], scenarios, output)
    summary = {"dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
               "source_result": str(result), "train_sample_rows": len(x),
               "train_sample_selection": "64 states per condition/trajectory/trial, fixed seed 2026",
               "grid_coverage": grid,
               "validation_speed_bin_rows": len(speed_rows),
               "note": "Test scores copied from prior completed benchmark; no new test-based tuning."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit(args.data_root.resolve(), args.result.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
