"""Study inertial mismatch and held-out payload generalization on A0509 simulations."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn

from .combined import CONDITIONS, ROOT, SOURCE_REPO, inspect
from .data import load_prepared
from .networks import JointwiseSpline, KANLayer, ResidualMLP, parameter_count
from .train import NormalizedBranch, _campaign_weights, _fit


DEFAULT_PREPARED = ROOT / "result/combined_legacy_wide_v3/prepared"
DEFAULT_OUTPUT = ROOT / "result/condition_adaptive_rbd_kan_v1"
CONFIG_PATH = ROOT / "src/model/condition_adaptive_config.json"
CAMPAIGN_ROOTS = ("grid_16_trajectories", "grid_16_wide_v3")
PROTOCOLS = {
    "pooled": (),
    "holdout_p05": ("ag145_p05", "mdpg_p05"),
    "holdout_p10": ("ag145_p10", "mdpg_p10"),
}


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _metrics(error: np.ndarray) -> dict:
    values = np.abs(error.astype(np.float64))
    return {
        "samples": int(error.shape[0]),
        "rmse_nm": float(np.sqrt(np.mean(values ** 2))),
        "mae_nm": float(np.mean(values)),
        "max_abs_nm": float(values.max()),
        "p95_abs_nm": float(np.percentile(values, 95)),
    }


def _feature_keys(features: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(features)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


def _split_duplicate_audit(arrays: dict) -> list[dict]:
    keys = _feature_keys(arrays["features"])
    records = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        left = np.flatnonzero(arrays["split"] == first)
        right = np.flatnonzero(arrays["split"] == second)
        common = np.intersect1d(np.unique(keys[left]), np.unique(keys[right]))
        label_spread = 0.0
        if len(common):
            matched = np.r_[left[np.isin(keys[left], common)],
                            right[np.isin(keys[right], common)]]
            for key in common:
                values = arrays["tau_reference"][matched[keys[matched] == key]].astype(np.float64)
                label_spread = max(label_spread, float(np.max(values.max(0) - values.min(0))))
        records.append({"first_split": first, "second_split": second,
                        "unique_duplicate_states": len(common),
                        "first_rows_matching": int(np.isin(keys[left], common).sum()),
                        "second_rows_matching": int(np.isin(keys[right], common).sum()),
                        "max_reference_label_spread_nm": label_spread})
    return records


def _clean_validation(train: np.ndarray, validation: np.ndarray,
                      features: np.ndarray) -> tuple[np.ndarray, int]:
    keys = _feature_keys(features)
    duplicate = np.isin(keys[validation], np.unique(keys[train]))
    clean = validation[~duplicate]
    if not len(clean) or np.intersect1d(np.unique(keys[train]), np.unique(keys[clean])).size:
        raise ValueError("training/validation exact feature overlap remains")
    return clean, int(duplicate.sum())


def _load_verified(prepared: Path, source_repo: Path) -> tuple[dict, dict, list[dict], dict]:
    legacy = ROOT / "data/grid16_legacy_joint_dataset_m0_m4"
    wide = ROOT / "data/wide16_v3_joint_dataset"
    loaded, audit = inspect(legacy, wide, source_repo)
    manifest, arrays = load_prepared(prepared)
    if manifest.get("schema_version") != 2 or manifest.get("campaigns") != ["legacy", "wide16_v3"]:
        raise ValueError("unexpected combined dataset")
    if manifest.get("exact_feature_row_overlap") != 0 or audit["exact_feature_row_overlap"] != 0:
        raise ValueError("cross-campaign duplicate feature rows")
    if len(arrays["features"]) != audit["total_rows"] or manifest["rows"] != audit["rows"]:
        raise ValueError("combined row counts differ from source archives")
    expected = np.concatenate([item["arrays"]["features"] for item in loaded])
    if not np.array_equal(arrays["features"], expected):
        raise ValueError("combined features differ from source archives")
    for code, item in enumerate(loaded):
        source = manifest["source_archives"][item["name"]]
        if (source["dataset_sha256"] != item["manifest"]["dataset_sha256"]
                or source["manifest_sha256"] != item["manifest_sha256"]):
            raise ValueError(f"source digest mismatch: {item['name']}")
        mask = arrays["campaign"] == code
        if int(mask.sum()) != len(item["arrays"]["features"]):
            raise ValueError(f"campaign count mismatch: {item['name']}")
        for key in ("tau_reference", "tau_nominal", "condition", "trajectory"):
            if not np.array_equal(arrays[key][mask], item["arrays"][key]):
                raise ValueError(f"combined {key} differs from {item['name']}")
    if any(not np.isfinite(arrays[key]).all()
           for key in ("features", "tau_reference", "tau_nominal", "tau_friction")):
        raise ValueError("nonfinite values")
    if arrays["features"].shape != (296520, 22):
        raise ValueError("wrong feature shape")
    if any(arrays[key].shape != (296520, 6)
           for key in ("tau_reference", "tau_nominal", "tau_friction")):
        raise ValueError("wrong torque shape")
    for ci, name in enumerate(CONDITIONS):
        mask = arrays["condition"] == ci
        onehot = np.zeros(3, dtype=np.float32)
        onehot[0 if ci == 0 else 1 if ci < 4 else 2] = 1
        payload = 0 if ci in (0, 1, 4) else 0.5 if ci in (2, 5) else 1.0
        if not np.all(arrays["features"][mask, 18:21] == onehot):
            raise ValueError(f"condition one-hot mismatch: {name}")
        if not np.all(arrays["features"][mask, 21] == payload):
            raise ValueError(f"payload mismatch: {name}")
    for trajectory in range(1, 17):
        if len(np.unique(arrays["split"][arrays["trajectory"] == trajectory])) != 1:
            raise ValueError(f"trajectory T{trajectory} crosses splits")
    for name, code in manifest["split_codes"].items():
        if int((arrays["split"] == code).sum()) != manifest["rows"][name]:
            raise ValueError(f"split count mismatch: {name}")
    return manifest, arrays, loaded, audit


def _scene_audit(loaded: list[dict], arrays: dict, source_repo: Path) -> dict:
    sys.path.insert(0, str(source_repo.resolve()))
    from model.campaign_rbd import load_scene
    from model.rbd import RBDModel

    scene_rows = []
    max_exact = max_nominal = 0.0
    for ci, condition in enumerate(CONDITIONS):
        paths = [item["scenes"][condition] for item in loaded]
        texts = [p.read_text(encoding="utf-8") for p in paths]
        normalized = [text.replace(CAMPAIGN_ROOTS[0], "<campaign>")
                          .replace(CAMPAIGN_ROOTS[1], "<campaign>") for text in texts]
        # The AG145 XML hashes differ solely because the mesh paths name their campaign.
        if normalized[0] != normalized[1]:
            raise ValueError(f"physical scene content may differ: {condition}")
        mesh_equal = True
        if condition.startswith("ag145"):
            mesh_roots = [source_repo / "data" / directory / "models/conditions/ag145_meshes"
                          for directory in CAMPAIGN_ROOTS]
            mesh_names = sorted(path.name for path in mesh_roots[0].iterdir() if path.is_file())
            if mesh_names != sorted(path.name for path in mesh_roots[1].iterdir() if path.is_file()):
                raise ValueError("AG145 mesh file lists differ")
            for name in mesh_names:
                digests = [hashlib.sha256((root / name).read_bytes()).digest()
                           for root in mesh_roots]
                if digests[0] != digests[1]:
                    mesh_equal = False
                    break
            if not mesh_equal:
                raise ValueError("AG145 mesh contents differ")
        scene_rows.append({
            "condition": condition,
            "xml_hash_equal": loaded[0]["manifest"]["scene_sha256"][condition]
                              == loaded[1]["manifest"]["scene_sha256"][condition],
            "xml_equal_after_mesh_path_normalization": True,
            "mesh_contents_equal": mesh_equal,
        })
        for code, item in enumerate(loaded):
            exact = RBDModel(load_scene(item["scenes"][condition]))
            nominal = RBDModel(load_scene(item["scenes"][condition], 0.8))
            group = np.flatnonzero((arrays["campaign"] == code) & (arrays["condition"] == ci))
            for trajectory in range(1, 17):
                trial = group[arrays["trajectory"][group] == trajectory]
                if len(trial) == 0:
                    raise ValueError(f"missing trial {condition}/T{trajectory}")
                for index in (int(trial[0]), int(trial[-1])):
                    x = arrays["features"][index].astype(np.float64)
                    a = exact.evaluate(x[:6], x[6:12], x[12:18])["torque"]
                    n = nominal.evaluate(x[:6], x[6:12], x[12:18])["torque"]
                    max_exact = max(max_exact, float(np.max(np.abs(a - arrays["tau_reference"][index]))))
                    max_nominal = max(max_nominal, float(np.max(np.abs(n - arrays["tau_nominal"][index]))))
    if max_exact > 2e-5 or max_nominal > 2e-5:
        raise ValueError(f"archived labels mismatch scenes: exact={max_exact}, nominal={max_nominal}")
    return {"scenes": scene_rows, "max_endpoint_reference_error_nm": max_exact,
            "max_endpoint_nominal_error_nm": max_nominal}


def _residual_metrics(arrays: dict) -> tuple[list[dict], list[dict]]:
    residual = arrays["tau_reference"].astype(np.float64) - arrays["tau_nominal"].astype(np.float64)
    records = []
    summary = []
    for campaign_code, campaign_name in ((None, "pooled"), (0, "legacy"), (1, "wide16_v3")):
        for ci, condition in enumerate(CONDITIONS):
            mask = arrays["condition"] == ci
            if campaign_code is not None:
                mask &= arrays["campaign"] == campaign_code
            values = residual[mask]
            summary.append({"campaign": campaign_name, "condition": condition, **_metrics(values)})
            for joint in range(6):
                records.append({"campaign": campaign_name, "condition": condition,
                                "joint": joint + 1, **_metrics(values[:, joint:joint + 1])})
    return records, summary


def _paired_blocks(arrays: dict, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    residual = arrays["tau_reference"].astype(np.float64) - arrays["tau_nominal"].astype(np.float64)
    left_scores, right_scores = [], []
    for campaign in (0, 1):
        for trajectory in range(1, 17):
            group = (arrays["campaign"] == campaign) & (arrays["trajectory"] == trajectory)
            a = residual[group & (arrays["condition"] == left)]
            b = residual[group & (arrays["condition"] == right)]
            if not len(a) or not len(b):
                raise ValueError("unpaired condition block")
            left_scores.append(np.sqrt(np.mean(a ** 2)))
            right_scores.append(np.sqrt(np.mean(b ** 2)))
    return np.asarray(left_scores), np.asarray(right_scores)


def _condition_effects(arrays: dict, replicates: int) -> list[dict]:
    comparisons = [
        ("ag145_p0", "ag145_p05"), ("ag145_p05", "ag145_p10"), ("ag145_p0", "ag145_p10"),
        ("mdpg_p0", "mdpg_p05"), ("mdpg_p05", "mdpg_p10"), ("mdpg_p0", "mdpg_p10"),
        ("mdpg_p0", "ag145_p0"), ("mdpg_p05", "ag145_p05"), ("mdpg_p10", "ag145_p10"),
    ]
    rng = np.random.default_rng(20261119)
    records = []
    for left, right in comparisons:
        a, b = _paired_blocks(arrays, CONDITIONS.index(left), CONDITIONS.index(right))
        difference = b - a
        samples = difference[rng.integers(0, len(difference), (replicates, len(difference)))].mean(axis=1)
        records.append({"left": left, "right": right, "blocks": len(difference),
                        "mean_paired_rmse_difference_nm": float(difference.mean()),
                        "ci95_low_nm": float(np.percentile(samples, 2.5)),
                        "ci95_high_nm": float(np.percentile(samples, 97.5)),
                        "all_blocks_positive": bool(np.all(difference > 0))})
    return records


def _gripper_structure(arrays: dict) -> list[dict]:
    residual = arrays["tau_reference"].astype(np.float64) - arrays["tau_nominal"].astype(np.float64)
    records = []
    for payload, ag, md in ((0.0, 1, 4), (0.5, 2, 5), (1.0, 3, 6)):
        ag_profile = np.sqrt(np.mean(residual[arrays["condition"] == ag] ** 2, axis=0))
        md_profile = np.sqrt(np.mean(residual[arrays["condition"] == md] ** 2, axis=0))
        cosine = float(np.dot(ag_profile, md_profile)
                       / (np.linalg.norm(ag_profile) * np.linalg.norm(md_profile)))
        records.append({"payload_kg": payload, "joint_rmse_profile_cosine": cosine,
                        "AG145_over_mDPG_total_rmse_ratio":
                            float(np.sqrt(np.mean(ag_profile ** 2))
                                  / np.sqrt(np.mean(md_profile ** 2))),
                        **{f"ag145_J{j+1}_rmse_nm": float(ag_profile[j]) for j in range(6)},
                        **{f"mdpg_J{j+1}_rmse_nm": float(md_profile[j]) for j in range(6)}})
    return records


def audit(prepared: Path, source_repo: Path, output: Path, config: dict) -> tuple[dict, dict, list[dict]]:
    manifest, arrays, loaded, source_audit = _load_verified(prepared, source_repo)
    scenes = _scene_audit(loaded, arrays, source_repo)
    output.mkdir(parents=True, exist_ok=False)
    details, summary = _residual_metrics(arrays)
    effects = _condition_effects(arrays, config["bootstrap_replicates"])
    _write_csv(output / "inertia_residual_7x6.csv", details)
    _write_csv(output / "inertia_residual_conditions.csv", summary)
    _write_csv(output / "inertia_condition_effects.csv", effects)
    _write_csv(output / "gripper_joint_profiles.csv", _gripper_structure(arrays))
    duplicates = _split_duplicate_audit(arrays)
    _write_csv(output / "split_duplicate_audit.csv", duplicates)
    if any(row["max_reference_label_spread_nm"] > 2e-5 for row in duplicates):
        raise ValueError("same feature state has inconsistent reference labels")
    if any(row["unique_duplicate_states"] for row in duplicates if row["second_split"] == 2):
        raise ValueError("exact test feature rows overlap training or validation")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "prepared": str(prepared.resolve()), "source_repo": str(source_repo.resolve()),
              "source_audit": source_audit, "scene_audit": scenes,
              "split_duplicate_audit": duplicates,
              "label": "MuJoCo inverse dynamics generalized torque; not measured motor torque",
              "residual": "tau_reference - tau_nominal",
              "config": config}
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print("Audit passed: 296520 rows, seven conditions, zero exact cross-campaign overlaps.", flush=True)
    return manifest, arrays, loaded


def prepare_physics_basis(prepared: Path, source_repo: Path, output: Path) -> None:
    manifest, arrays, loaded, _ = _load_verified(prepared, source_repo)
    sys.path.insert(0, str(source_repo.resolve()))
    from model.campaign_rbd import load_scene
    from model.rbd import RBDModel

    basis = np.empty_like(arrays["tau_nominal"])
    nominal = arrays["tau_nominal"].astype(np.float64)
    for code, item in enumerate(loaded):
        for ci, condition in enumerate(CONDITIONS):
            model = RBDModel(load_scene(item["scenes"][condition], 0.6))
            indices = np.flatnonzero((arrays["campaign"] == code) & (arrays["condition"] == ci))
            for index in indices:
                x = arrays["features"][index].astype(np.float64)
                lower = model.evaluate(x[:6], x[6:12], x[12:18])["torque"]
                basis[index] = (nominal[index] - lower) / 0.2
            print(f"Physics basis: {item['name']}/{condition} {len(indices)} rows", flush=True)
    if not np.isfinite(basis).all():
        raise ValueError("nonfinite physical basis")
    # Full-row label consistency check; no reference labels are used to make the basis.
    identity_error = np.max(np.abs(nominal + 0.2 * basis.astype(np.float64)
                                   - arrays["tau_reference"].astype(np.float64)))
    if identity_error > 5e-5:
        raise ValueError(f"physical scale family does not reproduce labels: {identity_error}")
    np.savez_compressed(output / "physics_basis.npz", inertia_scale_derivative_nm=basis)
    report = {"basis": "(tau_nominal_at_scale_0.8 - tau_nominal_at_scale_0.6) / 0.2",
              "R1": "tau_nominal + gamma * basis; gamma fit using training labels only",
              "full_row_max_scale_family_error_nm": float(identity_error),
              "physical_parameter_count": 1}
    (output / "physics_basis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Physics basis ready; max scale-family error {identity_error:.7g} Nm", flush=True)


class KAN22(nn.Module):
    """Same two spline layers as the existing KAN, with all 22 stored inputs."""

    def __init__(self, intervals: int):
        super().__init__()
        self.first = KANLayer(22, 8, intervals)
        self.second = KANLayer(8, 6, intervals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))

    def smoothness(self) -> torch.Tensor:
        return self.first.smoothness() + self.second.smoothness()


def _make_branch(kind: str, x: torch.Tensor, target: torch.Tensor,
                 train: np.ndarray, weights: torch.Tensor, config: dict) -> NormalizedBranch:
    columns = list(range(6, 12)) if kind == "R2" else list(range(22))
    selected = x[train][:, columns]
    selected_target = target[train]
    weight = weights[train, None]
    denominator = weight.sum()
    x_mean = (selected * weight).sum(0) / denominator
    x_scale = ((selected - x_mean).square() * weight).sum(0).div(denominator).sqrt().clamp_min(0.01)
    y_mean = (selected_target * weight).sum(0) / denominator
    y_scale = ((selected_target - y_mean).square() * weight).sum(0).div(denominator).sqrt().clamp_min(0.02)
    if kind == "R2":
        network = JointwiseSpline(config["grid_intervals"])
    elif kind == "R3":
        network = ResidualMLP(22, 30)
    elif kind == "R4":
        network = KAN22(config["grid_intervals"])
    else:
        raise ValueError(kind)
    return NormalizedBranch(network, columns, x_mean, x_scale, y_mean, y_scale)


def _predict_branch(model: NormalizedBranch, features: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(part) for part in features.split(batch_size)]).numpy()


def _record_metrics(protocol: str, model: str, seed: int | None, predicted: np.ndarray,
                    actual: np.ndarray, test_arrays: dict, heldout: tuple[str, ...],
                    parameters: int, training_s: float) -> list[dict]:
    error = predicted.astype(np.float64) - actual.astype(np.float64)
    conditions = test_arrays["condition"]
    campaigns = test_arrays["campaign"]
    unseen_codes = [CONDITIONS.index(name) for name in heldout]
    seen = ~np.isin(conditions, unseen_codes)
    masks = [("test_all", "all", np.ones(len(error), dtype=bool)),
             ("test_seen", "all", seen)]
    if unseen_codes:
        masks.append(("test_unseen", "all", ~seen))
    masks.extend(("condition", name, conditions == ci) for ci, name in enumerate(CONDITIONS))
    masks.extend(("campaign", name, campaigns == ci)
                 for ci, name in enumerate(("legacy", "wide16_v3")))
    masks.extend(("campaign_condition", f"{campaign}/{condition}",
                  (campaigns == cc) & (conditions == ci))
                 for cc, campaign in enumerate(("legacy", "wide16_v3"))
                 for ci, condition in enumerate(CONDITIONS))
    records = []
    for scope, group, mask in masks:
        if not np.any(mask):
            continue
        records.append({"protocol": protocol, "model": model, "seed": "" if seed is None else seed,
                        "scope": scope, "group": group, "joint": "",
                        "condition_status": ("unseen" if group in heldout else "seen")
                                            if scope == "condition" else "",
                        **_metrics(error[mask]),
                        "parameters": parameters, "training_s": training_s})
    for joint in range(6):
        records.append({"protocol": protocol, "model": model, "seed": "" if seed is None else seed,
                        "scope": "joint", "group": f"J{joint + 1}", "joint": joint + 1,
                        "condition_status": "", **_metrics(error[:, joint:joint + 1]),
                        "parameters": parameters, "training_s": training_s})
    return records


def _fit_physics_scalar(basis: np.ndarray, residual: np.ndarray, train: np.ndarray,
                        weights: torch.Tensor) -> tuple[float, float]:
    start = time.perf_counter()
    b = basis[train].astype(np.float64)
    y = residual[train].astype(np.float64)
    w = weights[train].numpy().astype(np.float64)[:, None]
    numerator = float(np.sum(w * b * y))
    denominator = float(np.sum(w * b * b))
    if denominator <= 0:
        raise ValueError("no trainable inertial sensitivity")
    gamma = float(np.clip(numerator / denominator, 0, 0.4))
    return gamma, time.perf_counter() - start


def train_and_evaluate(prepared: Path, output: Path, config: dict) -> None:
    manifest, arrays = load_prepared(prepared)
    with np.load(output / "physics_basis.npz", allow_pickle=False) as saved:
        basis = saved["inertia_scale_derivative_nm"]
    if basis.shape != arrays["tau_reference"].shape:
        raise ValueError("physics basis shape mismatch")
    x = torch.from_numpy(arrays["features"])
    nominal = arrays["tau_nominal"].astype(np.float64)
    actual = arrays["tau_reference"].astype(np.float64)
    residual = actual - nominal
    target = torch.from_numpy(residual.astype(np.float32))
    codes = manifest["split_codes"]
    test = np.flatnonzero(arrays["split"] == codes["test"])
    test_arrays = {key: value[test] for key, value in arrays.items()
                   if key in ("condition", "campaign")}
    torch.set_num_threads(config["threads"])
    records, selection, histories, physics_parameters = [], [], [], []
    model_root = output / "models"
    model_root.mkdir(exist_ok=True)
    for protocol, heldout in PROTOCOLS.items():
        excluded = np.isin(arrays["condition"], [CONDITIONS.index(name) for name in heldout])
        train = np.flatnonzero((arrays["split"] == codes["train"]) & ~excluded)
        raw_val = np.flatnonzero((arrays["split"] == codes["validation"]) & ~excluded)
        val, duplicate_val_removed = _clean_validation(train, raw_val, arrays["features"])
        unseen_test = test[excluded[test]]
        if np.any(excluded[train]) or np.any(excluded[val]) or not len(train) or not len(val):
            raise ValueError(f"selection leakage or empty split: {protocol}")
        if heldout and not len(unseen_test):
            raise ValueError(f"no held-out test rows: {protocol}")
        for split_name, indices in (("train", train), ("validation", val),
                                    ("test_all", test), ("test_unseen", unseen_test)):
            selection.append({"protocol": protocol, "split": split_name, "rows": len(indices),
                              "heldout_conditions": ";".join(heldout),
                              "excluded_rows_in_selection": int(excluded[indices].sum())
                              if split_name in ("train", "validation") else ""})
        selection.append({"protocol": protocol, "split": "validation_duplicate_rows_removed",
                          "rows": duplicate_val_removed, "heldout_conditions": ";".join(heldout),
                          "excluded_rows_in_selection": ""})
        train_weights = _campaign_weights(arrays, train)
        val_weights = _campaign_weights(arrays, val)
        gamma, r1_time = _fit_physics_scalar(basis, residual, train, train_weights)
        physics_parameters.append({"protocol": protocol, "gamma": gamma,
                                   "inertial_scale_estimate": 0.8 + gamma,
                                   "training_rows": len(train), "training_s": r1_time})
        print(f"{protocol}: train {len(train)}, val {len(val)}, unseen test {len(unseen_test)}, "
              f"R1 scale {0.8 + gamma:.8f}", flush=True)
        records.extend(_record_metrics(protocol, "R0", None, nominal[test], actual[test],
                                       test_arrays, heldout, 0, 0))
        r1 = nominal[test] + gamma * basis[test].astype(np.float64)
        records.extend(_record_metrics(protocol, "R1", None, r1, actual[test],
                                       test_arrays, heldout, 1, r1_time))
        protocol_dir = model_root / protocol
        protocol_dir.mkdir(exist_ok=True)
        for seed in config["seeds"]:
            seed_dir = protocol_dir / f"seed_{seed}"
            seed_dir.mkdir(exist_ok=True)
            for kind in ("R2", "R3", "R4"):
                checkpoint_path = seed_dir / f"{kind}.pt"
                torch.manual_seed(seed)
                model = _make_branch(kind, x, target, train, train_weights, config)
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    if (checkpoint["config"] != config or checkpoint["train_rows"] != len(train)
                            or checkpoint["val_rows"] != len(val)):
                        raise ValueError(f"checkpoint configuration mismatch: {checkpoint_path}")
                    model.load_state_dict(checkpoint["state"])
                else:
                    history, best_epoch, training_s = _fit(
                        model, x, target, train, val, config, seed, train_weights, val_weights)
                    checkpoint = {"kind": kind, "seed": seed, "state": model.state_dict(),
                                  "config": config, "train_rows": len(train), "val_rows": len(val),
                                  "best_epoch": best_epoch, "history": history,
                                  "training_s": training_s, "parameters": parameter_count(model)}
                    torch.save(checkpoint, checkpoint_path)
                for item in checkpoint["history"]:
                    histories.append({"protocol": protocol, "model": kind, "seed": seed,
                                      **{key: value for key, value in item.items()
                                         if key != "validation_joint_rmse_nm"}})
                prediction = nominal[test] + _predict_branch(model, x[test], config["batch_size"])
                records.extend(_record_metrics(protocol, kind, seed, prediction, actual[test],
                                               test_arrays, heldout, checkpoint["parameters"],
                                               checkpoint["training_s"]))
                all_rmse = next(row["rmse_nm"] for row in records[::-1]
                                if row["protocol"] == protocol and row["model"] == kind
                                and row["seed"] == seed and row["scope"] == "test_all")
                print(f"{protocol} seed {seed} {kind}: epoch {checkpoint['best_epoch']}, "
                      f"test {all_rmse:.6f} Nm, {checkpoint['training_s']:.1f}s", flush=True)
    _write_csv(output / "selection_counts.csv", selection)
    _write_csv(output / "physical_parameters.csv", physics_parameters)
    _write_csv(output / "training_history.csv", histories)
    _write_csv(output / "model_metrics.csv", records)
    (output / "protocols.json").write_text(
        json.dumps({"protocols": {
            "pooled": {"heldout": [], "train": "all conditions, T1-T11 and T15",
                       "validation": "all conditions, T12"},
            "holdout_p05": {"heldout": list(PROTOCOLS["holdout_p05"]),
                            "train": "bare and both grippers at 0 and 1.0 kg, T1-T11 and T15",
                            "validation": "same seen conditions, T12"},
            "holdout_p10": {"heldout": list(PROTOCOLS["holdout_p10"]),
                            "train": "bare and both grippers at 0 and 0.5 kg, T1-T11 and T15",
                            "validation": "same seen conditions, T12"}},
                    "test": "T13,T14,T16; unseen conditions used only for final evaluation",
                    "learned_residual_base": "R0 nominal RBD",
                    "R1": "one scalar physical inertial scale fitted on training labels only",
                    "R2": "R0 + six independent velocity splines",
                    "R3": "R0 + 22->30->30->6 MLP",
                    "R4": "R0 + 22->8->6 edge-spline KAN",
                    "R3_R4_equal_parameters": 1806,
                    "config": config}, indent=2) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _aggregate_metrics(rows: list[dict], *, protocol: str, scope: str,
                       group: str | None = None) -> list[dict]:
    selected = [row for row in rows if row["protocol"] == protocol and row["scope"] == scope
                and (group is None or row["group"] == group)]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in selected:
        groups.setdefault((row["model"], row["group"]), []).append(row)
    result = []
    for (model, name), items in groups.items():
        values = np.asarray([float(item["rmse_nm"]) for item in items])
        times = np.asarray([float(item["training_s"]) for item in items])
        result.append({"protocol": protocol, "model": model, "scope": scope,
                       "group": name, "seed_count": len(items),
                       "rmse_mean_nm": float(values.mean()),
                       "rmse_sd_nm": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                       "parameters": int(items[0]["parameters"]),
                       "training_mean_s": float(times.mean())})
    return sorted(result, key=lambda row: (row["model"], row["group"]))


def make_figures(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(output / "model_metrics.csv")
    residual = _read_csv(output / "inertia_residual_conditions.csv")
    figure_root = output / "figures"
    figure_root.mkdir(exist_ok=True)
    colors = dict(zip(("R0", "R1", "R2", "R3", "R4"),
                      ("#7f7f7f", "#1b9e77", "#d95f02", "#7570b3", "#e7298a")))

    def save(fig, name: str, data: list[dict]) -> None:
        fig.tight_layout()
        fig.savefig(figure_root / f"{name}.png", dpi=180)
        fig.savefig(figure_root / f"{name}.svg")
        plt.close(fig)
        _write_csv(figure_root / f"{name}.csv", data)

    # Figure 1: original nominal RBD mismatch, with no learned corrections.
    data = [{"condition": row["condition"], "rmse_nm": float(row["rmse_nm"])}
            for row in residual if row["campaign"] == "pooled"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([row["condition"] for row in data], [row["rmse_nm"] for row in data],
           color="#4c78a8")
    ax.set(title="Figure 1. Nominal RBD inertia residual by condition",
           ylabel="RMSE (Nm)", xlabel="Condition")
    ax.tick_params(axis="x", rotation=35)
    save(fig, "figure1_condition_rbd_residual", data)

    # Figure 2: pooled T13/T14/T16 torque RMSE.
    data = _aggregate_metrics(rows, protocol="pooled", scope="test_all", group="all")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for index, row in enumerate(data):
        ax.errorbar(index, row["rmse_mean_nm"], yerr=row["rmse_sd_nm"],
                    marker="o", markersize=9, capsize=4, linestyle="none",
                    color=colors[row["model"]])
        ax.annotate(f"{row['rmse_mean_nm']:.3g}", (index, row["rmse_mean_nm"]),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax.set(title="Figure 2. Model comparison on pooled test trajectories",
           ylabel="Torque RMSE (Nm, log scale)", xlabel="Model",
           xticks=range(len(data)), xticklabels=[row["model"] for row in data],
           yscale="log", ylim=(5e-7, 2.0))
    save(fig, "figure2_test_rmse", data)

    # Figure 3: six joint errors on pooled test trajectories.
    data = _aggregate_metrics(rows, protocol="pooled", scope="joint")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in colors:
        subset = sorted((row for row in data if row["model"] == model),
                        key=lambda row: int(row["group"][1:]))
        ax.plot(range(1, 7), [row["rmse_mean_nm"] for row in subset],
                marker="o", label=model, color=colors[model])
    ax.set(title="Figure 3. Joint-wise test RMSE", xlabel="Joint",
           ylabel="Torque RMSE (Nm, log scale)", xticks=range(1, 7),
           yscale="log", ylim=(1e-7, 3.0))
    ax.legend()
    save(fig, "figure3_joint_rmse", data)

    # Figure 4: tool and payload curves for every model.
    all_conditions = _aggregate_metrics(rows, protocol="pooled", scope="condition")
    data = []
    for row in all_conditions:
        condition = row["group"]
        if condition.startswith("ag145"):
            tool = "AG145"
        elif condition.startswith("mdpg"):
            tool = "mDPG"
        else:
            continue
        payload = 0.0 if condition.endswith("_p0") else 0.5 if condition.endswith("_p05") else 1.0
        data.append({**row, "tool": tool, "payload_kg": payload})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, tool in zip(axes, ("AG145", "mDPG")):
        for model in colors:
            subset = sorted((row for row in data if row["tool"] == tool and row["model"] == model),
                            key=lambda row: row["payload_kg"])
            ax.plot([row["payload_kg"] for row in subset],
                    [max(row["rmse_mean_nm"], 1e-6) for row in subset],
                    marker="o", label=model, color=colors[model])
        ax.set(title=tool, xlabel="Payload (kg)", xticks=(0, 0.5, 1.0), yscale="log")
    axes[0].set_ylabel("Torque RMSE (Nm, log scale)")
    axes[1].legend()
    fig.suptitle("Figure 4. Tool and payload test errors")
    save(fig, "figure4_payload_curves", data)

    # Figure 5: both tools' entirely unseen payload conditions.
    data = []
    for protocol in ("holdout_p05", "holdout_p10"):
        heldout = PROTOCOLS[protocol]
        for row in _aggregate_metrics(rows, protocol=protocol, scope="condition"):
            if row["group"] in heldout:
                data.append(row)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, protocol in zip(axes, ("holdout_p05", "holdout_p10")):
        payload = "0.5 kg" if protocol == "holdout_p05" else "1.0 kg"
        for tool, prefix in (("AG145", "ag145"), ("mDPG", "mdpg")):
            name = f"{prefix}_p05" if protocol == "holdout_p05" else f"{prefix}_p10"
            subset = [row for row in data if row["protocol"] == protocol and row["group"] == name]
            ax.plot(list(colors), [max(next(row["rmse_mean_nm"] for row in subset
                                            if row["model"] == model), 1e-6)
                                   for model in colors],
                    marker="o", label=tool)
        ax.set(title=f"Unseen {payload}", xlabel="Model", yscale="log")
    axes[0].set_ylabel("Torque RMSE (Nm, log scale)")
    axes[1].legend()
    fig.suptitle("Figure 5. Held-out payload generalization")
    save(fig, "figure5_unseen_payload", data)

    # Figure 6: test error against learned parameter count.
    data = _aggregate_metrics(rows, protocol="pooled", scope="test_all", group="all")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in data:
        ax.scatter(row["parameters"] + 1, max(row["rmse_mean_nm"], 1e-6),
                   color=colors[row["model"]], s=75)
        ax.annotate(row["model"], (row["parameters"] + 1, max(row["rmse_mean_nm"], 1e-6)),
                    xytext=(5, 5), textcoords="offset points")
    ax.set(title="Figure 6. Test RMSE versus learned parameters",
           xlabel="Parameters + 1 (log scale)", ylabel="Torque RMSE (Nm, log scale)",
           xscale="log", yscale="log")
    save(fig, "figure6_rmse_vs_parameters", data)
    write_report(output)
    print(f"Saved six PNG/SVG/CSV figures to {figure_root}", flush=True)


def write_report(output: Path) -> None:
    metrics = _read_csv(output / "model_metrics.csv")
    residual = _read_csv(output / "inertia_residual_conditions.csv")
    effects = _read_csv(output / "inertia_condition_effects.csv")
    profiles = _read_csv(output / "gripper_joint_profiles.csv")
    duplicates = _read_csv(output / "split_duplicate_audit.csv")
    selections = _read_csv(output / "selection_counts.csv")
    physics = _read_csv(output / "physical_parameters.csv")
    basis = json.loads((output / "physics_basis.json").read_text(encoding="utf-8"))
    audit_info = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    pooled = {row["model"]: row for row in
              _aggregate_metrics(metrics, protocol="pooled", scope="test_all", group="all")}
    heldout = {}
    pooled_conditions = {row["group"]: row for row in
                         _aggregate_metrics(metrics, protocol="pooled", scope="condition")
                         if row["model"] == "R3"}
    for protocol, conditions in PROTOCOLS.items():
        if not conditions:
            continue
        for row in _aggregate_metrics(metrics, protocol=protocol, scope="condition"):
            if row["group"] in conditions:
                heldout[(protocol, row["group"], row["model"])] = row

    def number(row: dict, field: str, digits: int = 6) -> str:
        return f"{float(row[field]):.{digits}g}"

    def performance(row: dict) -> str:
        mean = number(row, "rmse_mean_nm")
        return f"{mean} ± {number(row, 'rmse_sd_nm', 3)}" if row["seed_count"] > 1 else mean

    lines = [
        "# Condition-adaptive RBD/KAN v1: A0509 simulation study",
        "",
        "## Data integrity and protocol",
        "",
        "The prepared archive contains 296,520 rows from the legacy and wide16_v3 campaigns, "
        "seven conditions and 16 trajectories. Source archive and scene hashes, 22-feature "
        "schema, finite values, condition identity, and label reconstruction were checked. "
        "The three AG145 XML hash differences are campaign-specific mesh paths; their eight "
        "mesh files have identical contents and the XMLs match after path normalization.",
        "",
        "The original split is train T1–T11/T15, validation T12, test T13/T14/T16. "
        f"Exact feature overlap between train and validation comprised "
        f"{duplicates[0]['unique_duplicate_states']} unique states and "
        f"{duplicates[0]['second_rows_matching']} validation rows. "
        "Those rows were excluded from validation before model selection. "
        f"The maximum reference-label spread among these identical states was "
        f"{float(duplicates[0]['max_reference_label_spread_nm']):.3g} Nm. "
        f"Exact train/test and validation/test overlaps were "
        f"{duplicates[1]['unique_duplicate_states']} and "
        f"{duplicates[2]['unique_duplicate_states']}. "
        "All models use the same cleaned split within each protocol.",
        "",
        "The maximum archived-scene endpoint errors were "
        f"{audit_info['scene_audit']['max_endpoint_reference_error_nm']:.3g} Nm "
        "for the reference label and "
        f"{audit_info['scene_audit']['max_endpoint_nominal_error_nm']:.3g} Nm "
        "for the nominal label. The full-row physical scale-family identity differs "
        f"from the reference by at most {basis['full_row_max_scale_family_error_nm']:.3g} Nm.",
        "",
        "Both held-out protocols exclude the target payload from **both grippers** in training "
        "and validation. Bare remains present. Validation uses only seen-condition T12, and "
        "final unseen evaluation uses T13/T14/T16. The nominal RBD for each condition uses its "
        "saved scene; this is conditional transfer with a known simulation model, not a test "
        "of identifying entirely unknown hardware parameters.",
        "",
        "## Nominal RBD inertia mismatch",
        "",
        "Residual: `tau_reference - tau_nominal`. Metrics pool all rows and six joints. "
        "The full 7 × 6 metrics are in `inertia_residual_7x6.csv`.",
        "",
        "| Condition | RMSE Nm | MAE Nm | Max abs Nm | 95th abs Nm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in residual:
        if row["campaign"] == "pooled":
            lines.append(f"| {row['condition']} | {number(row,'rmse_nm')} | "
                         f"{number(row,'mae_nm')} | {number(row,'max_abs_nm')} | "
                         f"{number(row,'p95_abs_nm')} |")
    lines += [
        "",
        "Paired differences below use 32 campaign × trajectory blocks; confidence intervals "
        "come from 2,000 paired block bootstrap resamples.",
        "",
        "| Comparison (right − left) | Mean block RMSE difference Nm | 95% CI Nm |",
        "| --- | ---: | ---: |",
    ]
    for row in effects:
        lines.append(f"| {row['right']} − {row['left']} | "
                     f"{number(row,'mean_paired_rmse_difference_nm')} | "
                     f"[{number(row,'ci95_low_nm')}, {number(row,'ci95_high_nm')}] |")
    lines += [
        "",
        "## R0–R4 on pooled held-out trajectories",
        "",
        "R0 is nominal RBD. R1 fits one global attachment-inertia scale using training labels "
        "and MuJoCo nominal-scene finite differences; it does not use test labels. "
        "R2/R3/R4 learn residuals over **R0**. R2 is six independent velocity splines "
        "(54 parameters). R3 is a 22→30→30→6 MLP, R4 a 22→8→6 edge-spline KAN. "
        "R3 and R4 each have 1,806 parameters and use the same 22 inputs, three seeds, "
        "30-epoch ceiling, batch size, optimizer, and validation rule.",
        "The R1 training time below covers its scalar fit; computing the MuJoCo "
        "physical-sensitivity basis is a separate preprocessing step.",
        "",
        "| Model | Test RMSE Nm | Parameters | Mean training s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in ("R0", "R1", "R2", "R3", "R4"):
        row = pooled[model]
        lines.append(f"| {model} | {performance(row)} | {row['parameters']} | "
                     f"{number(row,'training_mean_s',4)} |")
    lines += [
        "",
        "Per-condition, per-joint, per-campaign and per-seed test metrics are in "
        "`model_metrics.csv`. Training histories are in `training_history.csv`.",
        "",
        "## Entirely unseen payloads",
        "",
        "`holdout_p05`: both 0.5 kg conditions excluded from train/validation. "
        "`holdout_p10`: both 1.0 kg conditions excluded. Values below are final "
        "T13/T14/T16 torque RMSE in Nm.",
        "",
        "| Model | AG145 unseen 0.5 | mDPG unseen 0.5 | AG145 unseen 1.0 | mDPG unseen 1.0 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in ("R0", "R1", "R2", "R3", "R4"):
        entries = [heldout[(protocol, condition, model)]
                   for protocol, condition in (("holdout_p05", "ag145_p05"),
                                               ("holdout_p05", "mdpg_p05"),
                                               ("holdout_p10", "ag145_p10"),
                                               ("holdout_p10", "mdpg_p10"))]
        lines.append(f"| {model} | " + " | ".join(performance(row) for row in entries) + " |")
    lines += [
        "",
        "Selection counts and zero excluded-condition training/validation rows are in "
        "`selection_counts.csv`. Estimated R1 scales are in `physical_parameters.csv`.",
        "",
        "## Answers to the research questions",
        "",
        "1. **Does mismatch grow with payload?** Yes in this synthetic scale-perturbation "
        "dataset. Every paired payload step for each gripper has a positive 95% block-bootstrap "
        "interval; see the table above.",
        "2. **Do grippers have different residual structures?** Their error amplitudes differ, "
        "but the six-joint RMSE profiles are highly aligned "
        f"(cosines {', '.join(number(row,'joint_rmse_profile_cosine',4) for row in profiles)} "
        "for 0/0.5/1.0 kg). These data do not establish a qualitatively distinct joint-wise "
        "residual structure.",
        f"3. **Does physical calibration resolve most error?** R1 estimates attachment inertia "
        f"scales {', '.join(number(row,'inertial_scale_estimate',9) for row in physics)} and "
        f"gets pooled test RMSE {performance(pooled['R1'])} Nm. The nominal models were "
        "deliberately generated at 0.8 scale from the saved scenes; this near-exact recovery "
        "is expected for that known synthetic error family.",
        f"4. **Are MLP/KAN better than the spline branch here?** On pooled test, R2/R3/R4 "
        f"achieve {performance(pooled['R2'])}, {performance(pooled['R3'])}, and "
        f"{performance(pooled['R4'])} Nm. R2 uses only joint velocity, while R3/R4 use "
        "all 22 inputs; the gap cannot be assigned to architecture alone.",
        f"5. **Does KAN beat MLP in this setting?** R3 versus R4 pooled test RMSE is "
        f"{performance(pooled['R3'])} versus {performance(pooled['R4'])} Nm, with mean "
        f"training times {number(pooled['R3'],'training_mean_s',4)} versus "
        f"{number(pooled['R4'],'training_mean_s',4)} s. This fixed-budget experiment "
        "does not support a KAN performance or training-efficiency advantage. "
        "R3 and R4 both selected epoch 30 (the cap) in all three pooled seeds, "
        "so this does not rank their fully converged performance.",
        "6. **Does unseen-payload performance hold?** For R3, the held-out to pooled "
        "same-condition RMSE ratios are "
        + ", ".join(
            f"{condition} {heldout[(protocol, condition, 'R3')]['rmse_mean_nm'] / pooled_conditions[condition]['rmse_mean_nm']:.2f}×"
            for protocol, condition in (("holdout_p05", "ag145_p05"),
                                        ("holdout_p05", "mdpg_p05"),
                                        ("holdout_p10", "ag145_p10"),
                                        ("holdout_p10", "mdpg_p10")))
        + ". Interpolation to 0.5 kg loses less accuracy than extrapolation to 1.0 kg "
        "in this comparison. R1 remains near the simulation label because its single "
        "physical scale transfers across conditions in this constructed family.",
        "7. **Is continual learning justified?** No sequential adaptation or real hardware "
        "distribution shift was measured. These results alone provide no evidence that "
        "continual learning is necessary.",
        "",
        "## Six exported figures",
        "",
        "Each numbered figure has PNG, SVG and source CSV files under `figures/`: "
        "condition RBD mismatch, model comparison, six-joint comparison, payload curves, "
        "held-out payload transfer, and RMSE versus parameter count.",
        "",
        "## What can and cannot be claimed",
        "",
        "**Supported:** In this archived MuJoCo simulation, a deliberately imposed 20% "
        "attachment-inertia scaling error grows with payload, a one-parameter physical "
        "correction almost exactly reverses it, and the fixed-budget 22-input MLP outperforms "
        "the parameter-matched KAN on the pooled test.",
        "",
        "**Unsupported:** measured A0509 motor-torque accuracy, real robot or sim-to-real "
        "performance, a general KAN ranking across architectures and budgets, distinct "
        "gripper residual mechanisms, or a need for continual learning. The separate "
        "21-run J1 friction-compensation pilot is untouched and is not evidence in this study.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "basis", "train", "figures", "all"))
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--source-repo", type=Path, default=SOURCE_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.stage in ("audit", "all"):
        if args.output.exists():
            raise FileExistsError(args.output)
        audit(args.prepared, args.source_repo, args.output, config)
    if args.stage in ("basis", "all"):
        prepare_physics_basis(args.prepared, args.source_repo, args.output)
    if args.stage in ("train", "all"):
        train_and_evaluate(args.prepared, args.output, config)
    if args.stage in ("figures", "all"):
        make_figures(args.output)


if __name__ == "__main__":
    main()
