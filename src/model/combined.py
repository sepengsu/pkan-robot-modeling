"""Prepare and run a shared M0-M5 study on legacy and wide16_v3 campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from .data import FRICTION_NM, load_prepared


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = Path(r"D:\Code\sdml-doosan-robot")
CONDITIONS = ("bare_p0", "ag145_p0", "ag145_p05", "ag145_p10",
              "mdpg_p0", "mdpg_p05", "mdpg_p10")
FEATURE_ORDER = "q1..6, dq1..6, ddq1..6, bare/ag145/mdpg one-hot, payload_mass_kg"
CAMPAIGNS = (
    ("legacy", "grid16_legacy_joint_dataset_m0_m4", "grid_16_trajectories", "trajectory16_legacy"),
    ("wide16_v3", "wide16_v3_joint_dataset", "grid_16_wide_v3", "wide16_v3"),
)
SPLIT_POLICY = {
    "train": [f"T{i}" for i in range(1, 12)] + ["T15"],
    "validation": ["T12"],
    "test": ["T13", "T14", "T16"],
}
SPLIT_CODES = {name: code for code, name in enumerate(SPLIT_POLICY)}
TRAJECTORY_SPLIT = {int(t[1:]): code for name, code in SPLIT_CODES.items()
                    for t in SPLIT_POLICY[name]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_campaign(name: str, archive: Path, campaign_root: Path, profile: str) -> dict:
    manifest_path = archive / "manifest.json"
    dataset_path = archive / "dataset.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("profile") != profile or tuple(manifest.get("conditions", ())) != CONDITIONS:
        raise ValueError(f"wrong profile or conditions in {archive}")
    if manifest.get("feature_order") != FEATURE_ORDER or manifest.get("nominal_inertial_scale") != 0.8:
        raise ValueError(f"incompatible feature or nominal definition in {archive}")
    if manifest.get("torque_label") != "MuJoCo qfrc_inverse, not motor torque":
        raise ValueError(f"incompatible torque label in {archive}")
    if sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError(f"archive digest mismatch: {dataset_path}")
    trials = manifest.get("trials", [])
    pairs = {(row["condition"], row["trajectory"]) for row in trials}
    if len(trials) != 112 or pairs != {(c, f"T{t}") for c in CONDITIONS for t in range(1, 17)}:
        raise ValueError(f"expected one selected trial for each condition and T1-T16: {archive}")
    with np.load(dataset_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ("features", "tau_reference", "tau_nominal",
                                               "condition", "trajectory", "split")}
    count = len(arrays["features"])
    if (arrays["features"].shape != (count, 22)
            or arrays["tau_reference"].shape != (count, 6)
            or arrays["tau_nominal"].shape != (count, 6)
            or any(arrays[key].shape != (count,) for key in ("condition", "trajectory", "split"))):
        raise ValueError(f"invalid archive shapes: {archive}")
    if not all(np.isfinite(arrays[key]).all() for key in ("features", "tau_reference", "tau_nominal")):
        raise ValueError(f"nonfinite archive data: {archive}")
    if set(np.unique(arrays["condition"])) != set(range(7)) or set(np.unique(arrays["trajectory"])) != set(range(1, 17)):
        raise ValueError(f"missing condition or trajectory: {archive}")
    for row in trials:
        mask = ((arrays["condition"] == CONDITIONS.index(row["condition"]))
                & (arrays["trajectory"] == int(row["trajectory"][1:])))
        if int(mask.sum()) != row["selected_rows"]:
            raise ValueError(f"trial row count mismatch: {archive}, {row['condition']}/{row['trajectory']}")
        expected_code = manifest["split_codes"][row["split"]]
        if not np.all(arrays["split"][mask] == expected_code):
            raise ValueError(f"source split mismatch: {archive}, {row['condition']}/{row['trajectory']}")
    for split_name, split_code in manifest["split_codes"].items():
        if int((arrays["split"] == split_code).sum()) != manifest["rows"][split_name]:
            raise ValueError(f"source split row count mismatch: {archive}, {split_name}")
    scenes = {}
    for condition in CONDITIONS:
        scene = campaign_root / "models" / "conditions" / f"{condition}.xml"
        if sha256(scene) != manifest["scene_sha256"][condition]:
            raise ValueError(f"scene digest mismatch: {scene}")
        scenes[condition] = scene
    return {"name": name, "archive": archive, "manifest": manifest, "manifest_sha256": sha256(manifest_path),
            "arrays": arrays, "scenes": scenes, "campaign_root": campaign_root}


def inspect(legacy: Path, wide: Path, source_repo: Path) -> tuple[list[dict], dict]:
    paths = {"legacy": legacy, "wide16_v3": wide}
    loaded = [_load_campaign(name, paths[name], source_repo / "data" / campaign_dir, profile)
              for name, _, campaign_dir, profile in CAMPAIGNS]
    rows = {name: 0 for name in SPLIT_CODES}
    per_campaign = {}
    for item in loaded:
        trajectory = item["arrays"]["trajectory"]
        counts = {name: int(np.isin(trajectory, [int(t[1:]) for t in ids]).sum())
                  for name, ids in SPLIT_POLICY.items()}
        per_campaign[item["name"]] = counts
        for name, count in counts.items():
            rows[name] += count
    a, b = (np.ascontiguousarray(item["arrays"]["features"]) for item in loaded)
    av = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    bv = b.view(np.dtype((np.void, b.dtype.itemsize * b.shape[1]))).ravel()
    overlap = int(len(np.intersect1d(av, bv)))
    return loaded, {"rows": rows, "rows_by_campaign": per_campaign, "total_rows": sum(rows.values()),
                    "exact_feature_row_overlap": overlap}


def _friction_labels(item: dict) -> tuple[np.ndarray, float]:
    import mujoco

    from model.campaign_rbd import load_scene
    from model.rbd import RBDModel

    arrays = item["arrays"]
    features = arrays["features"]
    labels = np.empty_like(arrays["tau_reference"])
    max_exact_error = 0.0
    for ci, condition in enumerate(CONDITIONS):
        scene = item["scenes"][condition]
        exact = RBDModel(load_scene(scene))
        model = load_scene(scene)
        dof = exact.dof_addresses
        model.dof_frictionloss[dof] = FRICTION_NM
        state = mujoco.MjData(model)
        indices = np.flatnonzero(arrays["condition"] == ci)
        # Verify both endpoints of every trajectory against its archived label.
        for trajectory in range(1, 17):
            trial_indices = indices[arrays["trajectory"][indices] == trajectory]
            for index in (int(trial_indices[0]), int(trial_indices[-1])):
                q, dq, ddq = (features[index, start:start + 6].astype(np.float64)
                              for start in (0, 6, 12))
                error = np.max(np.abs(exact.evaluate(q, dq, ddq)["torque"]
                                      - arrays["tau_reference"][index]))
                max_exact_error = max(max_exact_error, float(error))
        for index in indices:
            q, dq, ddq = (features[index, start:start + 6].astype(np.float64)
                          for start in (0, 6, 12))
            state.qpos[:] = model.qpos0
            state.qvel[:] = 0
            state.qacc[:] = 0
            state.qpos[exact.qpos_addresses] = q
            state.qvel[dof] = dq
            mujoco.mj_forward(model, state)
            state.qacc[:] = 0
            state.qacc[dof] = ddq
            mujoco.mj_inverse(model, state)
            labels[index] = state.qfrc_inverse[dof]
        print(f"{item['name']}/{condition}: {len(indices)} friction labels", flush=True)
    if max_exact_error > 2e-5 or not np.isfinite(labels).all():
        raise ValueError(f"scene or label mismatch: {item['name']}, max error {max_exact_error}")
    return labels, max_exact_error


def prepare(legacy: Path, wide: Path, source_repo: Path, output: Path) -> dict:
    loaded, audit = inspect(legacy, wide, source_repo)
    if output.exists():
        raise FileExistsError(f"choose a new preparation output: {output}")
    sys.path.insert(0, str(source_repo.resolve()))
    friction = {}
    exact_errors = {}
    for item in loaded:
        friction[item["name"]], exact_errors[item["name"]] = _friction_labels(item)
    chunks = {key: [] for key in ("features", "tau_reference", "tau_nominal", "tau_friction",
                                  "condition", "trajectory", "split", "campaign")}
    for code, item in enumerate(loaded):
        arrays = item["arrays"]
        count = len(arrays["features"])
        for key in ("features", "tau_reference", "tau_nominal", "condition", "trajectory"):
            chunks[key].append(arrays[key])
        chunks["tau_friction"].append(friction[item["name"]])
        chunks["campaign"].append(np.full(count, code, dtype=np.uint8))
        chunks["split"].append(np.asarray([TRAJECTORY_SPLIT[int(t)] for t in arrays["trajectory"]],
                                          dtype=np.uint8))
    combined = {key: np.concatenate(parts) for key, parts in chunks.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".combined_", dir=output.parent))
    try:
        np.savez_compressed(temporary / "dataset.npz", **combined)
        report = {
            "schema_version": 2,
            "experiment": "legacy + wide16_v3, shared simulated friction residual",
            "conditions": CONDITIONS,
            "campaigns": [item["name"] for item in loaded],
            "feature_order": FEATURE_ORDER,
            "split_policy": SPLIT_POLICY,
            "split_codes": SPLIT_CODES,
            "rows": audit["rows"],
            "rows_by_campaign": audit["rows_by_campaign"],
            "exact_feature_row_overlap": audit["exact_feature_row_overlap"],
            "frictionloss_nm": FRICTION_NM.tolist(),
            "label_A": "archived MuJoCo qfrc_inverse; not motor torque",
            "label_B": "MuJoCo mj_inverse with additional joint frictionloss; not motor torque",
            "source_archives": {item["name"]: {"path": str(item["archive"].resolve()),
                                               "dataset_sha256": item["manifest"]["dataset_sha256"],
                                               "manifest_sha256": item["manifest_sha256"],
                                               "original_split_policy": item["manifest"]["split_policy"],
                                               "scene_sha256": item["manifest"]["scene_sha256"],
                                               "max_exact_scene_error_nm": exact_errors[item["name"]]}
                                for item in loaded},
            "dataset_sha256": sha256(temporary / "dataset.npz"),
        }
        (temporary / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare", "train", "evaluate"))
    parser.add_argument("--legacy", type=Path, default=ROOT / "data/grid16_legacy_joint_dataset_m0_m4")
    parser.add_argument("--wide", type=Path, default=ROOT / "data/wide16_v3_joint_dataset")
    parser.add_argument("--source-repo", type=Path, default=SOURCE_REPO)
    parser.add_argument("--prepared", type=Path, default=ROOT / "result/combined_legacy_wide_v3/prepared")
    parser.add_argument("--models", type=Path, default=ROOT / "result/combined_legacy_wide_v3/models")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "result/combined_legacy_wide_v3/evaluation")
    parser.add_argument("--config", type=Path, default=ROOT / "src/model/combined_config.json")
    args = parser.parse_args()
    if args.command == "validate":
        _, summary = inspect(args.legacy, args.wide, args.source_repo)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        report = prepare(args.legacy, args.wide, args.source_repo, args.prepared)
        print(json.dumps({"rows": report["rows"], "prepared": str(args.prepared.resolve())}, indent=2))
    elif args.command == "train":
        from .train import train_all
        manifest, _ = load_prepared(args.prepared)
        if manifest.get("schema_version") != 2 or manifest.get("campaigns") != ["legacy", "wide16_v3"]:
            raise ValueError("train requires the combined prepared dataset")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = train_all(args.prepared, args.models, config)
        print(json.dumps({"runs": len(result["runs"]), "training_rows": result["training_rows"]}, indent=2))
    else:
        from .evaluate import evaluate
        result = evaluate(args.prepared, args.models, args.evaluation)
        print(json.dumps({"records": len(result["records"]), "evaluation": str(args.evaluation.resolve())}, indent=2))


if __name__ == "__main__":
    main()
