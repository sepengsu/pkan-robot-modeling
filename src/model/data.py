"""Validate the archived seven-condition dataset and build MuJoCo friction labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


EXPECTED_SPLITS = {
    "train": ["T5", "T6", "T7", "T8", "T9", "T10", "T11", "T15"],
    "validation": ["T12"],
    "test": ["T13", "T14", "T16"],
    "diagnostic": ["T1", "T2", "T3", "T4"],
}
FRICTION_NM = np.array([0.40, 0.35, 0.25, 0.15, 0.12, 0.08], dtype=np.float64)


def load_archived(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("profile") != "trajectory16_legacy" or manifest.get("split_policy") != EXPECTED_SPLITS:
        raise ValueError("archived dataset does not have the requested trajectory split")
    npz = root / "dataset.npz"
    if hashlib.sha256(npz.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("archived dataset SHA-256 mismatch")
    with np.load(npz, allow_pickle=False) as raw:
        data = {k: raw[k] for k in ("features", "tau_reference", "tau_nominal", "condition", "trajectory", "split")}
    n = len(data["features"])
    if data["features"].shape != (n, 22) or any(data[k].shape != (n, 6) for k in ("tau_reference", "tau_nominal")):
        raise ValueError("invalid archived array shapes")
    if any(len(v) != n for v in data.values()) or not all(np.isfinite(data[k]).all() for k in ("features", "tau_reference", "tau_nominal")):
        raise ValueError("invalid archived array lengths or values")
    if len(manifest["trials"]) != len({(x["condition"], x["trajectory"]) for x in manifest["trials"]}):
        raise ValueError("duplicate condition/trajectory trial")
    if set(manifest["conditions"]) != {"bare_p0", "ag145_p0", "ag145_p05", "ag145_p10", "mdpg_p0", "mdpg_p05", "mdpg_p10"}:
        raise ValueError("unexpected campaign conditions")
    for name, code in manifest["split_codes"].items():
        if int(np.sum(data["split"] == code)) != manifest["rows"][name]:
            raise ValueError(f"{name} row count mismatch")
    for trial in manifest["trials"]:
        code = manifest["conditions"].index(trial["condition"])
        mask = (data["condition"] == code) & (data["trajectory"] == int(trial["trajectory"][1:]))
        if int(mask.sum()) != trial["selected_rows"] or not np.all(data["split"][mask] == manifest["split_codes"][trial["split"]]):
            raise ValueError(f"trial/split mismatch: {trial['condition']}/{trial['trajectory']}")
    return manifest, data


def prepare(archive: Path, legacy_repo: Path, output: Path) -> dict:
    import mujoco

    # Reuse the source repository's RBD and scene asset resolver without altering it.
    sys.path.insert(0, str(legacy_repo.resolve()))
    from model.campaign_rbd import load_scene
    from model.rbd import RBDModel

    source, data = load_archived(archive)
    if output.exists():
        raise FileExistsError(output)
    target_b = np.empty_like(data["tau_reference"])
    exact_errors = []
    scene_audit = {}
    for ci, condition in enumerate(source["conditions"]):
        scene = legacy_repo / "data/grid_16_trajectories/models/conditions" / f"{condition}.xml"
        if hashlib.sha256(scene.read_bytes()).hexdigest() != source["scene_sha256"][condition]:
            raise ValueError(f"scene digest mismatch: {condition}")
        exact = RBDModel(load_scene(scene))
        friction_model = load_scene(scene)
        dof = exact.dof_addresses
        friction_model.dof_frictionloss[dof] = FRICTION_NM
        friction_data = mujoco.MjData(friction_model)
        rows = np.flatnonzero((data["condition"] == ci) & (data["split"] != source["split_codes"]["diagnostic"]))
        for index in rows:
            q = data["features"][index, :6].astype(np.float64)
            dq = data["features"][index, 6:12].astype(np.float64)
            ddq = data["features"][index, 12:18].astype(np.float64)
            torque_a = exact.evaluate(q, dq, ddq)["torque"]
            exact_errors.append(np.max(np.abs(torque_a - data["tau_reference"][index])))
            friction_data.qpos[:] = friction_model.qpos0
            friction_data.qvel[:] = 0
            friction_data.qacc[:] = 0
            friction_data.qpos[exact.qpos_addresses] = q
            friction_data.qvel[dof] = dq
            mujoco.mj_forward(friction_model, friction_data)
            friction_data.qacc[:] = 0
            friction_data.qacc[dof] = ddq
            mujoco.mj_inverse(friction_model, friction_data)
            target_b[index] = friction_data.qfrc_inverse[dof]
        scene_audit[condition] = {
            "sha256": source["scene_sha256"][condition],
            "gravity": exact.model.opt.gravity.tolist(),
            "damping": exact.model.dof_damping[dof].tolist(),
            "armature": exact.model.dof_armature[dof].tolist(),
            "frictionloss_A": exact.model.dof_frictionloss[dof].tolist(),
            "frictionloss_B": friction_model.dof_frictionloss[dof].tolist(),
            "bodies": {mujoco.mj_id2name(exact.model, mujoco.mjtObj.mjOBJ_BODY, i): {
                "mass_kg": float(exact.model.body_mass[i]),
                "inertia_kg_m2": exact.model.body_inertia[i].tolist(),
            } for i in range(exact.model.nbody)},
        }
    if max(exact_errors) > 2e-5:
        raise ValueError(f"saved scene does not reproduce qfrc_inverse: max {max(exact_errors):.7g} Nm")
    keep = data["split"] != source["split_codes"]["diagnostic"]
    arrays = {k: v[keep] for k, v in data.items()}
    arrays["tau_friction"] = target_b[keep]
    output.mkdir(parents=True)
    np.savez_compressed(output / "dataset.npz", **arrays)
    report = {
        "source_dataset_sha256": source["dataset_sha256"],
        "source_manifest": str((archive / "manifest.json").resolve()),
        "label_A": "saved MuJoCo scene qfrc_inverse; not motor torque",
        "label_B": "MuJoCo mj_inverse with added joint frictionloss; not motor torque",
        "scene_reconstruction_max_abs_nm": float(max(exact_errors)),
        "conditions": source["conditions"], "split_codes": source["split_codes"],
        "split_policy": source["split_policy"],
        "rows": {name: int(np.sum(arrays["split"] == code)) for name, code in source["split_codes"].items() if name != "diagnostic"},
        "trials": [x for x in source["trials"] if x["split"] != "diagnostic"],
        "physics": scene_audit,
        "friction_difference_nm": {
            "rmse": float(np.sqrt(np.mean((arrays["tau_friction"] - arrays["tau_reference"]) ** 2))),
            "joint_rmse": np.sqrt(np.mean((arrays["tau_friction"] - arrays["tau_reference"]) ** 2, axis=0)).tolist(),
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def load_prepared(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    with np.load(root / "dataset.npz", allow_pickle=False) as raw:
        arrays = {k: raw[k] for k in raw.files}
    for name, code in manifest["split_codes"].items():
        if name != "diagnostic" and int(np.sum(arrays["split"] == code)) != manifest["rows"][name]:
            raise ValueError(f"prepared {name} row count mismatch")
    return manifest, arrays
