"""Paper-ready PNG/SVG figures and numerical spline probes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .networks import SplineBasis


def _save(fig, folder: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=220)
    fig.savefig(folder / f"{name}.svg")
    plt.close(fig)


def _row(records: list[dict], model: str, *, split: str = "test", scope: str = "all", group: str = "all") -> dict:
    return next(r for r in records if r["scenario"] == "B" and r["split"] == split and
                r["scope"] == scope and r["group"] == group and r["model"] == model)


def _spline_curves(spline: dict, x_grid: np.ndarray) -> np.ndarray:
    basis = SplineBasis(len(spline["knots"]) - 7)
    with torch.inference_mode():
        x = torch.from_numpy(x_grid.astype(np.float32))
        normalized = (x - torch.tensor(spline["input_mean"])) / torch.tensor(spline["input_scale"])
        values = torch.einsum("bin,in->bi", basis(normalized), torch.tensor(spline["coefficients"]))
        values = (values + torch.tensor(spline["bias"])) * torch.tensor(spline["output_scale"]) + torch.tensor(spline["output_mean"])
    return values.numpy()


def make_figures(output: Path, manifest: dict, records: list[dict], seed: int) -> None:
    folder = output / "figures"
    folder.mkdir()
    with np.load(output / "predictions_seed_first.npz", allow_pickle=False) as raw:
        p = {k: raw[k] for k in raw.files}
    with np.load(output / "branch_contributions.npz", allow_pickle=False) as raw:
        branches = {k: raw[k] for k in raw.files}
    test = p["split"] == manifest["split_codes"]["test"]
    r = p["target_B"] - p["M1"]
    m2, m3, m4, m5 = (p[f"{name}_seed_{seed}"] for name in ("M2", "M3", "M4", "M5"))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    for x, y, w, h, label, color in (
        (.3, 1.55, 2.1, .9, "Saved scene RBD\nM(q)ddq + bias + Ddq", "#dce9f5"),
        (3.1, 2.6, 2.4, .9, "Jointwise spline\n6 velocity functions", "#d9efdf"),
        (3.1, .5, 2.4, .9, "Coupled KAN\nq,dq → 8 → 6", "#fce8ce"),
        (6.45, 1.55, 1.2, .9, "Sum", "#ebdef7"),
        (8.1, 1.55, 1.6, .9, "Predicted\ntorque", "#f8dede")):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#39424e"))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=10)
    for start, end in (((2.4, 2), (6.45, 2)), ((5.5, 3.05), (6.45, 2.2)),
                       ((5.5, .95), (6.45, 1.8)), ((7.65, 2), (8.1, 2))):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.set_title("Fig. 1  Physics-structured residual model (M5: staged training)")
    _save(fig, folder, "fig1_architecture")

    spline = json.loads((output / f"jointwise_spline_seed_{seed}.json").read_text())
    raw_dq = p["features"][:, 6:12]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    probes = {}
    rng = np.random.default_rng(123)
    for j, ax in enumerate(axes.flat):
        low, high = np.quantile(raw_dq[test, j], [.01, .99])
        grid = np.linspace(low, high, 300)
        x = np.zeros((len(grid), 6), np.float32); x[:, j] = grid
        curve = _spline_curves(spline, x)[:, j]
        candidates = np.flatnonzero(test)
        sample = rng.choice(candidates, min(700, len(candidates)), replace=False)
        ax.scatter(raw_dq[sample, j], r[sample, j], s=3, alpha=.15, color="#626e79")
        ax.plot(grid, curve, color="#c53d2b", lw=2)
        ax.axvline(0, lw=.6, color="black")
        ax.set_title(f"Joint {j+1}"); ax.set_xlabel("Velocity (rad/s)"); ax.set_ylabel("Residual (Nm)")
        probe = np.zeros((5, 6), np.float32); probe[:, j] = [-.2, -.05, 0, .05, .2]
        values = _spline_curves(spline, probe)[:, j]
        probes[f"joint_{j+1}"] = {"velocity_rad_s": [-.2, -.05, 0, .05, .2],
                                   "predicted_residual_nm": values.tolist(),
                                   "positive_negative_asymmetry_nm_at_0p2": float(values[-1] + values[0]),
                                   "near_zero_jump_nm": float(values[3] - values[1])}
    fig.suptitle("Fig. 2  Six independent velocity splines; gray: MuJoCo friction residual")
    _save(fig, folder, "fig2_jointwise_splines")
    (output / "spline_analysis.json").write_text(json.dumps(probes, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, pred, name in zip(axes, (m3, m4, m5), ("M3 jointwise", "M4 coupled", "M5 structured")):
        actual = r[test].ravel(); estimate = (pred - p["M1"])[test].ravel()
        selection = rng.choice(len(actual), min(5000, len(actual)), replace=False)
        ax.scatter(actual[selection], estimate[selection], s=2, alpha=.2)
        lim = max(np.max(np.abs(actual[selection])), np.max(np.abs(estimate[selection])))
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=.7)
        ax.set_title(name); ax.set_xlabel("Target residual (Nm)"); ax.set_ylabel("Predicted residual (Nm)")
    fig.suptitle("Fig. 3  Held-out residual torque")
    _save(fig, folder, "fig3_residual_scatter")

    fig, ax = plt.subplots(figsize=(11, 4))
    index = np.flatnonzero(test)[:300]
    for values, label, color in ((p["target_B"], "MuJoCo target", "black"), (p["M1"], "M1 scene RBD", "#9b9b9b"),
                                 (m2, "M2 MLP", "#277da1"), (m3, "M3 spline", "#e76f51"), (m4, "M4 KAN", "#2a9d8f")):
        ax.plot(np.arange(len(index)), values[index, 0], label=label, lw=1.2, color=color)
    ax.set_xlabel("Held-out sample index"); ax.set_ylabel("Joint 1 generalized torque (Nm)"); ax.legend(ncol=5, fontsize=8)
    ax.set_title("Fig. 4  RBD, MLP, and spline/KAN torque prediction")
    _save(fig, folder, "fig4_torque_prediction")

    fig, ax = plt.subplots(figsize=(10, 4))
    names = ["M1", f"M2_seed_{seed}", f"M3_seed_{seed}", f"M4_seed_{seed}", f"M5_seed_{seed}"]
    xbar = np.arange(6); width = .15
    for i, name in enumerate(names):
        score = _row(records, name)
        ax.bar(xbar + (i-2)*width, score["per_joint_rmse_nm"], width, label=name)
    ax.set_xticks(xbar, [f"J{i}" for i in range(1, 7)]); ax.set_ylabel("Test RMSE (Nm)"); ax.legend(fontsize=8)
    ax.set_title("Fig. 5  Per-joint test RMSE")
    _save(fig, folder, "fig5_joint_rmse")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, ax in enumerate(axes.flat):
        vel = np.abs(raw_dq[test, j]); edges = np.quantile(vel, np.linspace(0, 1, 9))
        edges = np.unique(edges)
        for estimate, name in ((m3, "M3"), (m4, "M4"), (m5, "M5")):
            error = (estimate - p["target_B"])[test, j]
            centers, rms = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (vel >= lo) & (vel <= hi if hi == edges[-1] else vel < hi)
                if np.any(mask):
                    centers.append(float(np.mean(vel[mask]))); rms.append(float(np.sqrt(np.mean(error[mask]**2))))
            ax.plot(centers, rms, marker="o", ms=3, label=name)
        ax.set_title(f"Joint {j+1}"); ax.set_xlabel("|velocity| (rad/s)"); ax.set_ylabel("RMSE (Nm)")
    axes.flat[0].legend()
    fig.suptitle("Fig. 6  Error across joint-velocity bins")
    _save(fig, folder, "fig6_velocity_error")

    fig, ax = plt.subplots(figsize=(12, 4))
    conds = manifest["conditions"]
    xbar = np.arange(len(conds)); width = .18
    for i, name in enumerate(("M1", f"M2_seed_{seed}", f"M3_seed_{seed}", f"M4_seed_{seed}", f"M5_seed_{seed}")):
        values = [_row(records, name, scope="condition", group=cond)["rmse_nm"] for cond in conds]
        ax.bar(xbar + (i-2)*width, values, width, label=name)
    ax.set_xticks(xbar, conds, rotation=25, ha="right"); ax.set_ylabel("Test RMSE (Nm)"); ax.legend(fontsize=8)
    ax.set_title("Fig. 7  Gripper/payload conditions (shared models)")
    _save(fig, folder, "fig7_condition_rmse")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    joint, coupled = branches["jointwise_nm"][test], branches["coupled_nm"][test]
    axes[0].bar(np.arange(1, 7)-.17, np.sqrt(np.mean(joint**2, axis=0)), .34, label="Jointwise")
    axes[0].bar(np.arange(1, 7)+.17, np.sqrt(np.mean(coupled**2, axis=0)), .34, label="Coupled")
    axes[0].set_xlabel("Joint"); axes[0].set_ylabel("Contribution RMS (Nm)"); axes[0].legend()
    sample = np.flatnonzero(test)[:1200]
    axes[1].plot(r[sample, 0], label="Target residual", lw=1)
    axes[1].plot(branches["jointwise_nm"][sample, 0], label="Jointwise", lw=1)
    axes[1].plot(branches["coupled_nm"][sample, 0], label="Coupled", lw=1)
    axes[1].set_xlabel("Held-out sample index"); axes[1].set_ylabel("Joint 1 torque (Nm)"); axes[1].legend(fontsize=8)
    fig.suptitle("Fig. 8  Stagewise branch contributions")
    _save(fig, folder, "fig8_branch_contributions")
