"""Plot the completed friction validation CSV results without retraining."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .friction_validate import DEFAULT_OUTPUT


ORDER = ("RBD_zero_friction", "Linear_friction", "Jointwise_spline", "MLP", "KAN")
LABELS = ("RBD only", "Linear", "Spline", "MLP", "KAN")
FAMILIES = ("coulomb", "viscous", "stribeck", "asymmetric", "lugre")


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty result: {path}")
    return [row for row in rows if row["split"] == "test"]


def _save(fig, root: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(root / f"{name}.png", dpi=220)
    fig.savefig(root / f"{name}.svg")
    plt.close(fig)


def plot(result: Path) -> None:
    overall = _read(result / "overall.csv")
    scenarios = _read(result / "by_scenario.csv")
    conditions = _read(result / "by_condition.csv")

    means, sds = [], []
    for model in ORDER:
        scores = [float(row["torque_rmse_nm"]) for row in overall if row["model"] == model]
        means.append(np.mean(scores))
        sds.append(np.std(scores, ddof=1) if len(scores) > 1 else 0)
    fig, axis = plt.subplots(figsize=(8.8, 4.5))
    bars = axis.bar(LABELS, means, yerr=sds, capsize=4,
                    color=["#75849b", "#56a194", "#449cb7", "#d29759", "#816bb0"])
    for bar, value in zip(bars, means):
        axis.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
                      xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    axis.set(ylabel="Test torque RMSE (Nm)",
             title="Synthetic friction benchmark · unseen trajectories T13/T14/T16")
    axis.set_ylim(0, max(means) * 1.17)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, result, "test_rmse_models")

    family_values = np.asarray([
        [np.mean([float(row["torque_rmse_nm"]) for row in scenarios
                  if row["model"] == model and row["model_family"] == family])
         for family in FAMILIES] for model in ORDER])
    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    positions = np.arange(len(FAMILIES))
    width = 0.16
    for index, label in enumerate(LABELS):
        axis.bar(positions + (index - 2) * width, family_values[index], width, label=label)
    axis.set_xticks(positions, [name.title() for name in FAMILIES])
    axis.set(ylabel="Mean scenario Test RMSE (Nm)",
             title="Error by synthetic friction law")
    axis.legend(ncol=5, fontsize=8, loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, result, "test_rmse_by_friction_law")

    condition_values = np.asarray([
        [np.mean([float(row["torque_rmse_nm"]) for row in conditions
                  if row["model"] == model and row["condition"] == condition])
         for condition in ("bare_p0", "ag145_p0", "ag145_p05", "ag145_p10",
                           "mdpg_p0", "mdpg_p05", "mdpg_p10")]
        for model in ORDER])
    fig, axis = plt.subplots(figsize=(9.5, 4.4))
    image = axis.imshow(condition_values, cmap="viridis", aspect="auto")
    axis.set_yticks(range(len(ORDER)), LABELS)
    axis.set_xticks(range(7), ("Bare 0", "AG145 0", "AG145 0.5", "AG145 1",
                                    "mDPG 0", "mDPG 0.5", "mDPG 1"), rotation=25, ha="right")
    axis.set_title("Test RMSE by gripper and payload (Nm)")
    fig.colorbar(image, ax=axis, label="Nm")
    _save(fig, result, "test_rmse_by_condition")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.result.resolve())


if __name__ == "__main__":
    main()
