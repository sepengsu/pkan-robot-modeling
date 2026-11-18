"""Fair, seed-controlled training of M2-M5 on the same prepared split."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .data import load_prepared
from .networks import CoupledKAN, JointwiseSpline, ResidualMLP, parameter_count


class NormalizedBranch(nn.Module):
    def __init__(self, network: nn.Module, columns: list[int], x_mean: torch.Tensor,
                 x_scale: torch.Tensor, y_mean: torch.Tensor, y_scale: torch.Tensor):
        super().__init__()
        self.network = network
        self.columns = columns
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_scale", x_scale)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_scale", y_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        z = (features[:, self.columns] - self.x_mean) / self.x_scale
        return self.network(z) * self.y_scale + self.y_mean

    def smoothness(self) -> torch.Tensor:
        return self.network.smoothness()


def _arrays(data_root: Path):
    manifest, arrays = load_prepared(data_root)
    x = torch.from_numpy(arrays["features"].astype(np.float32, copy=False))
    a = torch.from_numpy(arrays["tau_reference"].astype(np.float32, copy=False))
    b = torch.from_numpy(arrays["tau_friction"].astype(np.float32, copy=False))
    # M1 is exact-scene RBD; M2-M5 model only added friction generalized force.
    y = b - a
    train = np.flatnonzero(arrays["split"] == manifest["split_codes"]["train"])
    val = np.flatnonzero(arrays["split"] == manifest["split_codes"]["validation"])
    return manifest, arrays, x, y, train, val


def _new_branch(kind: str, x: torch.Tensor, target: torch.Tensor, train: np.ndarray,
                intervals: int = 5, use_ddq: bool = False) -> NormalizedBranch:
    columns = list(range(6, 12)) if kind == "M3" else list(range(18 if use_ddq else 12))
    x_train = x[train][:, columns]
    y_train = target[train]
    x_mean = x_train.mean(0)
    x_scale = x_train.std(0, unbiased=False).clamp_min(0.01)
    y_mean = y_train.mean(0)
    y_scale = y_train.std(0, unbiased=False).clamp_min(0.02)
    if kind == "M2":
        network = ResidualMLP(len(columns))
    elif kind == "M3":
        network = JointwiseSpline(intervals)
    elif kind in ("M4", "M5"):
        network = CoupledKAN(len(columns), intervals)
    else:
        raise ValueError(kind)
    return NormalizedBranch(network, columns, x_mean, x_scale, y_mean, y_scale)


def _fit(model: NormalizedBranch, x: torch.Tensor, target: torch.Tensor,
         train: np.ndarray, val: np.ndarray, config: dict, seed: int) -> tuple[list[dict], int, float]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        permutation = torch.as_tensor(np.random.permutation(train), dtype=torch.long)
        total, seen = 0.0, 0
        for indices in permutation.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            predicted = model(x[indices])
            torque_loss = ((predicted - target[indices]) / model.y_scale).square().mean()
            smooth_loss = model.smoothness()
            correction_loss = predicted.square().mean()
            loss = (torque_loss + config["lambda_smooth"] * smooth_loss
                    + config["lambda_residual"] * correction_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
        model.eval()
        with torch.inference_mode():
            parts = [model(x[idx]) for idx in torch.as_tensor(val).split(config["batch_size"])]
            pred_val = torch.cat(parts)
            error = pred_val - target[val]
            val_loss = float((error / model.y_scale).square().mean())
            rmse = torch.sqrt(error.square().mean(0)).tolist()
            total_rmse = float(torch.sqrt(error.square().mean()))
        elapsed = time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": total / seen,
                        "validation_loss": val_loss, "validation_rmse_nm": total_rmse,
                        "validation_joint_rmse_nm": rmse, "elapsed_s": elapsed})
        if not math_isfinite(val_loss):
            raise ValueError("nonfinite validation loss")
        if val_loss < best_loss - config["min_delta"]:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= config["patience"]:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return history, best_epoch, time.perf_counter() - start


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))


def _checkpoint(kind: str, model: NormalizedBranch, config: dict, seed: int,
                history: list[dict], best_epoch: int, training_s: float, extra: dict | None = None) -> dict:
    return {"kind": kind, "state": model.state_dict(), "columns": model.columns,
            "intervals": config["grid_intervals"], "seed": seed,
            "config": config, "history": history, "best_epoch": best_epoch,
            "training_s": training_s, "parameter_count": parameter_count(model),
            "trainable_parameter_count": parameter_count(model, trainable_only=True),
            **(extra or {})}


def train_all(data_root: Path, output: Path, config: dict) -> dict:
    _, arrays, x, y, train, val = _arrays(data_root)
    if output.exists():
        raise FileExistsError(f"choose a new training output: {output}")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.set_num_threads(config.get("threads", 1))
    summary = []
    for seed in config["seeds"]:
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        models = {}
        for kind in ("M2", "M3", "M4"):
            torch.manual_seed(seed)
            model = _new_branch(kind, x, y, train, config["grid_intervals"])
            history, best_epoch, duration = _fit(model, x, y, train, val, config, seed)
            checkpoint = _checkpoint(kind, model, config, seed, history, best_epoch, duration)
            torch.save(checkpoint, seed_dir / f"{kind}.pt")
            models[kind] = model
            summary.append({"seed": seed, "model": kind, "best_epoch": best_epoch,
                            "validation_rmse_nm": history[best_epoch - 1]["validation_rmse_nm"],
                            "parameters": checkpoint["parameter_count"], "training_s": duration})
            print(f"seed {seed} {kind}: epoch {best_epoch}, val {summary[-1]['validation_rmse_nm']:.5f} Nm, {duration:.1f}s", flush=True)
        # Freeze the M3 function and learn only what its six 1D functions leave unexplained.
        with torch.inference_mode():
            joint_prediction = torch.cat([models["M3"](part) for part in x.split(config["batch_size"])])
        remaining = y - joint_prediction
        torch.manual_seed(seed)
        coupled = _new_branch("M5", x, remaining, train, config["grid_intervals"])
        history, best_epoch, duration = _fit(coupled, x, remaining, train, val, config, seed)
        checkpoint = _checkpoint("M5", coupled, config, seed, history, best_epoch, duration,
                                 {"jointwise_checkpoint": "M3.pt", "stagewise": True,
                                  "total_parameter_count": parameter_count(models["M3"]) + parameter_count(coupled)})
        torch.save(checkpoint, seed_dir / "M5.pt")
        summary.append({"seed": seed, "model": "M5", "best_epoch": best_epoch,
                        "validation_rmse_nm": history[best_epoch - 1]["validation_rmse_nm"],
                        "parameters": checkpoint["total_parameter_count"],
                        "training_s": duration + next(row["training_s"] for row in summary if row["seed"] == seed and row["model"] == "M3")})
        print(f"seed {seed} M5: epoch {best_epoch}, stage val {summary[-1]['validation_rmse_nm']:.5f} Nm, {duration:.1f}s", flush=True)
    with (output / "training_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    return {"runs": summary, "training_rows": int(len(train)), "validation_rows": int(len(val)),
            "conditions": sorted(set(int(c) for c in arrays["condition"]))}


def load_branch(path: Path) -> tuple[NormalizedBranch, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state"]
    columns = checkpoint["columns"]
    kind = checkpoint["kind"]
    intervals = checkpoint["intervals"]
    if kind == "M2":
        network = ResidualMLP(len(columns))
    elif kind == "M3":
        network = JointwiseSpline(intervals)
    else:
        network = CoupledKAN(len(columns), intervals)
    model = NormalizedBranch(network, columns, state["x_mean"], state["x_scale"],
                             state["y_mean"], state["y_scale"])
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint
