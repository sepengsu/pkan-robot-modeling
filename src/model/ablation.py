"""Controlled grid, smoothness, and acceleration-input ablations for M4."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import metrics
from .train import _arrays, _fit, _new_branch
from .networks import parameter_count


def run(data_root: Path, output: Path, config: dict) -> list[dict]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest, arrays, x, target, train, val = _arrays(data_root)
    test = np.flatnonzero(arrays["split"] == manifest["split_codes"]["test"])
    torch.set_num_threads(config.get("threads", 1))
    rows = []
    for intervals in (3, 5, 8):
        for use_ddq in (False, True):
            for smooth in (False, True):
                for seed in config["seeds"]:
                    specific = dict(config, grid_intervals=intervals,
                                    lambda_smooth=config["lambda_smooth"] if smooth else 0.0)
                    torch.manual_seed(seed)
                    model = _new_branch("M4", x, target, train, intervals, use_ddq)
                    history, best_epoch, duration = _fit(model, x, target, train, val, specific, seed)
                    model.eval()
                    with torch.inference_mode():
                        pred = torch.cat([model(batch) for batch in x[test].split(1024)]).numpy()
                    result = metrics(target[test].numpy(), pred)
                    row = {"grid_intervals": intervals, "inputs": "q,dq,ddq" if use_ddq else "q,dq",
                           "smoothness": smooth, "seed": seed, "best_epoch": best_epoch,
                           "parameters": parameter_count(model), "training_s": duration,
                           "validation_rmse_nm": history[best_epoch-1]["validation_rmse_nm"],
                           "test_rmse_nm": result["rmse_nm"],
                           "test_mae_nm": result["mae_nm"],
                           "test_max_abs_nm": result["max_abs_error_nm"]}
                    rows.append(row)
                    print(f"grid={intervals} ddq={use_ddq} smooth={smooth} seed={seed}: val={row['validation_rmse_nm']:.5f} test={row['test_rmse_nm']:.5f}", flush=True)
    with (output / "ablation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (output / "ablation.json").write_text(json.dumps({"config": config, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return rows
