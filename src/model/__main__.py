"""Offline research entry point: python -m src.model {prepare,train,evaluate,ablate}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import prepare
from .train import train_all
from .evaluate import evaluate
from .ablation import run as run_ablation


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "ablate"))
    parser.add_argument("--archive", type=Path, default=Path(r"D:\Code\sdml-doosan-robot\data\grid16_legacy_joint_dataset_m0_m4"))
    parser.add_argument("--legacy-repo", type=Path, default=Path(r"D:\Code\sdml-doosan-robot"))
    parser.add_argument("--prepared", type=Path, default=ROOT / "result/structured_kan/prepared_d")
    parser.add_argument("--models", type=Path, default=ROOT / "result/structured_kan/models_d")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "result/structured_kan/evaluation_d")
    parser.add_argument("--ablations", type=Path, default=ROOT / "result/structured_kan/ablations_d")
    parser.add_argument("--config", type=Path, default=ROOT / "src/model/config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.command != "prepare" else None
    if args.command == "prepare":
        result = prepare(args.archive, args.legacy_repo, args.prepared)
        print(json.dumps({"rows": result["rows"], "scene_max_error_nm": result["scene_reconstruction_max_abs_nm"]}, indent=2))
    elif args.command == "train":
        result = train_all(args.prepared, args.models, config)
        print(json.dumps({"runs": len(result["runs"]), "training_rows": result["training_rows"]}, indent=2))
    elif args.command == "evaluate":
        result = evaluate(args.prepared, args.models, args.evaluation)
        print(json.dumps({"records": len(result["records"]), "output": str(args.evaluation.resolve())}, indent=2))
    else:
        result = run_ablation(args.prepared, args.ablations, config)
        print(json.dumps({"runs": len(result), "output": str(args.ablations.resolve())}, indent=2))


if __name__ == "__main__":
    main()
