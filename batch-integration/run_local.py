"""Dry-run the ConDo batch-integration method locally, bypassing viash.

Usage:
    python batch-integration/run_local.py \\
        --divergence kld --transform-type location-scale \\
        --rep features \\
        --input  /path/to/dataset.h5ad \\
        --output /tmp/out.h5ad

Used during development to fit + transform without docker. The same
``run_condo`` entry point is what the viash component invokes — this
script just sets the ``par``/``meta`` dicts the way viash would.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--divergence", choices=["kld", "mmd"], required=True)
    parser.add_argument(
        "--transform-type",
        dest="transform_type",
        choices=["location-scale", "affine", "diagonal-plus-low-rank"],
        required=True,
    )
    parser.add_argument(
        "--dplr-rank", dest="dplr_rank", type=int, default=16,
        help="rank of the low-rank perturbation in diagonal-plus-low-rank",
    )
    parser.add_argument(
        "--rep", choices=["features", "pca"], default="features"
    )
    parser.add_argument("--hvg-only", dest="hvg_only", action="store_true")
    parser.add_argument("--bootstrap-fraction", type=float, default=1.0)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mmd-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--wd-on-bias", dest="wd_on_bias",
                        action="store_true",
                        help="apply --weight-decay to the bias (location) "
                             "parameter too (default: bias has weight_decay=0)")
    parser.add_argument("--patience", type=int, default=3,
                        help="early-stopping patience for the MMD training")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: 'cpu', 'cuda', or e.g. 'cuda:0'",
    )
    parser.add_argument(
        "--method-dir",
        default=os.environ.get("CONDO_METHOD_DIR"),
        help=(
            "Path to task_batch_integration_forked/src/methods/condo (contains "
            "condo_runner.py). Defaults to $CONDO_METHOD_DIR or the sibling "
            "task_batch_integration_forked checkout."
        ),
    )
    parser.add_argument(
        "--utils-dir",
        default=os.environ.get("CONDO_UTILS_DIR"),
        help=(
            "Path to task_batch_integration_forked/src/utils (contains "
            "read_anndata_partial.py)."
        ),
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    default_fork = here.parent.parent / "task_batch_integration_forked"
    method_dir = Path(
        args.method_dir or (default_fork / "src" / "methods" / "condo")
    )
    utils_dir = Path(args.utils_dir or (default_fork / "src" / "utils"))

    if not (method_dir / "condo_runner.py").exists():
        parser.error(
            f"condo_runner.py not found under {method_dir}; pass --method-dir."
        )

    sys.path.insert(0, str(method_dir))
    from condo_runner import run_condo  # noqa: E402

    name_parts = ["condo", args.divergence, args.transform_type.replace("-", "_")]
    if args.rep == "pca":
        name_parts.append("pca")
    if args.hvg_only:
        name_parts.append("hvg")
    name = "_".join(name_parts)

    par = {
        "input": args.input,
        "output": args.output,
        "divergence": args.divergence,
        "transform_type": args.transform_type,
        "rep": args.rep,
        "hvg_only": args.hvg_only,
        "bootstrap_fraction": args.bootstrap_fraction,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "mmd_size": args.mmd_size,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "wd_on_bias": args.wd_on_bias,
        "patience": args.patience,
        "dplr_rank": args.dplr_rank,
        "random_state": args.random_state,
        "device": args.device,
    }
    meta = {"name": name, "resources_dir": str(utils_dir)}
    run_condo(par, meta)


if __name__ == "__main__":
    main()
