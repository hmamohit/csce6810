#!/usr/bin/env python3
"""
End-to-end pipeline runner for multimodal Hi-C → TPM.

Steps (run in order):
  0. convert    — text Hi-C → float32 memmap (one-time, resumable)
  1. build      — EP regions + NPZ shards + manifest + splits
  2. splits     — rescan NPZ → manifest + splits only (fast)
  3. train      — train model for one or all profiles
  4. test       — evaluate checkpoint on all split modes
  5. analyze    — plots from test predictions CSV

Examples:
  # Smoke test (small chroms first)
  python scripts/run_pipeline.py convert --organism hg38 --chroms chr21
  python scripts/run_pipeline.py build --skip-ep --organism hg38 --chroms chr21 --resume

  # Full first-time pipeline
  python scripts/run_pipeline.py convert --workers 4
  python scripts/run_pipeline.py build --skip-ep --resume --require-memmap

  # After config change: refresh splits only
  python scripts/run_pipeline.py splits

  # Full chain with convert + build + train + test
  python scripts/run_pipeline.py all --convert --skip-ep --profile hg38 --chroms chr21
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSCE = Path(__file__).resolve().parents[1]
PROFILES = ("hg38", "mm10", "multiorganism")


def _run(cmd: list[str], step: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"STEP: {step}")
    print(f"CMD:  {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=CSCE)
    if result.returncode != 0:
        raise SystemExit(f"Step failed ({step}), exit code {result.returncode}")


def _python() -> str:
    return sys.executable


def _checkpoint(profile: str) -> Path:
    return ROOT / "data" / "output" / f"{profile}_multimodal" / f"{profile}_multimodal_best.pt"


def _chrom_args(args: argparse.Namespace) -> list[str]:
    return ["--chroms", args.chroms] if getattr(args, "chroms", None) else []


def step_convert(args: argparse.Namespace) -> None:
    cmd = [_python(), "scripts/convert_hic_memmap.py"]
    if args.organism != "all":
        cmd.extend(["--organism", args.organism])
    cmd.extend(_chrom_args(args))
    if args.workers > 1:
        cmd.extend(["--workers", str(args.workers)])
    _run(cmd, "convert (text Hi-C → memmap)")


def step_build(args: argparse.Namespace) -> None:
    cmd = [_python(), "scripts/build_dataset.py"]
    if args.skip_ep:
        cmd.append("--skip-ep")
    if args.organism != "all":
        cmd.extend(["--organism", args.organism])
    cmd.extend(_chrom_args(args))
    if args.resume:
        cmd.append("--resume")
    if args.require_memmap:
        cmd.append("--require-memmap")
    _run(cmd, "build (EP + NPZ + manifest + splits)")


def step_splits(args: argparse.Namespace) -> None:
    cmd = [_python(), "scripts/build_dataset.py", "--only-manifest"]
    if args.organism != "all":
        cmd.extend(["--organism", args.organism])
    _run(cmd, "splits (manifest scan + split JSONs)")


def step_train(args: argparse.Namespace) -> None:
    profiles = PROFILES if args.profile == "all" else (args.profile,)
    script_map = {
        "hg38": "scripts/train_hg38.py",
        "mm10": "scripts/train_mm10.py",
        "multiorganism": "scripts/train_multiorganism.py",
    }
    for profile in profiles:
        ckpt = _checkpoint(profile)
        if args.skip_if_exists and ckpt.exists():
            print(f"Skip train {profile}: checkpoint exists at {ckpt}")
            continue
        _run([_python(), script_map[profile]], f"train {profile}")


def step_test(args: argparse.Namespace) -> None:
    profiles = PROFILES if args.profile == "all" else (args.profile,)
    for profile in profiles:
        ckpt = args.checkpoint or _checkpoint(profile)
        if not ckpt.exists():
            print(f"Skip test {profile}: missing checkpoint {ckpt}")
            continue
        cmd = [
            _python(), "scripts/test.py",
            "--organism", profile,
            "--checkpoint", str(ckpt),
            "--split", args.split,
        ]
        _run(cmd, f"test {profile}")


def step_analyze(args: argparse.Namespace) -> None:
    results_dir = ROOT / "data" / "output" / "results"
    if not results_dir.exists():
        raise SystemExit(f"No results dir: {results_dir}")

    profiles = PROFILES if args.profile == "all" else (args.profile,)
    found = False
    for profile in profiles:
        for pred_csv in sorted(results_dir.glob(f"{profile}_*_predictions.csv")):
            found = True
            metrics_csv = pred_csv.with_name(pred_csv.name.replace("_predictions.csv", "_metrics.csv"))
            cmd = [_python(), "scripts/analyze_results.py", "--predictions", str(pred_csv)]
            if metrics_csv.exists():
                cmd.extend(["--metrics", str(metrics_csv)])
            _run(cmd, f"analyze {pred_csv.name}")

    if not found:
        print(f"No prediction CSVs found in {results_dir} for profile(s) {profiles}")


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--organism", choices=["hg38", "mm10", "all"], default="all")
    parser.add_argument("--chroms", default=None, help="Comma-separated chroms, e.g. chr21,chr22")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multimodal Hi-C → TPM pipeline steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="Convert text Hi-C to memmap cache")
    _add_common_data_args(p_convert)
    p_convert.add_argument("--workers", type=int, default=1)
    p_convert.set_defaults(func=step_convert)

    p_build = sub.add_parser("build", help="EP regions + NPZ + manifest + splits")
    p_build.add_argument("--skip-ep", action="store_true", help="Skip EP JSON generation")
    p_build.add_argument("--resume", action="store_true", help="Skip existing NPZ shards")
    p_build.add_argument("--require-memmap", action="store_true", help="Require memmap cache")
    _add_common_data_args(p_build)
    p_build.set_defaults(func=step_build)

    p_splits = sub.add_parser("splits", help="Rescan NPZ → manifest + splits only (fast)")
    _add_common_data_args(p_splits)
    p_splits.set_defaults(func=step_splits)

    p_train = sub.add_parser("train", help="Train model(s)")
    p_train.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    p_train.add_argument("--skip-if-exists", action="store_true",
                         help="Skip profile if best checkpoint already exists")
    p_train.set_defaults(func=step_train)

    p_test = sub.add_parser("test", help="Evaluate checkpoint(s) on split modes")
    p_test.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    p_test.add_argument("--checkpoint", type=Path, default=None,
                        help="Override checkpoint path (single-profile runs only)")
    p_test.add_argument("--split", default="all", help="Test mode or 'all'")
    p_test.set_defaults(func=step_test)

    p_analyze = sub.add_parser("analyze", help="Plot latest test prediction CSVs")
    p_analyze.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    p_analyze.set_defaults(func=step_analyze)

    p_all = sub.add_parser("all", help="Run convert → build → train → test")
    p_all.add_argument("--convert", action="store_true", help="Run memmap convert before build")
    p_all.add_argument("--build", action="store_true", help="Alias for running build step (default on)")
    p_all.add_argument("--skip-ep", action="store_true", help="Pass --skip-ep to build step")
    p_all.add_argument("--resume", action="store_true", help="Pass --resume to build step")
    p_all.add_argument("--require-memmap", action="store_true", help="Pass --require-memmap to build")
    p_all.add_argument("--workers", type=int, default=1, help="Convert parallel workers")
    _add_common_data_args(p_all)
    p_all.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    p_all.add_argument("--skip-train-if-exists", action="store_true")
    p_all.add_argument("--split", default="all")
    p_all.add_argument("--analyze", action="store_true", help="Run analyze after test")
    p_all.set_defaults(func=lambda args: None)
    return parser


def step_all(args: argparse.Namespace) -> None:
    if args.convert:
        step_convert(args)
    step_build(args)
    args.skip_if_exists = args.skip_train_if_exists
    step_train(args)
    step_test(args)
    if args.analyze:
        step_analyze(args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "all":
        step_all(args)
    else:
        args.func(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
