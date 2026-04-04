import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path


def parse_metrics_from_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8")

    def find_float(pattern: str):
        m = re.search(pattern, text)
        if not m:
            return None
        return float(m.group(1))

    def find_int(pattern: str):
        m = re.search(pattern, text)
        if not m:
            return None
        return int(m.group(1))

    return {
        "best_epoch": find_int(r"best_epoch=(\d+)"),
        "best_val_macro_f1": find_float(r"best_val_macro_f1=([0-9]*\.?[0-9]+)"),
        "test_accuracy": find_float(r"accuracy\s*:\s*([0-9]*\.?[0-9]+)"),
        "test_macro_f1": find_float(r"macro_f1\s*:\s*([0-9]*\.?[0-9]+)"),
    }


def run_one_seed(python_exe: str, repo_dir: Path, args, seed: int):
    cmd = [
        python_exe,
        "train_gat_classification.py",
        "--seed", str(seed),
        "--num-epochs", str(args.num_epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--graph-k", str(args.graph_k),
        "--lr", str(args.lr),
        "--scheduler", args.scheduler,
        "--plateau-factor", str(args.plateau_factor),
        "--plateau-patience", str(args.plateau_patience),
        "--num-sample-infer", "2",
        "--no-tqdm",
    ]

    print(f"\n=== Running seed {seed} ===")
    result = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    output = result.stdout
    if result.returncode != 0:
        print(output)
        raise RuntimeError(f"Run failed for seed {seed} with code {result.returncode}")

    m = re.search(r"log_file\s*:\s*(.+)", output)
    if not m:
        raise RuntimeError(f"Could not find log path for seed {seed}")

    log_path = Path(m.group(1).strip())
    metrics = parse_metrics_from_log(log_path)
    metrics["seed"] = seed
    metrics["log_path"] = str(log_path)

    print(
        f"seed={seed} best_val_macro_f1={metrics['best_val_macro_f1']:.6f} "
        f"test_macro_f1={metrics['test_macro_f1']:.6f} test_acc={metrics['test_accuracy']:.6f}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run train_gat_classification.py for multiple seeds and summarize.")
    parser.add_argument("--seeds", type=str, default="11,22,33")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=4)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("No seeds provided")

    results = []
    for seed in seeds:
        results.append(run_one_seed(python_exe, repo_dir, args, seed))

    best = max(results, key=lambda r: r["test_macro_f1"])
    macro_vals = [r["test_macro_f1"] for r in results]
    acc_vals = [r["test_accuracy"] for r in results]

    print("\n=== Multi-seed summary ===")
    print(f"seeds={seeds}")
    print(f"test_macro_f1 mean={statistics.mean(macro_vals):.6f} std={statistics.pstdev(macro_vals):.6f}")
    print(f"test_accuracy mean={statistics.mean(acc_vals):.6f} std={statistics.pstdev(acc_vals):.6f}")
    print(
        f"best_seed={best['seed']} best_test_macro_f1={best['test_macro_f1']:.6f} "
        f"best_test_accuracy={best['test_accuracy']:.6f} log={best['log_path']}"
    )


if __name__ == "__main__":
    main()
