#!/usr/bin/env python3
"""
End-to-end single-agent reproduction demo for VSA-OGM.

Runs the single-agent experiments in a single process and compares the
results against the published tables:

  * ToySim    -> Table 4 (VSA-OGM vs. BHM-Full and BHM-Diag)
  * Intel Map -> Table 6 (VSA-OGM vs. OGM, I-GPOM, BHM, Fast-BHM, SBKM)

Both experiments use a fixed seed for the hyperdimensional basis and for the
per-frame train/test splits, so repeated runs are bit-for-bit reproducible.

Usage:
    python drivers/scripts/demo_single_agent.py
    python drivers/scripts/demo_single_agent.py --dataset intel --save-figures
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

from demo_common import (  # noqa: E402  (sets up sys.path on import)
    REPO_ROOT,
    decode_to_ogm,
    encode_points,
    evaluate,
    load_config,
    memory_size_mb,
    save_figure,
    select_device,
    splits_from_dataset,
    write_results,
)

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tabulate import tabulate  # noqa: E402

from vsa_ogm.data import load_data  # noqa: E402
from vsa_ogm.mappers.sa.sa_vsa_mapper import SA_VSA_OGM  # noqa: E402


# Published VSA-OGM rows, for the delta columns.
#   Snyder et al., "Brain Inspired Probabilistic Occupancy Grid Mapping with
#   Hyperdimensional Computing", Tables 4 and 6.
EXPERIMENTS: Dict[str, dict] = {
    "toysim": {
        "label": "ToySim",
        "config": "configs/experiments/exp3_toymap_vsa_ogm_cuda_general.yaml",
        "paper_table": "Table 4",
        "paper": {"auc": 0.93, "nll": 0.33, "cpu_s": 0.04,
                  "gpu_s": 0.005, "size_mb": 1.02},
        # (method, AUC, NLL, CPU latency, GPU latency, model size)
        "baselines": [
            ("BHM - Full", "0.96", "0.44", "6.5 s", "--", "6400 MB"),
            ("BHM - Diag", "0.97", "0.38", "0.45 s", "0.024 s", "0.04 MB"),
        ],
    },
    "intel": {
        "label": "Intel Map",
        "config": "configs/experiments/exp3_intel_vsa_ogm_cuda_general.yaml",
        "paper_table": "Table 6",
        "paper": {"auc": 0.95, "nll": 0.25, "cpu_s": 0.51,
                  "gpu_s": 0.04, "size_mb": 16.3},
        "baselines": [
            ("OGM", "0.93", "--", "1.12 s", "--", "--"),
            ("I-GPOM", "0.94", "--", "15.35 s", "--", "--"),
            ("I-GPOM2", "0.97", "--", "17.17 s", "--", "--"),
            ("BHM - Full (0.1)", "0.96", "--", "22.35 s", "--", "6400 MB"),
            ("BHM - Diag (0.1)", "0.96", "0.25", "8.11 s", "--", "0.04 MB"),
            ("Fast-BHM", "0.95", "--", "0.46 s", "--", "0.04 MB"),
            ("SBKM", "0.96", "0.36", "0.76 s", "--", "--"),
        ],
    },
}


def run_experiment(name: str, device: str, seed: int, args) -> dict:
    """Run one single-agent experiment end to end and return its metrics."""
    spec = EXPERIMENTS[name]
    config = load_config(spec["config"], device, seed)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    print(f"-- {spec['label']} " + "-" * (62 - len(spec["label"])))
    print(f"  config          : {spec['config']}")
    print(f"  world bounds    : {list(config.data.world_bounds)}")
    print(f"  axis resolution : {config.mapping.axis_resolution} m")
    print(f"  tiles per axis  : {config.mapping.num_tiles}")
    print(f"  length scale    : {config.mapping.vector_length_scale}")
    print(f"  disk radii      : {config.mapping.decoding.disk_radii_1}, "
          f"{config.mapping.decoding.disk_radii_2}")

    dataset = load_data(config, None)
    split = splits_from_dataset(dataset, config.data.test_split, seed)
    print(f"  scans           : {split['num_frames']}")
    print(f"  train / test    : {len(split['y_train']):,} / "
          f"{len(split['y_test']):,} points "
          f"({split['y_train'].mean():.3f} occupied)")

    mapper = SA_VSA_OGM(config, None)

    encode_start = time.perf_counter()
    occupied, empty = encode_points(mapper, split["X_train"], split["y_train"])
    encode_time = time.perf_counter() - encode_start

    _, decode_latency = decode_to_ogm(mapper, occupied, empty)
    train_auc, train_nll = evaluate(mapper, split["X_train"], split["y_train"])
    test_auc, test_nll = evaluate(mapper, split["X_test"], split["y_test"])
    size_mb = memory_size_mb(occupied, empty)

    per_scan_ms = 1000.0 * encode_time / split["num_frames"]
    print(f"  encode          : {encode_time:.1f}s total "
          f"({per_scan_ms:.1f} ms/scan)")
    print(f"  decode          : {decode_latency:.3f}s")
    print(f"  model size      : {size_mb:.2f} MB  "
          f"(grid {mapper.ogm.shape[0]}x{mapper.ogm.shape[1]})")
    print(f"  train AUC/NLL   : {train_auc:.3f} / {train_nll:.3f}")
    print(f"  test  AUC/NLL   : {test_auc:.3f} / {test_nll:.3f}")
    print()

    if args.save_figures:
        output_dir = os.path.join(REPO_ROOT, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        save_figure(mapper.ogm, f"{spec['label']} OGM (single agent)",
                    os.path.join(output_dir, f"{name}_ogm.png"))
        np.save(os.path.join(output_dir, f"{name}_ogm.npy"), mapper.ogm)

    return {
        "name": name,
        "label": spec["label"],
        "num_frames": split["num_frames"],
        "train_points": int(len(split["y_train"])),
        "test_points": int(len(split["y_test"])),
        "train_auc": train_auc,
        "train_nll": train_nll,
        "test_auc": test_auc,
        "test_nll": test_nll,
        "encode_s": encode_time,
        "per_scan_ms": per_scan_ms,
        "decode_s": decode_latency,
        "size_mb": size_mb,
        "grid": list(mapper.ogm.shape),
        "config": {
            "axis_resolution": config.mapping.axis_resolution,
            "num_tiles": config.mapping.num_tiles,
            "vector_dimensionality": config.mapping.vector_dimensionality,
            "vector_length_scale": config.mapping.vector_length_scale,
            "decoding_method": config.mapping.decoding.method,
            "disk_radii": [config.mapping.decoding.disk_radii_1,
                           config.mapping.decoding.disk_radii_2],
            "world_bounds": list(config.data.world_bounds),
            "test_split": config.data.test_split,
        },
    }


def print_comparison(result: dict, device: str) -> None:
    """Print one experiment's results beside the published table."""
    spec = EXPERIMENTS[result["name"]]
    paper = spec["paper"]

    print("=" * 74)
    print(f"{spec['paper_table']} -- {spec['label']} single-agent "
          f"vs. published results")
    print("=" * 74)

    rows = [[
        "VSA-OGM (this run)",
        f"{result['test_auc']:.3f}",
        f"{result['test_nll']:.3f}",
        f"{result['decode_s']:.3f} s ({device})",
        f"{result['size_mb']:.2f} MB",
    ], [
        "VSA-OGM (paper)",
        f"{paper['auc']:.2f}",
        f"{paper['nll']:.2f}",
        f"{paper['cpu_s']} s (CPU) / {paper['gpu_s']} s (GPU)",
        f"{paper['size_mb']} MB",
    ]]
    for method, auc, nll, cpu, gpu, size in spec["baselines"]:
        latency = cpu if gpu == "--" else f"{cpu} (CPU) / {gpu} (GPU)"
        rows.append([method, auc, nll, latency, size])

    print(tabulate(
        rows,
        headers=["Method", "AUC", "NLL", "Latency", "Model size"],
        tablefmt="simple_outline",
    ))
    print(f"  AUC delta vs. paper: {result['test_auc'] - paper['auc']:+.3f}   "
          f"NLL delta: {result['test_nll'] - paper['nll']:+.3f}")
    print("  Baseline rows are quoted from the paper, not re-run here.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all",
                        choices=["all", "toysim", "intel"],
                        help="which single-agent experiment to run")
    parser.add_argument("--device", default="auto",
                        help="torch device: auto (default), cpu, cuda, mps")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the VSA basis and the data splits")
    parser.add_argument("--save-figures", action="store_true",
                        help="also write OGM images and .npy grids")
    parser.add_argument("--output-dir", default="outputs/single_agent_demo",
                        help="directory for results and figures")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="config overrides in dotlist form, e.g. "
                             "--set mapping.axis_resolution=0.2")
    args = parser.parse_args()

    device = select_device(args.device)
    names = list(EXPERIMENTS) if args.dataset == "all" else [args.dataset]

    print("=" * 74)
    print("VSA-OGM :: Single-Agent Reproduction Demo")
    print("=" * 74)
    print(f"  device : {device}")
    print(f"  seed   : {args.seed}")
    print()

    results = [run_experiment(name, device, args.seed, args) for name in names]

    for result in results:
        print_comparison(result, device)

    output_dir = os.path.join(REPO_ROOT, args.output_dir)
    csv_fields = ["dataset", "scans", "train_points", "test_points",
                  "train_auc", "train_nll", "test_auc", "test_nll",
                  "auc_paper", "nll_paper", "auc_delta", "nll_delta",
                  "decode_latency_s", "per_scan_encode_ms", "model_size_mb",
                  "model_size_mb_paper"]
    rows = []
    for result in results:
        paper = EXPERIMENTS[result["name"]]["paper"]
        rows.append({
            "dataset": result["label"],
            "scans": result["num_frames"],
            "train_points": result["train_points"],
            "test_points": result["test_points"],
            "train_auc": f"{result['train_auc']:.6f}",
            "train_nll": f"{result['train_nll']:.6f}",
            "test_auc": f"{result['test_auc']:.6f}",
            "test_nll": f"{result['test_nll']:.6f}",
            "auc_paper": paper["auc"],
            "nll_paper": paper["nll"],
            "auc_delta": f"{result['test_auc'] - paper['auc']:+.6f}",
            "nll_delta": f"{result['test_nll'] - paper['nll']:+.6f}",
            "decode_latency_s": f"{result['decode_s']:.6f}",
            "per_scan_encode_ms": f"{result['per_scan_ms']:.3f}",
            "model_size_mb": f"{result['size_mb']:.3f}",
            "model_size_mb_paper": paper["size_mb"],
        })

    csv_path, json_path = write_results(
        output_dir,
        rows,
        {
            "run": {"device": device, "seed": args.seed},
            "measured": results,
            "paper": {n: EXPERIMENTS[n]["paper"] for n in names},
            "paper_baselines": {n: EXPERIMENTS[n]["baselines"] for n in names},
        },
        csv_fields,
    )

    print(f"Saved to {args.output_dir}/")
    print(f"  {os.path.basename(csv_path):<20} per-dataset metrics vs. paper")
    print(f"  {os.path.basename(json_path):<20} the above plus config and provenance")
    if args.save_figures:
        print(f"  *_ogm.png / *_ogm.npy   maps for each dataset")
        print("NOTE: .gitignore excludes **/*.png, so figures will not be committed.")


if __name__ == "__main__":
    main()
