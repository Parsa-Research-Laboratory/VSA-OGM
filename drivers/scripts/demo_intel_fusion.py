#!/usr/bin/env python3
"""
End-to-end multi-agent Intel Map fusion demo for VSA-OGM.

Runs the complete pipeline in a single process:

  1. loads the four Intel Map quadrant datasets (one per agent),
  2. builds per-frame stratified train/test splits,
  3. encodes each agent's training points into its own tile memory vectors,
  4. fuses the four agents by summing their memory vectors,
  5. decodes the fused memories into a global OGM,
  6. prints AUC / NLL / latency tables alongside the published results
     (Table 7 and Table 8).

All four agents share a single hyperdimensional basis (driven by
`mapping.seed`). This is required for fusion to be meaningful -- memory
vectors encoded against different bases live in different algebraic spaces
and cannot be summed. See `SSPGenerator` in vsa_ogm/mappers/sa/sa_vsa_mapper.py.

Usage:
    python drivers/scripts/demo_intel_fusion.py
    python drivers/scripts/demo_intel_fusion.py --device cpu --save-figures
"""

import argparse
import os
import pickle
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
    split_frame,
    write_results,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tabulate import tabulate  # noqa: E402

from vsa_ogm.mappers.sa.sa_vsa_mapper import SA_VSA_OGM  # noqa: E402

AGENT_QUADRANTS: Tuple[int, ...] = (1, 2, 3, 4)
BASE_CONFIG = "configs/experiments/exp4_intel_vsa_ogm_cuda_general_agent1.yaml"
DATASET_TEMPLATE = "datasets/ma/intel-quadrant{q}.pkl"

# Published results, for side-by-side comparison.
#   Snyder et al., "Brain Inspired Probabilistic Occupancy Grid Mapping with
#   Hyperdimensional Computing", Table 7 (per-agent + fusion) and Table 8
#   (fusion vs. baselines).
PAPER_TABLE_7: Dict[str, Tuple[float, float, float]] = {
    # label:      (AUC,  NLL,  latency_s)
    "Agent 1": (0.95, 0.45, 0.04),
    "Agent 2": (0.96, 0.39, 0.04),
    "Agent 3": (0.94, 0.42, 0.04),
    "Agent 4": (0.93, 0.37, 0.04),
    "Fusion": (0.95, 0.37, 0.03),
}
PAPER_TABLE_8: List[Tuple[str, str, str, str]] = [
    # metric,        VSA-OGM,   Fast-BHM,  OHM
    ("AUC", "0.95", "0.936", "0.92"),
    ("Model size", "16.4 MB", "0.04 MB", "--"),
    ("Covariance", "True", "False", "False"),
]


def load_agent_splits(quadrant: int, test_split: float,
                      seed: int) -> Dict[str, np.ndarray]:
    """Load one quadrant and build a per-frame stratified train/test split."""
    path = os.path.join(REPO_ROOT, DATASET_TEMPLATE.format(q=quadrant))
    with open(path, "rb") as f:
        frames: List[np.ndarray] = pickle.load(f)

    X_train, y_train, X_test, y_test = [], [], [], []
    for frame in frames:
        a, b, c, d = split_frame(frame[:, :2], frame[:, 2], test_split, seed)
        X_train.append(a)
        X_test.append(b)
        y_train.append(c)
        y_test.append(d)

    return {
        "num_frames": len(frames),
        "X_train": np.vstack(X_train),
        "y_train": np.concatenate(y_train),
        "X_test": np.vstack(X_test),
        "y_test": np.concatenate(y_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto",
                        help="torch device: auto (default), cpu, cuda, mps")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the shared VSA basis and the data splits")
    parser.add_argument("--save-figures", action="store_true",
                        help="also write per-agent and fused OGM images")
    parser.add_argument("--save-memories", action="store_true",
                        help="also write the fused tile memory vectors (~16 MB)")
    parser.add_argument("--output-dir", default="outputs/intel_fusion_demo",
                        help="directory for results and figures")
    args = parser.parse_args()

    device = select_device(args.device)
    config = load_config(BASE_CONFIG, device, args.seed)

    # Results are always persisted; figures and memories are opt-in.
    output_dir = os.path.join(REPO_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 74)
    print("VSA-OGM :: Multi-Agent Intel Map Fusion Demo")
    print("=" * 74)
    print(f"  device               : {device}")
    print(f"  seed (shared basis)  : {args.seed}")
    print(f"  vector dimensionality: {config.mapping.vector_dimensionality}")
    print(f"  tiles per axis       : {config.mapping.num_tiles}")
    print(f"  axis resolution      : {config.mapping.axis_resolution} m")
    print(f"  world bounds         : {list(config.data.world_bounds)}")
    print(f"  decoding method      : {config.mapping.decoding.method}")
    print()

    # ---------------------------------------------------------------- splits
    print("-- Loading data and building per-frame stratified splits ---------")
    agents = {}
    split_rows = []
    for quadrant in AGENT_QUADRANTS:
        split = load_agent_splits(quadrant, config.data.test_split, args.seed)
        agents[quadrant] = split
        split_rows.append([
            f"Agent {quadrant}",
            split["num_frames"],
            f"{len(split['y_train']):,}",
            f"{len(split['y_test']):,}",
            f"{split['y_train'].mean():.3f}",
        ])
    total_train = sum(len(a["y_train"]) for a in agents.values())
    total_test = sum(len(a["y_test"]) for a in agents.values())
    split_rows.append(["TOTAL", sum(a["num_frames"] for a in agents.values()),
                       f"{total_train:,}", f"{total_test:,}", ""])
    print(tabulate(
        split_rows,
        headers=["Agent", "Frames", "Train pts", "Test pts", "Occupied frac"],
        tablefmt="simple_outline",
    ))
    print()

    # ------------------------------------------------------ per-agent mapping
    print("-- Encoding each agent (VSA-OGM learning) ------------------------")
    global_mapper = SA_VSA_OGM(config, None)
    agent_memories = []
    results = {}

    for quadrant in AGENT_QUADRANTS:
        split = agents[quadrant]
        # Every agent uses the same seed, so every agent builds the same
        # axis vectors -- that is what makes the memories fusable.
        mapper = global_mapper if quadrant == 1 else SA_VSA_OGM(config, None)
        assert torch.equal(mapper.xy_axis_vectors, global_mapper.xy_axis_vectors), \
            "agents built different bases; fusion would be meaningless"

        encode_start = time.perf_counter()
        occupied, empty = encode_points(mapper, split["X_train"], split["y_train"])
        encode_time = time.perf_counter() - encode_start
        agent_memories.append((occupied, empty))

        _, latency = decode_to_ogm(mapper, occupied, empty)
        auc, nll = evaluate(mapper, split["X_test"], split["y_test"])
        results[f"Agent {quadrant}"] = (auc, nll, latency)

        print(f"  Agent {quadrant}: encode {encode_time:6.1f}s | "
              f"decode {latency:5.3f}s | AUC {auc:.3f} | NLL {nll:.3f}")

        if args.save_figures:
            save_figure(mapper.ogm, f"Agent {quadrant} OGM",
                        os.path.join(output_dir, f"agent{quadrant}_ogm.png"))
    print()

    # ------------------------------------------------------------- fusion
    print("-- Fusing agent memories into a global map -----------------------")
    fused_occupied = torch.stack([m[0] for m in agent_memories]).sum(dim=0)
    fused_empty = torch.stack([m[1] for m in agent_memories]).sum(dim=0)
    _, fusion_latency = decode_to_ogm(global_mapper, fused_occupied, fused_empty)

    pooled_X_test = np.vstack([agents[q]["X_test"] for q in AGENT_QUADRANTS])
    pooled_y_test = np.concatenate([agents[q]["y_test"] for q in AGENT_QUADRANTS])
    fusion_auc, fusion_nll = evaluate(global_mapper, pooled_X_test, pooled_y_test)
    results["Fusion"] = (fusion_auc, fusion_nll, fusion_latency)
    fused_size = memory_size_mb(fused_occupied, fused_empty)

    print(f"  Fused {len(AGENT_QUADRANTS)} agents -> "
          f"{tuple(fused_occupied.shape)} occupied + empty memories "
          f"({fused_size:.1f} MB)")
    print(f"  decode {fusion_latency:5.3f}s | AUC {fusion_auc:.3f} | "
          f"NLL {fusion_nll:.3f}")
    print()

    if args.save_figures:
        save_figure(global_mapper.ogm, "Fused Global OGM",
                    os.path.join(output_dir, "fused_ogm.png"))
    if args.save_memories:
        np.savez_compressed(
            os.path.join(output_dir, "fused_memories.npz"),
            occupied=fused_occupied.cpu().numpy(),
            empty=fused_empty.cpu().numpy(),
        )
    np.save(os.path.join(output_dir, "fused_ogm.npy"), global_mapper.ogm)

    # -------------------------------------------------------------- tables
    print("=" * 74)
    print("Table 7 -- per-agent and fused performance vs. published results")
    print("=" * 74)
    rows = []
    for label in list(PAPER_TABLE_7):
        auc, nll, latency = results[label]
        paper_auc, paper_nll, paper_latency = PAPER_TABLE_7[label]
        rows.append([
            label,
            f"{auc:.3f}", f"{paper_auc:.2f}", f"{auc - paper_auc:+.3f}",
            f"{nll:.3f}", f"{paper_nll:.2f}", f"{nll - paper_nll:+.3f}",
            f"{latency:.3f}", f"{paper_latency:.2f}",
        ])
    print(tabulate(
        rows,
        headers=["", "AUC", "paper", "delta",
                 "NLL", "paper", "delta",
                 f"Lat.(s) {device}", "paper (GPU)"],
        tablefmt="simple_outline",
    ))
    print("  Latency is the decode/inference step only. The published figures")
    print("  were measured on a CUDA GPU, so compare them with care.")
    print()

    print("=" * 74)
    print("Table 8 -- fused VSA-OGM vs. baselines (baselines quoted from paper)")
    print("=" * 74)
    measured = {
        "AUC": f"{fusion_auc:.3f}",
        "Model size": f"{fused_size:.1f} MB",
        "Covariance": "True",
    }
    print(tabulate(
        [[metric, measured[metric], paper_vsa, fast_bhm, ohm]
         for metric, paper_vsa, fast_bhm, ohm in PAPER_TABLE_8],
        headers=["Metric", "VSA-OGM (this run)", "VSA-OGM (paper)",
                 "Fast-BHM", "OHM"],
        tablefmt="simple_outline",
    ))
    print()

    worst = max(abs(results[k][0] - PAPER_TABLE_7[k][0]) for k in PAPER_TABLE_7)
    print(f"Largest AUC deviation from the published table: {worst:.3f}")
    print()

    # ------------------------------------------------------------ persist
    csv_fields = ["agent", "frames", "train_points", "test_points",
                  "auc", "auc_paper", "nll", "nll_paper",
                  "decode_latency_s", "decode_latency_s_paper_gpu"]
    csv_rows = []
    for quadrant in AGENT_QUADRANTS:
        label = f"Agent {quadrant}"
        auc, nll, latency = results[label]
        paper = PAPER_TABLE_7[label]
        split = agents[quadrant]
        csv_rows.append(dict(zip(csv_fields, [
            label, split["num_frames"],
            len(split["y_train"]), len(split["y_test"]),
            f"{auc:.6f}", paper[0], f"{nll:.6f}", paper[1],
            f"{latency:.6f}", paper[2]])))
    paper = PAPER_TABLE_7["Fusion"]
    csv_rows.append(dict(zip(csv_fields, [
        "Fusion", sum(a["num_frames"] for a in agents.values()),
        total_train, total_test,
        f"{fusion_auc:.6f}", paper[0], f"{fusion_nll:.6f}", paper[1],
        f"{fusion_latency:.6f}", paper[2]])))

    csv_path, json_path = write_results(
        output_dir,
        csv_rows,
        {
            "run": {
                "device": device,
                "seed": args.seed,
                "vector_dimensionality": config.mapping.vector_dimensionality,
                "num_tiles": config.mapping.num_tiles,
                "axis_resolution": config.mapping.axis_resolution,
                "world_bounds": list(config.data.world_bounds),
                "decoding_method": config.mapping.decoding.method,
                "test_split": config.data.test_split,
            },
            "splits": {
                f"agent_{q}": {
                    "frames": agents[q]["num_frames"],
                    "train_points": int(len(agents[q]["y_train"])),
                    "test_points": int(len(agents[q]["y_test"])),
                    "occupied_fraction": float(agents[q]["y_train"].mean()),
                } for q in AGENT_QUADRANTS
            },
            "measured": {
                label: {"auc": auc, "nll": nll, "decode_latency_s": latency}
                for label, (auc, nll, latency) in results.items()
            },
            "paper_table_7": {
                label: {"auc": v[0], "nll": v[1], "latency_s": v[2]}
                for label, v in PAPER_TABLE_7.items()
            },
            "paper_table_8": [
                {"metric": m, "vsa_ogm": v, "fast_bhm": b, "ohm": o}
                for m, v, b, o in PAPER_TABLE_8
            ],
            "fused_model_size_mb": fused_size,
            "largest_auc_deviation": worst,
        },
        csv_fields,
    )

    print(f"Saved to {args.output_dir}/")
    print(f"  {os.path.basename(csv_path):<20} per-agent + fusion metrics vs. paper")
    print(f"  {os.path.basename(json_path):<20} the above plus config and provenance")
    print(f"  fused_ogm.npy        the fused {global_mapper.ogm.shape} occupancy grid")
    if args.save_figures:
        print(f"  *_ogm.png            per-agent and fused map images")
    if args.save_memories:
        print(f"  fused_memories.npz   fused tile memory vectors")
    if args.save_figures:
        print("NOTE: .gitignore excludes **/*.png, so figures will not be committed.")


if __name__ == "__main__":
    main()
