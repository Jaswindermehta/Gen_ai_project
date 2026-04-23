"""
solver_assignment.py
--------------------
Offline one-time script that:
  1. Runs a full (no-cache) probe pass through SD U-Net
  2. Hooks into down_blocks[0] — the EXACT layer DeepCache caches
     (cache_layer_id=0, cache_block_id=0) → shape [B, 320, 64, 64]
  3. Clusters the 320 feature dimensions by temporal dynamics
  4. Assigns best solver per cluster → saves solver_assignments.json
     and cluster_labels_320.json

This replaces the old mid_block (1280-ch) approach which never matched
the actual cached features DeepCache uses.

Usage:
    python solver_assignment.py \
        --prompt "a photo of an astronaut on a moon" \
        --output solver_assignments.json \
        --cluster_output cluster_labels_320.json
"""

import argparse
import json
import logging

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from diffusers import StableDiffusionPipeline

from solvers import SOLVERS, SOLVER_NAMES, SOLVER_MIN_HISTORY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)


# ──────────────────────────────────────────────
# Step 1 — Probe: hook into down_blocks[0]
# ──────────────────────────────────────────────

def run_probe(model_id, prompt, seed=42, num_steps=50, device="cuda:0"):
    """
    Capture down_blocks[0] output at every denoising timestep.
    This is exactly what DeepCache caches with cache_layer_id=0, cache_block_id=0.
    Shape: [B, 320, 64, 64] for SD v1.5 at 512x512.
    """
    logging.info("Loading model: %s", model_id)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)

    trajectory = []

    def _hook(module, input, output):
        # down_blocks output: (hidden_states, res_hidden_states_tuple)
        feat = output[0] if isinstance(output, tuple) else output
        trajectory.append(feat.detach().float().cpu())

    # Hook into down_blocks[0] — same layer DeepCache caches
    hook = pipe.unet.down_blocks[0].register_forward_hook(_hook)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logging.info("Running probe (%d steps) on down_blocks[0]...", num_steps)
    with torch.no_grad():
        pipe(prompt, num_inference_steps=num_steps,
             output_type="pt", guidance_scale=7.5)

    hook.remove()
    del pipe
    torch.cuda.empty_cache()

    logging.info(
        "Probe done — %d tensors, shape: %s",
        len(trajectory),
        trajectory[0].shape if trajectory else "N/A",
    )
    return trajectory  # list of T tensors [B, C, H, W]


# ──────────────────────────────────────────────
# Step 2 — Cluster: group dimensions by dynamics
# ──────────────────────────────────────────────

def compute_indicators(trajectory):
    """
    For each channel, compute temporal dynamic indicators → [C, 6].
    """
    stacked = torch.stack(trajectory)       # [T, B, C, H, W]
    series  = stacked.mean(dim=[1, 3, 4])   # [T, C]
    T, C    = series.shape

    logging.info("Computing indicators: %d channels x %d timesteps", C, T)

    indicators = []
    for d in range(C):
        x = series[:, d]

        d1    = torch.abs(x[1:] - x[:-1]).mean().item()
        d2    = torch.abs(x[2:] - 2*x[1:-1] + x[:-2]).mean().item() if T > 2 else 0.0
        d3    = torch.abs(x[3:] - 3*x[2:-1] + 3*x[1:-2] - x[:-3]).mean().item() if T > 3 else 0.0
        eta   = d2 / (d1 + 1e-8)
        kappa = d3 / (d2 + 1e-8)
        energy = x.abs().mean().item()

        indicators.append([d1, d2, d3, eta, kappa, energy])

    return np.array(indicators, dtype=np.float32)  # [C, 6]


def cluster_dimensions(indicators, default_k=7):
    """K-means on indicator descriptors with elbow selection."""
    scaler = StandardScaler()
    X = scaler.fit_transform(indicators)

    # Elbow method
    k_range     = list(range(2, min(9, len(indicators))))
    distortions = []
    for ki in k_range:
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X)
        distortions.append(km.inertia_)
        logging.info("  k=%d  inertia=%.2f", ki, km.inertia_)

    if len(distortions) >= 3:
        diffs  = np.diff(distortions)
        diffs2 = np.diff(diffs)
        best_k = int(k_range[np.argmax(diffs2) + 1])
    else:
        best_k = default_k

    logging.info("Chosen k = %d", best_k)

    km     = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    for cid in np.unique(labels):
        logging.info("  Cluster %d: %d channels", cid, (labels == cid).sum())

    return labels.astype(np.int32)


# ──────────────────────────────────────────────
# Step 3 — Assign: best solver per cluster
# ──────────────────────────────────────────────

def assign_solvers(trajectory, cluster_labels):
    """Test all 5 solvers per cluster on the trajectory."""
    T               = len(trajectory)
    unique_clusters = np.unique(cluster_labels)
    assignments     = {}

    logging.info("Assigning solvers for %d clusters over %d steps...",
                 len(unique_clusters), T)

    for cid in unique_clusters:
        mask   = cluster_labels == cid
        n_ch   = int(mask.sum())
        errors = {name: 0.0 for name in SOLVER_NAMES}
        count  = 0

        for t in range(3, T):
            actual = trajectory[t][:, mask]
            for name in SOLVER_NAMES:
                min_h = SOLVER_MIN_HISTORY[name]
                hist  = [trajectory[t - min_h + i][:, mask] for i in range(min_h)]
                pred  = SOLVERS[name](hist)
                errors[name] += torch.mean((pred - actual) ** 2).item()
            count += 1

        mean_errors = {k: v / max(count, 1) for k, v in errors.items()}
        best        = min(mean_errors, key=mean_errors.get)
        assignments[str(int(cid))] = best

        logging.info(
            "  Cluster %2d (%3d ch) -> %-5s | %s",
            cid, n_ch, best,
            {k: f"{v:.4f}" for k, v in mean_errors.items()}
        )

    return assignments


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt",         type=str, default="a photo of an astronaut on a moon")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--steps",          type=int, default=50)
    parser.add_argument("--device",         type=str, default="cuda:0")
    parser.add_argument("--output",         type=str, default="solver_assignments.json")
    parser.add_argument("--cluster_output", type=str, default="cluster_labels_320.json")
    args = parser.parse_args()

    # 1. Probe the correct layer
    trajectory = run_probe(
        model_id=args.model,
        prompt=args.prompt,
        seed=args.seed,
        num_steps=args.steps,
        device=args.device,
    )

    C = trajectory[0].shape[1]
    logging.info("Captured feature channels: %d", C)

    # 2. Compute indicators and cluster
    indicators     = compute_indicators(trajectory)
    cluster_labels = cluster_dimensions(indicators)

    # Save cluster labels
    with open(args.cluster_output, "w") as f:
        json.dump({
            "num_dimensions": int(C),
            "labels": cluster_labels.tolist(),
            "note": f"down_blocks[0] features, shape [B,{C},H,W]"
        }, f, indent=2)
    logging.info("Cluster labels saved -> %s", args.cluster_output)

    # 3. Assign solvers
    assignments = assign_solvers(trajectory, cluster_labels)

    with open(args.output, "w") as f:
        json.dump(assignments, f, indent=2)
    logging.info("Solver assignments saved -> %s", args.output)

    from collections import Counter
    logging.info("Summary: %s", dict(Counter(assignments.values())))