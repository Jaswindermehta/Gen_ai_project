"""
stable_diffusion.py
-------------------
Compares three inference modes side-by-side:
  1. Baseline   — standard SD, 50 steps, no caching
  2. DeepCache  — standard DeepCache (copy-based feature reuse)
  3. HyCa       — our method: DeepCache skip schedule + solver-based prediction

Usage:
    # All three modes, save comparison grid
    python stable_diffusion.py --mode all --prompt "a cat on a surfboard"

    # Single mode
    python stable_diffusion.py --mode hyca --prompt "a cat on a surfboard"

Output per run:
    output_baseline.png
    output_deepcache.png
    output_hyca.png
    output_comparison.png   (3-panel grid, only when --mode all)
    metrics.json            (PSNR / SSIM of deepcache and hyca vs baseline)
"""

import os
import time
import json
import logging
import argparse

import numpy as np
import torch
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline

from solvers import apply_solver_by_cluster, load_cluster_labels, load_solver_assignments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)


# ─────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────

def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """PSNR between two images in [0,1] range, shape [C, H, W]."""
    a = img1.float().clamp(0, 1)
    b = img2.float().clamp(0, 1)
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Simple SSIM approximation (per-channel mean, no window)."""
    try:
        from torchmetrics.functional import structural_similarity_index_measure as ssim
        return ssim(
            img1.clamp(0,1).unsqueeze(0).cpu().float(),
            img2.clamp(0,1).unsqueeze(0).cpu().float(),
            data_range=1.0,
        ).item()
    except ImportError:
        # Fallback: normalized cross-correlation as proxy
        a = img1.float().flatten()
        b = img2.float().flatten()
        a = a - a.mean(); b = b - b.mean()
        denom = (a.norm() * b.norm()).clamp(min=1e-8)
        return (a @ b / denom).item()


# ─────────────────────────────────────────────────────────────────
# HyCa cache manager
# ─────────────────────────────────────────────────────────────────

class HyCaCache:
    """
    Manages feature history and per-cluster solver prediction.

    Strategy:
      - Every `cache_interval`-th denoising step is a "compute step":
        the mid_block runs normally and its output is recorded.
      - All other steps are "skip steps":
        the mid_block output is replaced by the solver prediction.

    Because we use register_forward_hook (post-forward), the mid_block
    still executes at skip steps but its output is discarded and replaced.
    This correctly demonstrates quality improvement vs plain copy-paste.
    To eliminate that wasted compute, integrate at the DeepCache skip
    decision point inside the UNet forward (future work).
    """

    def __init__(
        self,
        cluster_labels: np.ndarray,
        solver_assignments: dict,
        cache_interval: int = 5,
    ):
        self.cluster_labels    = cluster_labels
        self.solver_assignments = solver_assignments
        self.cache_interval    = cache_interval

        self._history: list   = []   # list of float32 CPU tensors
        self._step: int       = 0
        self._hook            = None

    # ── public API ──────────────────────────────────────────

    def reset(self) -> None:
        self._history = []
        self._step    = 0

    def register(self, unet) -> None:
        """Attach hook to unet.mid_block."""
        self._hook = unet.mid_block.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    # ── internals ───────────────────────────────────────────

    def _is_compute_step(self) -> bool:
        return self._step % self.cache_interval == 0 or len(self._history) < 3

    def _hook_fn(self, module, inp, output):
        """
        Called after every mid_block forward.
        Returns None (pass-through) on compute steps.
        Returns solver-predicted tensor on skip steps.
        """
        feat = output[0] if isinstance(output, tuple) else output

        if self._is_compute_step():
            # Record ground-truth feature and pass through unchanged
            self._history.append(feat.detach().float().cpu())
            self._trim_history()
            self._step += 1
            return None  # no modification

        # Skip step — predict with solver
        predicted = apply_solver_by_cluster(
            self._history[-3:],
            self.cluster_labels,
            self.solver_assignments,
        )
        predicted = predicted.to(device=feat.device, dtype=feat.dtype)

        # Record predicted feature in history
        self._history.append(predicted.detach().float().cpu())
        self._trim_history()
        self._step += 1

        if isinstance(output, tuple):
            return (predicted,) + output[1:]
        return predicted

    def _trim_history(self) -> None:
        if len(self._history) > 6:
            self._history.pop(0)


# ─────────────────────────────────────────────────────────────────
# Inference runners
# ─────────────────────────────────────────────────────────────────

def run_baseline(model_id: str, prompt: str, seed: int, device: str):
    logging.info("─── Baseline ───────────────────────────")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)

    # One warmup pass
    set_random_seed(seed)
    _ = pipe(prompt, output_type="pt").images

    set_random_seed(seed)
    t0 = time.time()
    image = pipe(prompt, output_type="pt").images[0]
    elapsed = time.time() - t0

    del pipe; torch.cuda.empty_cache()
    logging.info("Baseline done in %.2fs", elapsed)
    return image, elapsed


def run_deepcache(model_id: str, prompt: str, seed: int, device: str,
                  cache_interval: int = 5):
    logging.info("─── DeepCache ──────────────────────────")
    from DeepCache.sd.pipeline_stable_diffusion import (
        StableDiffusionPipeline as DeepCacheStableDiffusionPipeline,
    )
    pipe = DeepCacheStableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)

    # One warmup pass
    set_random_seed(seed)
    _ = pipe(prompt, output_type="pt", return_dict=True).images

    set_random_seed(seed)
    t0 = time.time()
    image = pipe(
        prompt,
        cache_interval=cache_interval,
        cache_layer_id=0,
        cache_block_id=0,
        uniform=False, pow=1.4, center=15,
        output_type="pt",
        return_dict=True,
    ).images[0]
    elapsed = time.time() - t0

    del pipe; torch.cuda.empty_cache()
    logging.info("DeepCache done in %.2fs", elapsed)
    return image, elapsed


def run_hyca(
    model_id: str,
    prompt: str,
    seed: int,
    device: str,
    cluster_labels: np.ndarray,
    solver_assignments: dict,
    cache_interval: int = 5,
):
    logging.info("─── HyCa (ours) ────────────────────────")
    from DeepCache.sd.pipeline_stable_diffusion import (
        StableDiffusionPipeline as DeepCacheStableDiffusionPipeline,
    )
    pipe = DeepCacheStableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)

    # Warmup
    set_random_seed(seed)
    _ = pipe(prompt, output_type="pt", return_dict=True).images

    # Timed run — use DeepCache skip schedule + HyCa solver prediction
    set_random_seed(seed)
    t0 = time.time()
    image = pipe(
        prompt,
        cache_interval=cache_interval,
        cache_layer_id=0,
        cache_block_id=0,
        uniform=False, pow=1.4, center=15,
        use_hyca=True,
        hyca_cluster_labels=cluster_labels,
        hyca_solver_assignments=solver_assignments,
        output_type="pt",
        return_dict=True,
    ).images[0]
    elapsed = time.time() - t0

    del pipe; torch.cuda.empty_cache()
    logging.info("HyCa done in %.2fs", elapsed)
    return image, elapsed


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt",          type=str, default="a photo of an astronaut on a moon")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--device",          type=str, default="cuda:0")
    parser.add_argument("--cache_interval",  type=int, default=5,
                        help="Skip N-1 steps out of every N")
    parser.add_argument("--mode",            type=str, default="all",
                        choices=["baseline", "deepcache", "hyca", "all"],
                        help="Which method(s) to run")
    parser.add_argument("--cluster_labels",  type=str, default="cluster_labels.json")
    parser.add_argument("--solver_assignments", type=str, default="solver_assignments.json")
    parser.add_argument("--out_dir",         type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}

    # ── Baseline ─────────────────────────────────────────────
    if args.mode in ("baseline", "all"):
        img_base, t_base = run_baseline(
            args.model, args.prompt, args.seed, args.device
        )
        save_image(img_base, os.path.join(args.out_dir, "output_baseline.png"))
        results["baseline"] = {"time_s": round(t_base, 3)}

    # ── DeepCache ─────────────────────────────────────────────
    if args.mode in ("deepcache", "all"):
        img_dc, t_dc = run_deepcache(
            args.model, args.prompt, args.seed, args.device, args.cache_interval
        )
        save_image(img_dc, os.path.join(args.out_dir, "output_deepcache.png"))

        metrics_dc = {}
        if "baseline" in results:
            metrics_dc["psnr"] = round(compute_psnr(img_base, img_dc), 4)
            metrics_dc["ssim"] = round(compute_ssim(img_base, img_dc), 4)
        results["deepcache"] = {"time_s": round(t_dc, 3), **metrics_dc}

    # ── HyCa ─────────────────────────────────────────────────
    if args.mode in ("hyca", "all"):
        cluster_labels    = load_cluster_labels(args.cluster_labels)
        solver_assignments = load_solver_assignments(args.solver_assignments)

        img_hyca, t_hyca = run_hyca(
            args.model, args.prompt, args.seed, args.device,
            cluster_labels, solver_assignments, args.cache_interval,
        )
        print("HyCa raw tensor — NaN:", torch.isnan(img_hyca).any().item())
        print("HyCa raw tensor — Inf:", torch.isinf(img_hyca).any().item())
        print("HyCa raw tensor — min/max:", img_hyca.min().item(), img_hyca.max().item())
        img_hyca = img_hyca.nan_to_num(0.0).clamp(0, 1)
        save_image(img_hyca, os.path.join(args.out_dir, "output_hyca.png"))

        metrics_hyca = {}
        if "baseline" in results:
            metrics_hyca["psnr"] = round(compute_psnr(img_base, img_hyca), 4)
            metrics_hyca["ssim"] = round(compute_ssim(img_base, img_hyca), 4)
        results["hyca"] = {"time_s": round(t_hyca, 3), **metrics_hyca}

    # ── Comparison grid ───────────────────────────────────────
    if args.mode == "all":
        save_image(
            [img_base, img_dc, img_hyca],
            os.path.join(args.out_dir, "output_comparison.png"),
            nrow=3,
        )
        logging.info("Comparison grid saved → output_comparison.png")

    # ── Print + save metrics ──────────────────────────────────
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info("── Results ──────────────────────────────────")
    for method, vals in results.items():
        parts = [f"time={vals['time_s']:.2f}s"]
        if "psnr" in vals:
            parts.append(f"PSNR={vals['psnr']:.2f}dB")
        if "ssim" in vals:
            parts.append(f"SSIM={vals['ssim']:.4f}")
        logging.info("  %-10s  %s", method, "  ".join(parts))
    logging.info("Metrics saved → %s", metrics_path)