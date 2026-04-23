"""
solvers.py
----------
Five ODE-solver-inspired predictors for feature caching.
Each solver takes a `history` list of tensors (oldest first)
and predicts the next feature tensor.

All shapes are [B, C, H, W] or [B, C] — solvers are dimension-agnostic.

Solvers included (from HyCa paper Appendix A.2.2):
  rk2  — Runge-Kutta 2  (explicit, needs 3 history points)
  ab2  — Adams-Bashforth 2  (explicit, needs 3 history points)
  tf   — Taylor / linear extrapolation  (needs 2 history points)
  bdf2 — Backward Differentiation Formula 2  (needs 2 history points)
  am   — Adams-Moulton predictor-corrector  (needs 3 history points)
"""

import torch
import numpy as np
import json


# ──────────────────────────────────────────────
# Individual solvers
# ──────────────────────────────────────────────

def predict_rk2(history: list) -> torch.Tensor:
    """
    Runge-Kutta 2 — derived in HyCa paper eq. 33.

    F_{t+1} = 1.5 * F_t  -  0.5 * F_{t-2}

    Needs at least 3 history entries: [..., F_{t-2}, F_{t-1}, F_t]
    """
    assert len(history) >= 3, "rk2 needs >= 3 history points"
    Ft   = history[-1]
    Ft_2 = history[-3]
    return 1.5 * Ft - 0.5 * Ft_2


def predict_ab2(history: list) -> torch.Tensor:
    """
    Adams-Bashforth 2-step (explicit multistep).

    Derived from AB coefficients β = [3/2, -1/2]:
    F_{t+1} = 2.5*F_t - 2*F_{t-1} + 0.5*F_{t-2}

    Needs at least 3 history entries.
    """
    assert len(history) >= 3, "ab2 needs >= 3 history points"
    Ft   = history[-1]
    Ft_1 = history[-2]
    Ft_2 = history[-3]
    return 2.5 * Ft - 2.0 * Ft_1 + 0.5 * Ft_2


def predict_tf(history: list) -> torch.Tensor:
    """
    Taylor Formula — first-order (linear) extrapolation.

    F_{t+1} = 2*F_t - F_{t-1}

    Needs at least 2 history entries. Best for very smooth dimensions.
    """
    assert len(history) >= 2, "tf needs >= 2 history points"
    Ft   = history[-1]
    Ft_1 = history[-2]
    return 2.0 * Ft - Ft_1


def predict_bdf2(history: list) -> torch.Tensor:
    """
    Backward Differentiation Formula 2 — implicit (used as explicit predictor).

    BDF2 predictor: F_{t+1} = (4/3)*F_t - (1/3)*F_{t-1}

    Needs at least 2 history entries. Strong stability for stiff dimensions.
    """
    assert len(history) >= 2, "bdf2 needs >= 2 history points"
    Ft   = history[-1]
    Ft_1 = history[-2]
    return (4.0 / 3.0) * Ft - (1.0 / 3.0) * Ft_1


def predict_am(history: list) -> torch.Tensor:
    """
    Adams-Moulton predictor-corrector.

    Step 1 — predictor (AB2):
        F* = 2.5*Ft - 2*Ft_1 + 0.5*Ft_2

    Step 2 — corrector (AM2 with predicted derivative):
        f*   = F*  - Ft      (predicted slope)
        ft   = Ft  - Ft_1    (current slope)
        ft_1 = Ft_1 - Ft_2   (previous slope)
        F_{t+1} = Ft + (5/12)*f* + (2/3)*ft - (1/12)*ft_1

    Needs at least 3 history entries. Better accuracy for oscillatory dims.
    """
    assert len(history) >= 3, "am needs >= 3 history points"
    Ft   = history[-1]
    Ft_1 = history[-2]
    Ft_2 = history[-3]

    # Predictor (AB2)
    F_pred = 2.5 * Ft - 2.0 * Ft_1 + 0.5 * Ft_2

    # Slopes
    f_pred = F_pred - Ft
    f_t    = Ft     - Ft_1
    f_t1   = Ft_1   - Ft_2

    return Ft + (5.0 / 12.0) * f_pred + (2.0 / 3.0) * f_t - (1.0 / 12.0) * f_t1


# ──────────────────────────────────────────────
# Solver registry
# ──────────────────────────────────────────────

SOLVERS = {
    "rk2":  predict_rk2,
    "ab2":  predict_ab2,
    "tf":   predict_tf,
    "bdf2": predict_bdf2,
    "am":   predict_am,
}

SOLVER_NAMES = list(SOLVERS.keys())

# Minimum history length required by each solver
SOLVER_MIN_HISTORY = {
    "rk2":  3,
    "ab2":  3,
    "tf":   2,
    "bdf2": 2,
    "am":   3,
}


# ──────────────────────────────────────────────
# Per-cluster prediction
# ──────────────────────────────────────────────

def apply_solver_by_cluster(
    history: list,
    cluster_labels: np.ndarray,
    solver_assignments: dict,
):
    """
    If history contains lists of tensors (DeepCache prv_features format),
    apply solver element-wise and return a list.
    If history contains single tensors, apply dimension-wise and return a tensor.
    """
    # DeepCache prv_features is a list of tensors (one per cached layer)
    # Only apply HyCa solvers to 1280-channel layer, copy all others as-is
    import os
    if os.environ.get('DEBUG_SOLVER'):
        shape = history[0].shape if hasattr(history[0],'shape') else '?'
        print(f'[SOLVER] called — history len={len(history)}, shape={shape}, cluster_labels len={len(cluster_labels)}')
    if isinstance(history[0], list):
        result = []
        for layer_idx in range(len(history[0])):
            layer_hist = [h[layer_idx] for h in history]
            C = layer_hist[-1].shape[1]
            if C == len(cluster_labels) and len(layer_hist) >= 3:
                # 1280-channel mid_block — apply cluster-wise solvers
                result.append(_apply_solver_by_cluster_tensor(
                    layer_hist, cluster_labels, solver_assignments
                ))
            else:
                # All other layers — plain copy (same as DeepCache)
                result.append(layer_hist[-1])
        return result

    # Single tensor path
    C = history[-1].shape[1] if hasattr(history[-1], 'shape') else -1
    if C == len(cluster_labels):
        # Channels match — apply cluster-wise solver
        return _apply_solver_by_cluster_tensor(history, cluster_labels, solver_assignments)
    else:
        # Channels don't match clusters (different layer) —
        # apply the most common solver uniformly across all channels
        from collections import Counter
        best_solver = Counter(solver_assignments.values()).most_common(1)[0][0]
        return SOLVERS[best_solver](history)


def _apply_solver_by_cluster_tensor(
    history: list,
    cluster_labels: np.ndarray,
    solver_assignments: dict,
) -> torch.Tensor:
    """
    Predict the next feature tensor using a different solver per cluster.

    Args:
        history           : list of tensors [B, C, H, W], length >= 3, oldest first
        cluster_labels    : np.ndarray [C] — cluster id (int) for each channel
        solver_assignments: dict {str(cluster_id): solver_name}

    Returns:
        predicted tensor [B, C, H, W], same dtype/device as history[-1]
    """
    Ft = history[-1]
    predicted = torch.empty_like(Ft)

    for cid in np.unique(cluster_labels):
        solver_name = solver_assignments.get(str(int(cid)), "tf")
        solver_fn   = SOLVERS[solver_name]
        min_hist    = SOLVER_MIN_HISTORY[solver_name]

        # Boolean mask over channel dimension
        mask = cluster_labels == cid   # [C] bool array

        # Slice history for this cluster: list of [B, n_c, H, W]
        cluster_hist = [h[:, mask] for h in history[-min_hist:]]

        # Predict and write back
        pred_channels = solver_fn(cluster_hist)
        predicted[:, mask] = pred_channels.to(Ft.dtype)

    return predicted


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def load_solver_assignments(path: str = "solver_assignments.json") -> dict:
    with open(path) as f:
        return json.load(f)


def load_cluster_labels(path: str = "cluster_labels.json") -> np.ndarray:
    """
    Expects JSON with key 'labels': list of int cluster ids per channel.
    Falls back to 'indicators' key (single-indicator file) and runs k-means.
    """
    with open(path) as f:
        data = json.load(f)

    if "labels" in data:
        return np.array(data["labels"], dtype=np.int32)

    # Fallback: cluster from single-indicator file
    from sklearn.cluster import KMeans
    indicators = np.array(data).reshape(-1, 1)
    k = 7  # default from elbow plot
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(indicators)
    return labels.astype(np.int32)
