"""Selection policies for stochastic Action completion candidates."""

from __future__ import annotations

import numpy as np


LATENT_MODES = {"prior_mean", "oracle_best_of_n"}


def select_latent_action_window(
    prior_mean: np.ndarray,
    candidates: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    latent_mode: str,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Select one whole Action candidate; oracle mode may inspect masked truth."""
    if latent_mode not in LATENT_MODES:
        raise ValueError(f"unsupported latent_mode {latent_mode!r}")
    if prior_mean.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError("prior_mean, truth, and mask must have identical [T,A] shapes")
    if candidates.ndim != 3 or tuple(candidates.shape[1:]) != tuple(truth.shape):
        raise ValueError("candidates must have shape [N,T,A] matching truth")
    if candidates.shape[0] == 0 or not mask.any():
        raise ValueError("latent candidate selection requires candidates and masked values")
    candidate_errors = np.sqrt(
        np.mean(np.square(candidates[:, mask] - truth[mask][None]), axis=1)
    ).astype(np.float32)
    if latent_mode == "prior_mean":
        return prior_mean, -1, candidate_errors
    selected_index = int(np.argmin(candidate_errors))
    return candidates[selected_index], selected_index, candidate_errors
