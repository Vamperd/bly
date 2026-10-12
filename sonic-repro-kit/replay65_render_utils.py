"""Shared frame provenance overlays for 65-token replay videos."""
from __future__ import annotations


FRAME_PROVENANCE_VERSION = "train_window_v1"


def frame_provenance(meta, horizon):
    """Describe which rendered State frames are from the selected window.

    A single-window State-only replay supplies the first T64 window to the
    model and uses only source State for the remaining chunks.  State frame 64
    is the shared terminal State of the selected window; continuation starts
    at frame 65.  Legacy multi-window runs retain the old all-window label.
    """
    horizon = int(horizon)
    last_state = horizon
    single = bool(meta.get("single_window_continuation", False))
    if single:
        train_end = min(64, last_state)
        general_start = train_end + 1 if train_end < last_state else None
        general_end = last_state if general_start is not None else None
        mode = "single_window_state_only"
        label = (
            f"TRAINING WINDOW frames 0-{train_end} | "
            f"STATE-ONLY GENERALIZATION frames {general_start}-{general_end}"
            if general_start is not None else
            f"TRAINING WINDOW frames 0-{train_end}"
        )
    else:
        train_end = last_state
        general_start = general_end = None
        mode = "dataset_windows"
        label = f"SELECTED DATASET WINDOW frames 0-{train_end}"
    return {
        "version": FRAME_PROVENANCE_VERSION,
        "mode": mode,
        "training_state_frames": [0, train_end],
        "state_only_generalization_frames": None if general_start is None else [general_start, general_end],
        "label": label,
    }


def draw_provenance_timeline(cv2, frame, t, horizon, contract, y):
    """Draw green training and orange State-only continuation segments."""
    width = frame.shape[1] - 20
    count = int(horizon) + 1
    train_end = int(contract["training_state_frames"][1])
    train_color = (65, 150, 70)       # BGR green
    general_color = (45, 135, 225)    # BGR orange
    for j in range(count):
        x0 = 10 + j * width // count
        x1 = max(x0, 10 + (j + 1) * width // count - 1)
        color = train_color if j <= train_end else general_color
        cv2.rectangle(frame, (x0, y), (x1, y + 10), color, -1)
    if train_end < count - 1:
        boundary = 10 + (train_end + 1) * width // count
        cv2.line(frame, (boundary, y - 2), (boundary, y + 12), (255, 255, 255), 1)
    x = 10 + int(t) * width // max(count - 1, 1)
    cv2.line(frame, (x, y - 4), (x, y + 14), (255, 255, 255), 2)
