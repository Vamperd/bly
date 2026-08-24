from __future__ import annotations

PHYSICAL_STATE_FIELDS = (
    ("joint_pos", 29),
    ("joint_vel", 29),
    ("base_ang_vel", 3),
    ("gravity_robot", 3),
)
PHYSICAL_STATE_DIM = 64
PREVIOUS_ACTION_DIM = 29
ACTION_DIM = 29
DEFAULT_WINDOW_TRANSITIONS = 128
DEFAULT_VALIDATION_STRIDE = 64
CONTROL_DT = 0.02
PACKAGES = (
    "Locomotion",
    "Communication",
    "Interactions",
    "Dances",
    "Gaming",
    "Everyday",
    "Sport",
    "Other",
)
SPLITS = ("train", "validation", "test")
TASK_NAMES = ("forward", "inverse", "completion")
COMPLETION_NAMES = ("element", "step", "feature")

