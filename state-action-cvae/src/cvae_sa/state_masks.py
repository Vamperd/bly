from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable

import numpy as np

from .action_masks import semantic_joint_groups
from .constants import ACTION_DIM, DEFAULT_WINDOW_TRANSITIONS


STATE_DIM = 70
STATE_STEPS = DEFAULT_WINDOW_TRANSITIONS + 1
PRESET_NAME = "state_prediction_v1"
SCHEMA_VERSION = "sonic_state_mask_scenario_v1"
STATE_FIELD_SLICES = {
    "joint_position": (0, 29),
    "joint_velocity": (29, 58),
    "base_linear_velocity": (58, 61),
    "base_angular_velocity": (61, 64),
    "gravity": (64, 67),
    "base_height": (67, 68),
    "foot_contact": (68, 70),
    "base_motion": (58, 64),
}


@dataclass(frozen=True)
class StateMaskScenario:
    name: str
    task: str
    granularity: str
    target: str
    window_start: int
    temporal_selector: dict[str, Any]
    feature_selector: dict[str, Any]
    fraction: float | None
    seed: int
    distribution_status: str
    rollout_horizon: int = 0
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def validate_scenario(scenario: StateMaskScenario) -> None:
    if scenario.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported State scenario schema {scenario.schema_version!r}")
    if scenario.task not in {"completion", "forward_rollout"}:
        raise ValueError(f"unsupported State task {scenario.task!r}")
    if scenario.granularity not in {"element", "step", "feature", "semantic", "rollout"}:
        raise ValueError(f"unsupported State granularity {scenario.granularity!r}")
    if scenario.target != "state":
        raise ValueError("State evaluator only accepts State-only masks")
    if scenario.window_start < 0:
        raise ValueError("window_start must be non-negative")
    if scenario.distribution_status not in {"in_distribution", "out_of_distribution"}:
        raise ValueError("invalid State scenario distribution status")
    if scenario.task == "forward_rollout":
        if scenario.rollout_horizon not in {1, 2, 4, 8, 32}:
            raise ValueError("forward rollout horizon must be 1, 2, 4, 8, or 32")
    elif scenario.rollout_horizon:
        raise ValueError("completion scenarios cannot declare a rollout horizon")
    if scenario.fraction is not None and not 0.0 < scenario.fraction <= 1.0:
        raise ValueError("State scenario fraction must be in (0, 1]")


def _joint_semantic_indices(
    joint_names: Iterable[str], group_name: str
) -> list[int]:
    names = tuple(str(value) for value in joint_names)
    groups = semantic_joint_groups(names)
    if group_name not in groups:
        raise ValueError(f"unknown State joint semantic group {group_name!r}")
    indices = [names.index(name) for name in groups[group_name]]
    return [*indices, *(ACTION_DIM + index for index in indices)]


def build_default_scenarios(
    window_start: int,
    peak_state_start: int,
    joint_names: Iterable[str],
    seed: int,
) -> list[StateMaskScenario]:
    names = tuple(str(value) for value in joint_names)
    semantic_joint_groups(names)  # validates the exact G1 partition
    if not 0 <= peak_state_start <= STATE_STEPS - 32:
        raise ValueError("peak State start must leave room for a 32-step scenario")
    scenarios: list[StateMaskScenario] = []

    for horizon in (1, 2, 4, 8, 32):
        name = f"forward_rollout_{horizon}"
        scenarios.append(
            StateMaskScenario(
                name=name,
                task="forward_rollout",
                granularity="rollout",
                target="state",
                window_start=window_start,
                temporal_selector={"mode": "contiguous", "start_in_window": peak_state_start + 1, "length": horizon},
                feature_selector={"mode": "all"},
                fraction=1.0,
                seed=_stable_seed(seed, name),
                distribution_status=("out_of_distribution" if horizon == 32 else "in_distribution"),
                rollout_horizon=horizon,
            )
        )

    for percent in (10, 25, 40):
        name = f"element_{percent}"
        scenarios.append(
            StateMaskScenario(
                name=name,
                task="completion",
                granularity="element",
                target="state",
                window_start=window_start,
                temporal_selector={"mode": "random_elements"},
                feature_selector={"mode": "all"},
                fraction=percent / 100.0,
                seed=_stable_seed(seed, name),
                distribution_status="in_distribution",
            )
        )

    for length in (1, 2, 4, 8, 16, 32):
        name = f"step_contiguous_{length}"
        start = peak_state_start + (32 - length) // 2
        scenarios.append(
            StateMaskScenario(
                name=name,
                task="completion",
                granularity="step",
                target="state",
                window_start=window_start,
                temporal_selector={"mode": "contiguous", "start_in_window": start, "length": length},
                feature_selector={"mode": "all"},
                fraction=None,
                seed=_stable_seed(seed, name),
                distribution_status="in_distribution",
            )
        )

    for percent in (10, 25, 40):
        name = f"feature_random_{percent}"
        scenarios.append(
            StateMaskScenario(
                name=name,
                task="completion",
                granularity="feature",
                target="state",
                window_start=window_start,
                temporal_selector={"mode": "all"},
                feature_selector={"mode": "random_fraction"},
                fraction=percent / 100.0,
                seed=_stable_seed(seed, name),
                distribution_status="in_distribution",
            )
        )

    semantic_names = (
        "waist", "left_leg", "right_leg", "left_arm", "right_arm",
        "base_linear_velocity", "base_angular_velocity", "gravity",
        "base_height", "foot_contact", "base_motion",
    )
    for semantic in semantic_names:
        name = f"semantic_{semantic}"
        scenarios.append(
            StateMaskScenario(
                name=name,
                task="completion",
                granularity="semantic",
                target="state",
                window_start=window_start,
                temporal_selector={"mode": "all"},
                feature_selector={"mode": "semantic", "name": semantic},
                fraction=None,
                seed=_stable_seed(seed, name),
                distribution_status="in_distribution",
            )
        )
    for scenario in scenarios:
        validate_scenario(scenario)
    if len({item.name for item in scenarios}) != len(scenarios):
        raise RuntimeError("default State scenario names are not unique")
    return scenarios


def scenario_mask(
    scenario: StateMaskScenario,
    joint_names: Iterable[str],
) -> np.ndarray:
    validate_scenario(scenario)
    rng = np.random.default_rng(scenario.seed)
    mask = np.zeros((STATE_STEPS, STATE_DIM), dtype=bool)
    if scenario.granularity == "element":
        count = max(1, round(mask.size * float(scenario.fraction)))
        mask.reshape(-1)[rng.permutation(mask.size)[:count]] = True
        return mask

    selector = scenario.temporal_selector
    if selector["mode"] == "all":
        times = np.arange(STATE_STEPS)
    elif selector["mode"] == "contiguous":
        start = int(selector["start_in_window"])
        length = int(selector["length"])
        if start < 0 or length <= 0 or start + length > STATE_STEPS:
            raise ValueError(f"invalid temporal selector for {scenario.name}")
        times = np.arange(start, start + length)
    else:
        raise ValueError(f"unsupported temporal selector for {scenario.name}")

    feature = scenario.feature_selector
    if feature["mode"] == "all":
        dimensions = np.arange(STATE_DIM)
    elif feature["mode"] == "random_fraction":
        count = max(1, round(STATE_DIM * float(scenario.fraction)))
        dimensions = rng.permutation(STATE_DIM)[:count]
    elif feature["mode"] == "semantic":
        name = str(feature["name"])
        if name in {"waist", "left_leg", "right_leg", "left_arm", "right_arm"}:
            dimensions = np.asarray(_joint_semantic_indices(joint_names, name), dtype=np.int64)
        elif name in STATE_FIELD_SLICES:
            low, high = STATE_FIELD_SLICES[name]
            dimensions = np.arange(low, high)
        else:
            raise ValueError(f"unknown State semantic group {name!r}")
    else:
        raise ValueError(f"unsupported feature selector for {scenario.name}")
    mask[np.ix_(times, dimensions)] = True
    return mask

