from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constants import ACTION_DIM, DEFAULT_WINDOW_TRANSITIONS, TASK_NAMES


SCENARIO_SCHEMA_VERSION = "sonic_action_mask_scenario_v1"
PRESET_NAME = "all_action_masks_v1"


@dataclass(frozen=True)
class ActionMaskScenario:
    """Serializable description of one Action-only completion experiment."""

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
    schema_version: str = SCENARIO_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionMaskScenario":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown ActionMaskScenario fields: {unknown}")
        scenario = cls(**value)
        validate_scenario(scenario)
        return scenario


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def semantic_joint_groups(joint_names: Iterable[str]) -> dict[str, tuple[str, ...]]:
    names = tuple(str(name) for name in joint_names)
    if len(names) != ACTION_DIM or len(set(names)) != ACTION_DIM:
        raise ValueError("semantic groups require exactly 29 unique joint names")

    def selected(side: str | None, parts: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            name
            for name in names
            if (side is None or name.startswith(f"{side}_"))
            and any(part in name for part in parts)
        )

    groups = {
        "waist": selected(None, ("waist_",)),
        "left_leg": selected("left", ("hip_", "knee_", "ankle_")),
        "right_leg": selected("right", ("hip_", "knee_", "ankle_")),
        "left_arm": selected("left", ("shoulder_", "elbow_", "wrist_")),
        "right_arm": selected("right", ("shoulder_", "elbow_", "wrist_")),
    }
    expected = {"waist": 3, "left_leg": 6, "right_leg": 6, "left_arm": 7, "right_arm": 7}
    actual = {name: len(values) for name, values in groups.items()}
    if actual != expected:
        raise ValueError(f"unexpected semantic joint grouping: {actual}")
    flattened = [name for group in groups.values() for name in group]
    if len(flattened) != ACTION_DIM or set(flattened) != set(names):
        raise ValueError("semantic joint groups do not form a partition of the action joints")
    return groups


def validate_scenario(scenario: ActionMaskScenario) -> None:
    if scenario.schema_version != SCENARIO_SCHEMA_VERSION:
        raise ValueError(f"unsupported scenario schema {scenario.schema_version!r}")
    if not scenario.name or any(char.isspace() for char in scenario.name):
        raise ValueError("scenario name must be non-empty and contain no whitespace")
    if scenario.task not in TASK_NAMES:
        raise ValueError(f"unsupported task {scenario.task!r}")
    if scenario.granularity not in {"element", "step", "feature", "full"}:
        raise ValueError(f"unsupported granularity {scenario.granularity!r}")
    if scenario.target != "action":
        raise ValueError("this evaluator accepts Action-only scenarios")
    if scenario.window_start < 0:
        raise ValueError("window_start must be non-negative")
    if scenario.distribution_status not in {
        "in_distribution",
        "near_out_of_distribution",
        "out_of_distribution",
    }:
        raise ValueError(f"invalid distribution status {scenario.distribution_status!r}")
    mode = scenario.temporal_selector.get("mode")
    if mode not in {"all", "contiguous", "random_elements"}:
        raise ValueError(f"unsupported temporal selector {mode!r}")
    feature_mode = scenario.feature_selector.get("mode")
    if feature_mode not in {"all", "random_fraction", "joint_names"}:
        raise ValueError(f"unsupported feature selector {feature_mode!r}")
    if scenario.fraction is not None and not 0.0 < float(scenario.fraction) <= 1.0:
        raise ValueError("scenario fraction must be in (0, 1]")
    if scenario.task == "inverse" and scenario.granularity != "full":
        raise ValueError("inverse scenarios must use full granularity")


def build_default_scenarios(
    window_start: int,
    peak_block_start: int,
    joint_names: Iterable[str],
    seed: int,
    inverse_full_in_distribution: bool = False,
) -> list[ActionMaskScenario]:
    if not 0 <= peak_block_start <= DEFAULT_WINDOW_TRANSITIONS - 8:
        raise ValueError("peak_block_start must fit an eight-step block in the model window")
    groups = semantic_joint_groups(joint_names)
    scenarios: list[ActionMaskScenario] = []

    for percent in (10, 25, 40):
        fraction = percent / 100.0
        scenarios.append(
            ActionMaskScenario(
                name=f"element_{percent}",
                task="completion",
                granularity="element",
                target="action",
                window_start=window_start,
                temporal_selector={"mode": "random_elements"},
                feature_selector={"mode": "all", "joint_names": []},
                fraction=fraction,
                seed=_stable_seed(seed, f"element_{percent}"),
                distribution_status="in_distribution",
            )
        )

    for length in (1, 2, 4, 8):
        start = peak_block_start + (8 - length) // 2
        scenarios.append(
            ActionMaskScenario(
                name=f"step_contiguous_{length}",
                task="completion",
                granularity="step",
                target="action",
                window_start=window_start,
                temporal_selector={"mode": "contiguous", "start_in_window": start, "length": length},
                feature_selector={"mode": "all", "joint_names": []},
                fraction=None,
                seed=_stable_seed(seed, f"step_contiguous_{length}"),
                distribution_status="in_distribution",
            )
        )

    for percent in (10, 25, 40):
        fraction = percent / 100.0
        scenarios.append(
            ActionMaskScenario(
                name=f"feature_random_{percent}",
                task="completion",
                granularity="feature",
                target="action",
                window_start=window_start,
                temporal_selector={"mode": "all"},
                feature_selector={"mode": "random_fraction", "joint_names": []},
                fraction=fraction,
                seed=_stable_seed(seed, f"feature_random_{percent}"),
                distribution_status="in_distribution",
            )
        )

    for group_name, names in groups.items():
        scenarios.append(
            ActionMaskScenario(
                name=f"semantic_{group_name}",
                task="completion",
                granularity="feature",
                target="action",
                window_start=window_start,
                temporal_selector={"mode": "all"},
                feature_selector={"mode": "joint_names", "joint_names": list(names)},
                fraction=len(names) / ACTION_DIM,
                seed=_stable_seed(seed, f"semantic_{group_name}"),
                distribution_status="in_distribution",
            )
        )

    scenarios.append(
        ActionMaskScenario(
            name="inverse_full_128",
            task="inverse",
            granularity="full",
            target="action",
            window_start=window_start,
            temporal_selector={"mode": "all"},
            feature_selector={"mode": "all", "joint_names": []},
            fraction=1.0,
            seed=_stable_seed(seed, "inverse_full_128"),
            distribution_status=(
                "in_distribution" if inverse_full_in_distribution
                else "out_of_distribution"
            ),
        )
    )
    for scenario in scenarios:
        validate_scenario(scenario)
    if len({scenario.name for scenario in scenarios}) != len(scenarios):
        raise RuntimeError("default scenario names are not unique")
    return scenarios


def add_combined_limb_scenarios(
    scenarios: list[ActionMaskScenario], joint_names: Iterable[str], seed: int
) -> list[ActionMaskScenario]:
    groups = semantic_joint_groups(joint_names)
    additions = (
        ("both_legs_12", groups["left_leg"] + groups["right_leg"], "near_out_of_distribution"),
        ("both_arms_14", groups["left_arm"] + groups["right_arm"], "out_of_distribution"),
    )
    result = list(scenarios)
    window_start = result[0].window_start if result else 0
    for name, names, status in additions:
        result.append(
            ActionMaskScenario(
                name=name,
                task="completion",
                granularity="feature",
                target="action",
                window_start=window_start,
                temporal_selector={"mode": "all"},
                feature_selector={"mode": "joint_names", "joint_names": list(names)},
                fraction=len(names) / ACTION_DIM,
                seed=_stable_seed(seed, name),
                distribution_status=status,
            )
        )
    return result


def scenario_mask(
    scenario: ActionMaskScenario,
    joint_names: Iterable[str],
    window: int = DEFAULT_WINDOW_TRANSITIONS,
) -> np.ndarray:
    validate_scenario(scenario)
    names = tuple(str(name) for name in joint_names)
    if len(names) != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} action joints, found {len(names)}")
    rng = np.random.default_rng(scenario.seed)
    mask = np.zeros((window, ACTION_DIM), dtype=bool)

    if scenario.granularity == "element":
        count = max(1, round(window * ACTION_DIM * float(scenario.fraction)))
        selected = rng.permutation(window * ACTION_DIM)[:count]
        mask.reshape(-1)[selected] = True
        return mask

    temporal_mode = scenario.temporal_selector["mode"]
    if temporal_mode == "all":
        time_indices = np.arange(window)
    elif temporal_mode == "contiguous":
        start = int(scenario.temporal_selector["start_in_window"])
        length = int(scenario.temporal_selector["length"])
        if length <= 0 or start < 0 or start + length > window:
            raise ValueError(f"invalid contiguous selector for {scenario.name}")
        time_indices = np.arange(start, start + length)
    else:
        raise ValueError(f"temporal selector {temporal_mode!r} is invalid for {scenario.name}")

    feature_mode = scenario.feature_selector["mode"]
    if feature_mode == "all":
        feature_indices = np.arange(ACTION_DIM)
    elif feature_mode == "random_fraction":
        count = max(1, round(ACTION_DIM * float(scenario.fraction)))
        feature_indices = rng.permutation(ACTION_DIM)[:count]
    elif feature_mode == "joint_names":
        requested = tuple(str(name) for name in scenario.feature_selector.get("joint_names", []))
        missing = sorted(set(requested) - set(names))
        if missing:
            raise ValueError(f"{scenario.name} requests unknown joints: {missing}")
        feature_indices = np.asarray([names.index(name) for name in requested], dtype=np.int64)
    else:
        raise ValueError(f"unsupported feature selector {feature_mode!r}")
    if feature_indices.size == 0:
        raise ValueError(f"{scenario.name} masks no Action dimensions")
    mask[np.ix_(time_indices, feature_indices)] = True
    return mask


def load_scenarios(path: Path) -> list[ActionMaskScenario]:
    scenarios = [
        ActionMaskScenario.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scenarios:
        raise ValueError(f"scenario file is empty: {path}")
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("custom scenario names must be unique")
    return scenarios


def write_scenarios(path: Path, scenarios: Iterable[ActionMaskScenario]) -> None:
    lines = [json.dumps(scenario.to_dict(), ensure_ascii=False, sort_keys=True) for scenario in scenarios]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def masked_baselines(actions: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return hold-last and two-sided linear interpolation Action baselines."""

    actions = np.asarray(actions, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if actions.shape != mask.shape or actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions and mask must both be [T,{ACTION_DIM}]")
    steps = np.arange(actions.shape[0])
    hold = actions.copy()
    linear = actions.copy()
    for feature in range(ACTION_DIM):
        observed = ~mask[:, feature]
        if not observed.any():
            hold[:, feature] = 0.0
            linear[:, feature] = 0.0
            continue
        observed_steps = steps[observed]
        observed_values = actions[observed, feature]
        last_value = float(observed_values[0])
        for step in steps:
            if observed[step]:
                last_value = float(actions[step, feature])
            else:
                hold[step, feature] = last_value
        linear[:, feature] = np.interp(steps, observed_steps, observed_values)
    hold[~mask] = actions[~mask]
    linear[~mask] = actions[~mask]
    return hold, linear


def relative_to_raw_action(
    relative: np.ndarray,
    default: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
    clip: np.ndarray | None,
    wrapper_clip: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert SONIC's runtime JointPositionAction mapping.

    Returns raw actions, the achieved relative target after clipping, and a
    per-element saturation mask.
    """

    relative = np.asarray(relative, dtype=np.float32)
    default = np.asarray(default, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    if relative.ndim != 2 or relative.shape[1] != ACTION_DIM:
        raise ValueError(f"relative Action must be [T,{ACTION_DIM}]")
    if default.shape != (ACTION_DIM,) or scale.shape != (ACTION_DIM,) or offset.shape != (ACTION_DIM,):
        raise ValueError("default, scale, and offset must be 29-vectors")
    if np.any(np.abs(scale) < 1.0e-8):
        raise ValueError("Action scale contains a zero or near-zero element")
    requested_target = relative + default
    target = requested_target.copy()
    if clip is not None:
        clip = np.asarray(clip, dtype=np.float32)
        if clip.shape != (ACTION_DIM, 2) or np.any(clip[:, 0] > clip[:, 1]):
            raise ValueError("Action clip must have shape [29,2] with lower <= upper")
        target = np.clip(target, clip[:, 0], clip[:, 1])
    raw = (target - offset) / scale
    if wrapper_clip is not None and wrapper_clip > 0:
        raw = np.clip(raw, -float(wrapper_clip), float(wrapper_clip))
        target = raw * scale + offset
        if clip is not None:
            target = np.clip(target, clip[:, 0], clip[:, 1])
    achieved = target - default
    saturated = np.abs(achieved - relative) > 1.0e-6
    round_trip = raw * scale + offset
    if clip is not None:
        round_trip = np.clip(round_trip, clip[:, 0], clip[:, 1])
    error = float(np.max(np.abs((round_trip - default) - achieved)))
    if error > 1.0e-6:
        raise ValueError(f"relative/raw/processed round-trip error is {error:.3e}")
    return raw.astype(np.float32), achieved.astype(np.float32), saturated
