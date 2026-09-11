from __future__ import annotations

import hashlib
from typing import Any

import torch

from .posterior_capacity import _stable_seed
from .posterior_t64_protocol import PHYSICAL_MASK_NAMES, identity, make_physical_masks
from .util import canonical_json_bytes


RANDOM_TOKEN_MASK_NAMES = (
    "random_state_10", "random_state_35", "random_state_65", "random_state_90",
    "random_action_10", "random_action_35", "random_action_65", "random_action_90",
    "random_both_10", "random_both_35", "random_both_65", "random_both_90",
    "single_state", "single_action", "full_state", "full_action",
)
RANDOM_TOKEN_RATES = (0.10, 0.35, 0.65, 0.90)
TRAINING_MIXTURE = {
    "independent_token": 0.55,
    "dynamic_physical": 0.25,
    "sparse_token": 0.10,
    "full_domain": 0.10,
}

# H50-SCVAE deliberately excludes independent/scattered token corruption.  Its
# random bank is made only from physically constrained gaps and their separated
# compositions.  The first eight slots are one dynamic instance of the eight
# fixed physical families; the remaining slots are four double and four triple
# gap compositions.
PHYSICAL_RANDOM_MASK_NAMES = (
    *PHYSICAL_MASK_NAMES,
    "physical_double_0", "physical_double_1", "physical_double_2", "physical_double_3",
    "physical_triple_0", "physical_triple_1", "physical_triple_2", "physical_triple_3",
)
PHYSICAL_RANDOM_TRAINING_MIXTURE = {"single_dynamic_physical": 0.60, "composed_physical": 0.40}


def _token_masks(
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_state = batch["valid_state"].bool()
    valid_action = batch["valid_action"].bool()
    return valid_state, valid_action, torch.zeros_like(valid_state), torch.zeros_like(valid_action)


def _expand(
    batch: dict[str, Any], state_tokens: torch.Tensor, action_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        state_tokens[..., None].expand_as(batch["physical_state"]).clone(),
        action_tokens[..., None].expand_as(batch["action"]).clone(),
    )


def _select_exact(
    valid: torch.Tensor, count: int, generator: torch.Generator
) -> torch.Tensor:
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    if indices.numel() == 0:
        return torch.zeros_like(valid)
    count = max(1, min(int(count), int(indices.numel())))
    chosen = indices[torch.randperm(indices.numel(), generator=generator)[:count]]
    result = torch.zeros_like(valid)
    result[chosen] = True
    return result


def _select_rate(
    valid: torch.Tensor, rate: float, generator: torch.Generator
) -> torch.Tensor:
    return _select_exact(valid, round(float(rate) * int(valid.sum())), generator)


def validate_complete_token_masks(
    batch: dict[str, Any], state_mask: torch.Tensor, action_mask: torch.Tensor
) -> None:
    if state_mask.shape != batch["physical_state"].shape:
        raise ValueError("State Mask shape does not match the 70-D State")
    if action_mask.shape != batch["action"].shape:
        raise ValueError("Action Mask shape does not match the 29-D Action")
    state_any = state_mask.any(dim=-1)
    action_any = action_mask.any(dim=-1)
    if not torch.equal(state_any, state_mask.all(dim=-1)):
        raise ValueError("complete-token protocol forbids partial State feature Masks")
    if not torch.equal(action_any, action_mask.all(dim=-1)):
        raise ValueError("complete-token protocol forbids partial Action feature Masks")
    if bool((state_any & ~batch["valid_state"].bool()).any()):
        raise ValueError("State Mask covers padding")
    if bool((action_any & ~batch["valid_action"].bool()).any()):
        raise ValueError("Action Mask covers padding")
    for index in range(state_mask.shape[0]):
        if not bool(state_any[index].any()) and not bool(action_any[index].any()):
            raise ValueError("every complete-token sample must contain at least one target")


def make_heldout_random_token_masks(
    batch: dict[str, Any], seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Build the fixed 16-query held-out bank for every seen window."""
    valid_state, valid_action, state_tokens, action_tokens = _token_masks(batch)
    slots = batch.get("mask_slot")
    if slots is None:
        raise ValueError("held-out complete-token evaluation requires mask_slot")
    names: list[str] = []
    for index, raw_slot in enumerate(slots.tolist()):
        slot = int(raw_slot)
        if not 0 <= slot < len(RANDOM_TOKEN_MASK_NAMES):
            raise ValueError("held-out complete-token mask_slot is out of range")
        name = RANDOM_TOKEN_MASK_NAMES[slot]
        names.append(name)
        generator = torch.Generator().manual_seed(
            _stable_seed(seed, "heldout-random-token", *identity(batch, index), slot)
        )
        if slot < 12:
            domain = slot // 4
            rate = RANDOM_TOKEN_RATES[slot % 4]
            if domain in {0, 2}:
                state_tokens[index] = _select_rate(valid_state[index], rate, generator)
            if domain in {1, 2}:
                action_tokens[index] = _select_rate(valid_action[index], rate, generator)
        elif slot == 12:
            state_tokens[index] = _select_exact(valid_state[index], 1, generator)
        elif slot == 13:
            action_tokens[index] = _select_exact(valid_action[index], 1, generator)
        elif slot == 14:
            state_tokens[index] = valid_state[index]
        else:
            action_tokens[index] = valid_action[index]
    state_mask, action_mask = _expand(batch, state_tokens, action_tokens)
    validate_complete_token_masks(batch, state_mask, action_mask)
    return state_mask, action_mask, names


def _uniform_rate(generator: torch.Generator) -> float:
    return 0.05 + 0.90 * float(torch.rand((), generator=generator))


def make_dynamic_random_token_masks(
    batch: dict[str, Any], seed: int, optimizer_step: int, sample_slot: int = 0
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Sample the registered 55/25/10/10 complete-token training mixture."""
    valid_state, valid_action, state_tokens, action_tokens = _token_masks(batch)
    names: list[str] = []
    sparse_counts = (1, 2, 4, 8)
    physical_slots = torch.zeros(len(batch["motion_key"]), dtype=torch.long)
    physical_indices: list[int] = []
    for index in range(len(batch["motion_key"])):
        generator = torch.Generator().manual_seed(
            _stable_seed(
                seed,
                "train-random-token",
                int(optimizer_step),
                int(sample_slot),
                *identity(batch, index),
            )
        )
        bucket = int(torch.randint(0, 100, (), generator=generator))
        if bucket < 55:
            domain = int(torch.randint(0, 3, (), generator=generator))
            if domain == 0:
                state_tokens[index] = _select_rate(
                    valid_state[index], _uniform_rate(generator), generator
                )
                names.append("train_independent_state")
            elif domain == 1:
                action_tokens[index] = _select_rate(
                    valid_action[index], _uniform_rate(generator), generator
                )
                names.append("train_independent_action")
            else:
                state_tokens[index] = _select_rate(
                    valid_state[index], _uniform_rate(generator), generator
                )
                action_tokens[index] = _select_rate(
                    valid_action[index], _uniform_rate(generator), generator
                )
                names.append("train_independent_both")
        elif bucket < 80:
            physical_slot = int(
                torch.randint(0, len(PHYSICAL_MASK_NAMES), (), generator=generator)
            )
            physical_slots[index] = physical_slot
            physical_indices.append(index)
            names.append(f"train_physical_{PHYSICAL_MASK_NAMES[physical_slot]}")
        elif bucket < 90:
            count = sparse_counts[
                int(torch.randint(0, len(sparse_counts), (), generator=generator))
            ]
            combined_valid = torch.cat((valid_state[index], valid_action[index]))
            combined = _select_exact(combined_valid, count, generator)
            state_tokens[index] = combined[: valid_state.shape[1]]
            action_tokens[index] = combined[valid_state.shape[1] :]
            names.append(f"train_sparse_{count}")
        elif bucket < 95:
            state_tokens[index] = valid_state[index]
            names.append("train_full_state")
        else:
            action_tokens[index] = valid_action[index]
            names.append("train_full_action")
    if physical_indices:
        physical_batch = dict(batch)
        physical_batch["mask_slot"] = physical_slots
        physical_state, physical_action, _ = make_physical_masks(
            physical_batch,
            seed,
            dynamic_step=int(optimizer_step) * 100_000 + int(sample_slot),
        )
        for index in physical_indices:
            state_tokens[index] = physical_state[index].any(dim=-1)
            action_tokens[index] = physical_action[index].any(dim=-1)
    state_mask, action_mask = _expand(batch, state_tokens, action_tokens)
    validate_complete_token_masks(batch, state_mask, action_mask)
    return state_mask, action_mask, names


def _physical_interval(
    valid_transitions: int,
    minimum: int,
    maximum: int,
    generator: torch.Generator,
) -> tuple[int, int]:
    if valid_transitions <= 0:
        raise ValueError("physical Mask interval requires a valid transition")
    lower = max(1, min(minimum, valid_transitions))
    upper = max(lower, min(maximum, valid_transitions))
    length = int(torch.randint(lower, upper + 1, (), generator=generator))
    start = int(torch.randint(0, valid_transitions - length + 1, (), generator=generator))
    return start, length


def _apply_physical_family(
    state_tokens: torch.Tensor,
    action_tokens: torch.Tensor,
    family: int,
    valid_transitions: int,
    generator: torch.Generator,
) -> None:
    if family == 0:  # short State gap; both State boundaries and all Actions remain visible
        start, length = _physical_interval(valid_transitions, 2, 8, generator)
        state_tokens[start + 1 : start + length] = True
    elif family == 1:  # long State gap
        start, length = _physical_interval(valid_transitions, 9, 24, generator)
        state_tokens[start + 1 : start + length] = True
    elif family == 2:  # forward rollout from one visible State anchor with all Actions visible
        maximum_anchor = max(valid_transitions - 8, 0)
        anchor = int(torch.randint(0, maximum_anchor + 1, (), generator=generator))
        state_tokens[anchor + 1 : valid_transitions + 1] = True
    elif family == 3:  # short Action gap; all States remain visible
        start, length = _physical_interval(valid_transitions, 1, 8, generator)
        action_tokens[start : start + length] = True
    elif family == 4:  # long Action gap
        start, length = _physical_interval(valid_transitions, 9, 24, generator)
        action_tokens[start : start + length] = True
    elif family == 5:  # inverse-style full Action completion
        action_tokens[:valid_transitions] = True
    elif family == 6:  # short joint gap with visible State endpoints
        start, length = _physical_interval(valid_transitions, 1, 4, generator)
        action_tokens[start : start + length] = True
        state_tokens[start + 1 : start + length] = True
    elif family == 7:  # longer, but still bounded, joint inpainting gap
        start, length = _physical_interval(valid_transitions, 5, 8, generator)
        action_tokens[start : start + length] = True
        state_tokens[start + 1 : start + length] = True
    else:
        raise ValueError("physical Mask family is out of range")


def _separated_intervals(
    valid_transitions: int,
    count: int,
    generator: torch.Generator,
) -> list[tuple[int, int]]:
    """Choose short intervals separated by at least one visible transition."""
    for _ in range(64):
        candidates = [
            _physical_interval(valid_transitions, 2, min(8, valid_transitions), generator)
            for _ in range(count)
        ]
        candidates.sort()
        if all(
            candidates[index][0] + candidates[index][1] + 1
            <= candidates[index + 1][0]
            for index in range(len(candidates) - 1)
        ):
            return candidates
    # Deterministic fallback for very short or unlucky windows.  T64 always has
    # ample room, but keeping a fallback makes the contract total and testable.
    spacing = max(valid_transitions // count, 2)
    result: list[tuple[int, int]] = []
    for index in range(count):
        start = min(index * spacing, valid_transitions - 1)
        next_start = min((index + 1) * spacing, valid_transitions)
        length = max(1, min(4, next_start - start - (1 if index + 1 < count else 0)))
        result.append((start, length))
    return result


def _apply_composed_physical_mask(
    state_tokens: torch.Tensor,
    action_tokens: torch.Tensor,
    count: int,
    valid_transitions: int,
    generator: torch.Generator,
) -> None:
    intervals = _separated_intervals(valid_transitions, count, generator)
    for start, length in intervals:
        family = int(torch.randint(0, 3, (), generator=generator))
        if family == 0:  # State-only gap
            state_tokens[start + 1 : start + length] = True
        elif family == 1:  # Action-only gap
            action_tokens[start : start + length] = True
        else:  # Joint gap, retaining the two boundary States
            action_tokens[start : start + length] = True
            state_tokens[start + 1 : start + length] = True


def validate_physically_inferable_token_masks(
    batch: dict[str, Any], state_mask: torch.Tensor, action_mask: torch.Tensor
) -> None:
    validate_complete_token_masks(batch, state_mask, action_mask)
    state_tokens = state_mask.any(dim=-1)
    action_tokens = action_mask.any(dim=-1)
    valid_state = batch["valid_state"].bool()
    valid_action = batch["valid_action"].bool()
    for index in range(state_tokens.shape[0]):
        if not bool(state_tokens[index].any() or action_tokens[index].any()):
            raise ValueError("physical random protocol requires at least one target token")
        if torch.equal(state_tokens[index], valid_state[index]):
            raise ValueError("physical random protocol forbids full-State Mask")
        if bool((state_tokens[index, :-1] & action_tokens[index]).all()) and bool(
            (state_tokens[index, 1:] & action_tokens[index]).all()
        ):
            raise ValueError("physical random protocol forbids an information-free full-both Mask")
        if not bool(valid_action[index].any()):
            raise ValueError("physical random protocol requires at least one valid transition")


def make_heldout_physical_random_masks(
    batch: dict[str, Any], seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Build 16 deterministic, unseen, physically constrained token Masks/window."""
    valid_state, valid_action, state_tokens, action_tokens = _token_masks(batch)
    slots = batch.get("mask_slot")
    if slots is None:
        raise ValueError("held-out physical evaluation requires mask_slot")
    names: list[str] = []
    for index, raw_slot in enumerate(slots.tolist()):
        slot = int(raw_slot)
        if not 0 <= slot < len(PHYSICAL_RANDOM_MASK_NAMES):
            raise ValueError("held-out physical mask_slot is out of range")
        name = PHYSICAL_RANDOM_MASK_NAMES[slot]
        names.append(name)
        generator = torch.Generator().manual_seed(
            _stable_seed(seed, "heldout-physical-token", *identity(batch, index), slot)
        )
        valid_transitions = int(valid_action[index].sum())
        if slot < len(PHYSICAL_MASK_NAMES):
            _apply_physical_family(
                state_tokens[index], action_tokens[index], slot, valid_transitions, generator
            )
        else:
            count = 2 if slot < 12 else 3
            _apply_composed_physical_mask(
                state_tokens[index], action_tokens[index], count, valid_transitions, generator
            )
        state_tokens[index] &= valid_state[index]
        action_tokens[index] &= valid_action[index]
    state_mask, action_mask = _expand(batch, state_tokens, action_tokens)
    validate_physically_inferable_token_masks(batch, state_mask, action_mask)
    return state_mask, action_mask, names


def make_dynamic_physical_random_masks(
    batch: dict[str, Any], seed: int, optimizer_step: int, sample_slot: int = 0
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Sample the registered 60/40 single/composed physical training mixture."""
    valid_state, valid_action, state_tokens, action_tokens = _token_masks(batch)
    names: list[str] = []
    for index in range(len(batch["motion_key"])):
        generator = torch.Generator().manual_seed(
            _stable_seed(
                seed,
                "train-physical-token",
                int(optimizer_step),
                int(sample_slot),
                *identity(batch, index),
            )
        )
        valid_transitions = int(valid_action[index].sum())
        if float(torch.rand((), generator=generator)) < 0.60:
            family = int(torch.randint(0, len(PHYSICAL_MASK_NAMES), (), generator=generator))
            _apply_physical_family(
                state_tokens[index], action_tokens[index], family, valid_transitions, generator
            )
            names.append(f"train_dynamic_{PHYSICAL_MASK_NAMES[family]}")
        else:
            count = 2 + int(torch.randint(0, 2, (), generator=generator))
            _apply_composed_physical_mask(
                state_tokens[index], action_tokens[index], count, valid_transitions, generator
            )
            names.append(f"train_physical_{count}_gap")
        state_tokens[index] &= valid_state[index]
        action_tokens[index] &= valid_action[index]
    state_mask, action_mask = _expand(batch, state_tokens, action_tokens)
    validate_physically_inferable_token_masks(batch, state_mask, action_mask)
    return state_mask, action_mask, names


def mask_identity_sha256(
    state_mask: torch.Tensor, action_mask: torch.Tensor, names: list[str]
) -> str:
    digest = hashlib.sha256()
    digest.update(state_mask.cpu().numpy().tobytes(order="C"))
    digest.update(action_mask.cpu().numpy().tobytes(order="C"))
    digest.update(canonical_json_bytes(names))
    return digest.hexdigest()
