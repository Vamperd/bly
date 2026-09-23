"""Opt-in runtime contract for replay65. Importable without Isaac for unit tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import torch


def suppress_interval_event(env, *args, **kwargs):
    """Recorded baseline has no external pushes; do not draw fresh interval noise."""
    del env, args, kwargs


def compare_runtime(expected, actual):
    checks = {}
    for key in ("sim_dt", "control_dt", "decimation", "gravity_w",
                "solver_position_iteration_count", "solver_velocity_iteration_count"):
        a, b = expected.get(key), actual.get(key)
        checks[key] = None if a is None or b is None else bool(np.allclose(a,b,atol=1e-9,rtol=0))
    return checks


def audit_and_freeze(raw, values):
    contract = json.loads(str(values["replay65_contract"].tolist()))
    source_events = contract.get("active_events")
    if isinstance(source_events,dict) and source_events.get("interval"):
        raise ValueError("source has interval events without recorded event schedule; cannot silently remove source forces")
    manager = raw.event_manager
    before = {str(k):list(v) for k,v in manager.active_terms.items()}
    suppressed = []
    # Use the public term configuration API; keep EventManager timers/indexing intact.
    for name in before.get("interval",[]):
        cfg = copy.copy(manager.get_term_cfg(name))
        cfg.func = suppress_interval_event
        manager.set_term_cfg(name,cfg)
        suppressed.append(name)
    robot = raw.scene["robot"]
    props = robot.cfg.spawn.articulation_props
    actual = {"sim_dt":float(raw.physics_dt),"control_dt":float(raw.step_dt),
        "decimation":int(raw.cfg.decimation),"gravity_w":list(raw.cfg.sim.gravity),
        "solver_position_iteration_count":getattr(props,"solver_position_iteration_count",None),
        "solver_velocity_iteration_count":getattr(props,"solver_velocity_iteration_count",None)}
    timing = compare_runtime(contract["simulation"],actual)
    if any(v is False for v in timing.values()):
        raise ValueError(f"replay simulation/solver contract mismatch: {timing}")
    actuator_rows = {}
    source_groups = contract.get("actuators")
    actuator_identity = isinstance(source_groups,dict)
    unknown_delay = False
    for name, actuator in robot.actuators.items():
        current = {"type":type(actuator).__name__,"joint_names":list(actuator.joint_names),
            "min_delay":int(getattr(actuator.cfg,"min_delay",0)),"max_delay":int(getattr(actuator.cfg,"max_delay",0))}
        if isinstance(source_groups,dict) and source_groups.get(name) != current:
            raise ValueError(f"actuator contract mismatch for {name}: {source_groups.get(name)} != {current}")
        buffers = {}
        for attr in ("positions_delay_buffer","velocities_delay_buffer","efforts_delay_buffer"):
            buf = getattr(actuator,attr,None)
            if buf is not None:
                lag = getattr(buf,"time_lags",None)
                buffers[attr] = {"runtime_lag":lag.detach().cpu().tolist() if isinstance(lag,torch.Tensor) else None,
                    "historical_lag":"unknown" if current["max_delay"] else "zero",
                    "queue_restoration":"unknown_not_fabricated" if current["max_delay"] else "not_needed_zero_delay"}
        unknown_delay |= current["max_delay"] > 0
        actuator_rows[name] = {**current,"buffers":buffers}
    if isinstance(source_groups,dict) and set(source_groups) != set(actuator_rows):
        raise ValueError("recorded/runtime actuator groups differ")
    asset_path = getattr(robot.cfg.spawn,"usd_path",None)
    asset_sha = None
    if asset_path and Path(asset_path).is_file():
        digest = hashlib.sha256()
        with Path(asset_path).open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):
                digest.update(block)
        asset_sha = digest.hexdigest()
    expected_asset = contract.get("asset") or {}
    asset_matches = None if not expected_asset.get("sha256") or not asset_sha else asset_sha == expected_asset["sha256"]
    if asset_matches is False:
        raise ValueError("source/runtime robot USD asset hash mismatch")
    contact = contract.get("contact") or {}
    if contact.get("threshold_n") is not None and not np.isclose(contact["threshold_n"],10.):
        raise ValueError("replay recorder contact threshold differs from source")
    report = {"version":"65-token-replay-runtime-v1","source_contract":contract,
        "simulation":actual,"simulation_checks":timing,"actuators":actuator_rows,
        "runtime_versions":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda},
        "asset":{"path":asset_path,"sha256":asset_sha,"matches_source":asset_matches},
        "events_before_freeze":before,"suppressed_interval_events":suppressed,
        "events_after_freeze":{k:[n for n in v if n not in suppressed] for k,v in before.items()},
        "hidden_state":{"solver_contact_cache":"unknown_not_recorded","actuator_queue":"unknown" if unknown_delay else "zero_delay_not_needed"},
        "contract_verified":all(v is True for v in timing.values()) and actuator_identity and not unknown_delay and isinstance(source_events,dict),
        "midrun_reset":False,"control_steps":0,"substeps_per_control":int(raw.cfg.decimation)}
    raw._replay65_audit = report
    # No claim that reconstructing exposed state restores the entire simulator.
    return report
