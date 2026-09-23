"""Read-only, full/missing/visible diagnostics for the 65-token CVAE.

All masks use True = hidden. Physical statistics are per feature, never across
incompatible units. Quantiles use NumPy, not torch.quantile's size-limited path.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .cvae_protocol import digest, isolated_rng, scalar
from .posterior_t64_protocol import make_physical_masks
from .util import atomic_write_json

STATE_NAMES = ([f"joint_pos_{i}" for i in range(29)] + [f"joint_vel_{i}" for i in range(29)]
               + [f"base_lin_vel_{i}" for i in range(3)] + [f"base_ang_vel_{i}" for i in range(3)]
               + [f"gravity_robot_{i}" for i in range(3)] + ["base_height"])
UNITS = ["rad"] * 29 + ["rad/s"] * 29 + ["m/s"] * 3 + ["rad/s"] * 3 + ["unitless"] * 3 + ["m"]
PARTS = ("full", "masked", "visible")


def stats(values):
    values = np.asarray(values).reshape(-1)
    count = int(values.size)
    if not count:
        return {"count": 0, **{key: None for key in ("mse", "rmse", "mae", "p95", "p99", "p999", "max_abs")},
                "exceedances": {str(t): {"count": 0, "fraction": None} for t in (.01, .05, .1, .2)}}
    absolute = np.abs(values)
    mse = float(np.mean(np.square(values, dtype=np.float64)))
    quantiles = np.quantile(absolute, [.95, .99, .999])
    return {"count": count, "mse": mse, "rmse": mse ** .5, "mae": float(np.mean(absolute, dtype=np.float64)),
            "p95": float(quantiles[0]), "p99": float(quantiles[1]), "p999": float(quantiles[2]),
            "max_abs": float(absolute.max()), "exceedances": {
                str(t): {"count": int(np.count_nonzero(absolute > t)), "fraction": float(np.mean(absolute > t))}
                for t in (.01, .05, .1, .2)}}


def moments(values):
    values = np.asarray(values).reshape(-1)
    return {"count": int(values.size), "sse": float(np.square(values, dtype=np.float64).sum()),
            "max_abs": float(np.abs(values).max()) if values.size else None,
            "rmse": float(np.mean(np.square(values, dtype=np.float64))) ** .5 if values.size else None}


def merge_moments(rows):
    rows = list(rows)
    n = sum(row["count"] for row in rows)
    sse = sum(row["sse"] for row in rows)
    available = [row["rmse"] for row in rows if row["count"]]
    return {"count": n, "mse": sse / n if n else None, "micro_rmse": (sse / n) ** .5 if n else None,
            "macro_window_rmse": float(np.mean(available)) if available else None,
            "windows_with_targets": len(available), "zero_target_windows": len(rows) - len(available)}


def read_normalization(path):
    with np.load(path, allow_pickle=False) as data:
        result = {domain: (np.array(data[prefix + "_mean"]), np.array(data[prefix + "_std"]))
                  for domain, prefix in (("state", "physical_state"), ("action", "action"))}
    for domain, (mean, std) in result.items():
        expected = 70 if domain == "state" else 29
        if mean.shape != (expected,) or std.shape != (expected,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f"invalid normalization: {domain}")
    return result


def batch_ids(batch):
    return [{key: scalar(batch, key, i) for key in ("stable_window_id", "window_index", "fixture_index",
             "episode_ref", "motion_key", "variant_id", "window_start", "mask_slot")}
            for i in range(len(batch["physical_state"]))]


def epsilon_for(model, batch, sample, seed, device):
    globals_, locals_ = [], []
    for row in batch_ids(batch):
        # Same fixture receives identical epsilon irrespective of loader order/size.
        identity = (row["stable_window_id"], row["mask_slot"], sample, seed)
        generator = torch.Generator().manual_seed(int(digest(identity)[:15], 16))
        globals_.append(torch.randn(model.global_latent_dim, generator=generator))
        locals_.append(torch.randn(16, model.local_latent_dim, generator=generator))
    return torch.stack(globals_).to(device), torch.stack(locals_).to(device)


def route_output(model, batch, sm, am, route, epsilon=None):
    if route == "A":
        return model(batch, stage="A")
    if route == "B":
        return model(batch, sm, am, stage="B")
    if route == "posterior_sample":
        return model(batch, sm, am, stage="C", epsilon=epsilon)
    if route == "standard_normal":
        return model.infer_from_condition(batch, sm, am, epsilon=epsilon)
    if route == "zero":
        return model.infer_from_condition(batch, sm, am, sample=False)
    if route == "posterior_mean":
        q = model.encode_posterior_distribution(batch)
        c = model.encode_condition(batch, sm, am)
        return model.decode_from_fused_latents(batch, model.fuse_latents(q, c))
    raise ValueError(f"unknown evaluation route {route}")


class Diagnostics:
    def __init__(self, normalization):
        self.normalization = normalization
        self.errors = {d: [] for d in ("state", "action")}
        self.valid = {d: [] for d in self.errors}
        self.hidden = {d: [] for d in self.errors}
        self.windows, self.top, self.traces = [], {d: [] for d in self.errors}, {}
        self.unique_top = {d: {} for d in self.errors}
        self.worst = {d: (-1., None) for d in self.errors}
        self.contacts = {part: {"count": 0, "bce_sum": 0., "tp": 0, "tn": 0, "fp": 0, "fn": 0} for part in PARTS}
        self.overlap = {}

    def add(self, batch, output, sm, am, names, *, export_dir=None):
        ids = batch_ids(batch)
        sm, am = sm.cpu().numpy(), am.cpu().numpy()
        predictions = {"state": output.physical_state.detach().cpu().numpy(), "action": output.action.detach().cpu().numpy()}
        targets = {"state": batch["physical_state"].cpu().numpy(), "action": batch["action"].cpu().numpy()}
        validity = {d: batch["valid_" + d].cpu().numpy().astype(bool) for d in self.errors}
        masks = {"state": sm, "action": am}
        contact_bce = F.binary_cross_entropy_with_logits(output.state_contact_logits, batch["physical_state"][..., 68:], reduction="none").cpu().numpy()
        if not np.isfinite(contact_bce[validity["state"]]).all():
            raise FloatingPointError("non-finite contact evaluation")
        c_pred, c_true = predictions["state"][..., 68:] >= .5, targets["state"][..., 68:] >= .5
        for part in PARTS:
            take = np.broadcast_to(validity["state"][..., None], c_pred.shape).copy()
            if part != "full":
                take &= sm[..., 68:] if part == "masked" else ~sm[..., 68:]
            c = self.contacts[part]
            c["count"] += int(take.sum())
            c["bce_sum"] += float(contact_bce[take].sum(dtype=np.float64))
            for key, condition in (("tp", c_pred & c_true), ("tn", ~c_pred & ~c_true), ("fp", c_pred & ~c_true), ("fn", ~c_pred & c_true)):
                c[key] += int((take & condition).sum())
        for i, identity in enumerate(ids):
            row = {**identity, "mask_name": names[i]}
            row["contact"] = {}
            for part in PARTS:
                take = np.broadcast_to(validity["state"][i, :, None], c_pred[i].shape).copy()
                if part != "full":
                    take &= sm[i, :, 68:] if part == "masked" else ~sm[i, :, 68:]
                row["contact"][part] = {"count": int(take.sum()), "bce_sum": float(contact_bce[i][take].sum(dtype=np.float64)),
                    "correct": int(((c_pred[i] == c_true[i]) & take).sum())}
            trace = {"identity": row.copy(), "domains": {}}
            for domain, width in (("state", 68), ("action", 29)):
                pred, target = predictions[domain][i, :, :width], targets[domain][i, :, :width]
                valid = np.broadcast_to(validity[domain][i, :, None], pred.shape).copy()
                hidden = masks[domain][i, :, :width] & valid
                error = pred - target
                if not np.isfinite(error[valid]).all():
                    raise FloatingPointError("non-finite evaluation prediction")
                self.errors[domain].append(error)
                self.valid[domain].append(valid)
                self.hidden[domain].append(hidden)
                row[domain] = {part: moments(error[take]) for part, take in (("full", valid), ("masked", hidden), ("visible", valid & ~hidden))}
                mean, std = self.normalization[domain]
                mean, std = mean[:width], std[:width]
                trace["domains"][domain] = {"prediction": pred.tolist(), "target": target.tolist(),
                    "physical_prediction": (pred * std + mean).tolist(), "physical_target": (target * std + mean).tolist(),
                    "mask": hidden.tolist(), "valid": validity[domain][i].tolist()}
                magnitude = np.where(valid, np.abs(error), -1.)
                candidates = np.argsort(magnitude.ravel())[-100:][::-1]
                for flat in candidates:
                    t, f = np.unravel_index(flat, magnitude.shape)
                    if not valid[t, f]:
                        continue
                    entry = {**identity, "domain": domain, "mask_name": names[i], "relative_frame": int(t),
                        "absolute_frame": int(identity["window_start"]) + int(t), "chunk": min(int(t) // 4, 15),
                        "chunk_position": int(t) - min(int(t) // 4, 15) * 4, "feature_index": int(f),
                        "feature_name": STATE_NAMES[f] if domain == "state" else f"action_{f}",
                        "unit": UNITS[f] if domain == "state" else "rad", "masked": bool(hidden[t, f]),
                        "prediction": float(pred[t, f]), "target": float(target[t, f]), "signed_error": float(error[t, f]),
                        "abs_error": float(magnitude[t, f]), "mean": float(mean[f]), "std": float(std[f]),
                        "physical_prediction": float(pred[t, f] * std[f] + mean[f]),
                        "physical_target": float(target[t, f] * std[f] + mean[f]),
                        "physical_error": float(error[t, f] * std[f]),
                        "physical_abs_error": float(abs(error[t, f] * std[f]))}
                    entry["raw_element_id"] = digest((identity["episode_ref"], entry["absolute_frame"], domain, int(f)))
                    self.top[domain].append(entry)
                    previous = self.unique_top[domain].get(entry["raw_element_id"])
                    if previous is None or entry["abs_error"] > previous["abs_error"]:
                        self.unique_top[domain][entry["raw_element_id"]] = entry
                self.top[domain] = sorted(self.top[domain], key=lambda x: x["abs_error"], reverse=True)[:100]
                unique = sorted(self.unique_top[domain].values(), key=lambda x: x["abs_error"], reverse=True)[:100]
                self.unique_top[domain] = {entry["raw_element_id"]: entry for entry in unique}
                if row[domain]["full"]["max_abs"] is not None and row[domain]["full"]["max_abs"] > self.worst[domain][0]:
                    self.worst[domain] = (row[domain]["full"]["max_abs"], str(identity["fixture_index"]))
                # Same raw frame, same family; different window contexts, not replicates.
                for t in np.flatnonzero(validity[domain][i]):
                    key = (domain, identity["episode_ref"], identity["window_start"] + int(t), names[i])
                    existing = self.overlap.get(key)
                    if existing is None:
                        self.overlap[key] = [pred[t].copy(), pred[t].copy(), 1]
                    else:
                        existing[0] = np.minimum(existing[0], pred[t])
                        existing[1] = np.maximum(existing[1], pred[t])
                        existing[2] += 1
            key = str(identity["fixture_index"])
            trace["contact"] = {"prediction_probability": predictions["state"][i, :, 68:].tolist(),
                "target": targets["state"][i, :, 68:].tolist(), "mask": sm[i, :, 68:].tolist(),
                "valid": validity["state"][i].tolist()}
            self.traces[key] = trace
            keep = {"0", "1"} | {value[1] for value in self.worst.values()}
            self.traces = {k: v for k, v in self.traces.items() if k in keep}
            if export_dir is not None:
                atomic_write_json(export_dir / f"fixture_{key}.json", trace)
            self.windows.append(row)

    def finish(self):
        result = {"partitions": {}, "features": {}, "time": {}, "chunks": {}, "chunk_positions": {},
                  "top_elements": self.top, "windows": self.windows, "mask_families": {}, "traces": self.traces}
        for part in PARTS:
            contact = self.contacts[part]
            n = contact["count"]
            result["partitions"][part] = {"contact": {**contact, "bce": contact["bce_sum"] / n if n else None,
                "accuracy": (contact["tp"] + contact["tn"]) / n if n else None}}
        for domain in self.errors:
            errors = np.stack(self.errors[domain])
            valid, hidden = np.stack(self.valid[domain]), np.stack(self.hidden[domain])
            for part, take in (("full", valid), ("masked", hidden), ("visible", valid & ~hidden)):
                result["partitions"][part][domain] = stats(errors[take])
            result["features"][domain] = []
            for f in range(errors.shape[2]):
                mean, std = self.normalization[domain]
                result["features"][domain].append({"index": f,
                    "name": STATE_NAMES[f] if domain == "state" else f"action_{f}",
                    "unit": UNITS[f] if domain == "state" else "rad", "mean": float(mean[f]), "std": float(std[f]),
                    **{part: stats(errors[:, :, f][take[:, :, f]]) for part, take in
                       (("full", valid), ("masked", hidden), ("visible", valid & ~hidden))},
                    "physical": stats(errors[:, :, f][valid[:, :, f]] * std[f])})
            result["time"][domain] = [{"t": t, **{part: moments(errors[:, t][take[:, t]]) for part, take in
                (("full", valid), ("masked", hidden), ("visible", valid & ~hidden))}} for t in range(errors.shape[1])]
            result["chunks"][domain] = [{"chunk": c, **moments(errors[:, 4*c:(4*c+4 if c < 15 else 65)][valid[:, 4*c:(4*c+4 if c < 15 else 65)]])} for c in range(16)]
            positions = np.arange(errors.shape[1]) - np.minimum(np.arange(errors.shape[1]) // 4, 15) * 4
            result["chunk_positions"][domain] = [{"position": p, **moments(errors[:, positions == p][valid[:, positions == p]])} for p in range(5)]
            result.setdefault("worst_windows", {})[domain] = {
                "by_rmse": sorted(self.windows, key=lambda row: row[domain]["full"]["rmse"] or 0, reverse=True)[:10],
                "by_max_abs": sorted(self.windows, key=lambda row: row[domain]["full"]["max_abs"] or 0, reverse=True)[:10]}
            result.setdefault("top_unique_raw_elements", {})[domain] = list(self.unique_top[domain].values())
        for name in sorted({row["mask_name"] for row in self.windows}):
            rows = [row for row in self.windows if row["mask_name"] == name]
            result["mask_families"][name] = {part: {d: merge_moments(row[d][part] for row in rows)
                for d in self.errors} for part in PARTS}
            result["mask_families"][name]["contacts"] = {}
            for part in PARTS:
                count = sum(row["contact"][part]["count"] for row in rows)
                result["mask_families"][name]["contacts"][part] = {"count": count,
                    "bce": sum(row["contact"][part]["bce_sum"] for row in rows) / count if count else None,
                    "accuracy": sum(row["contact"][part]["correct"] for row in rows) / count if count else None}
        result["overlap_consistency"] = {d: {"raw_frames_with_multiple_windows": sum(v[2] > 1 for k, v in self.overlap.items() if k[0] == d),
            "max_prediction_range": max((float((v[1] - v[0]).max()) for k, v in self.overlap.items() if k[0] == d and v[2] > 1), default=None)} for d in self.errors}
        for trace in self.traces.values():
            state = trace["domains"]["state"]
            pred, target = np.array(state["physical_prediction"]), np.array(state["physical_target"])
            count = sum(state["valid"])
            velocity = []
            for f in range(29, 58):
                # Same central target range for all lags; official metrics never shifted.
                lags = {str(lag): float(np.mean((pred[2+lag:count-2+lag, f] - target[2:count-2, f])**2))**.5
                        for lag in range(-2, 3)} if count > 4 else {}
                velocity.append({"feature": STATE_NAMES[f], "unit": "rad/s", "lag_rmse_diagnostic_only": lags,
                    "peak_absolute_target": float(np.abs(target[:count, f]).max()),
                    "peak_absolute_prediction": float(np.abs(pred[:count, f]).max()),
                    "peak_amplitude_error": float(np.abs(pred[:count, f]).max() - np.abs(target[:count, f]).max())})
            trace["velocity_diagnostics"] = velocity
        return result


def ensemble_scores(outputs, batch, sm, am):
    pred = torch.cat((torch.stack([o.physical_state[..., :68] for o in outputs]).flatten(2),
                      torch.stack([o.action for o in outputs]).flatten(2)), dim=2).double()
    truth = torch.cat((batch["physical_state"][..., :68].flatten(1), batch["action"].flatten(1)), dim=1).double()
    valid = torch.cat((batch["valid_state"][..., None].expand(-1, -1, 68).flatten(1),
                       batch["valid_action"][..., None].expand(-1, -1, 29).flatten(1)), dim=1).bool()
    hidden = torch.cat((sm[..., :68].flatten(1), am.flatten(1)), dim=1) & valid
    rows = []
    for i in range(len(truth)):
        take = hidden[i]
        n = int(take.sum())
        row = {"masked_count": n, "energy_score": None, "expected_masked_mse": None, "best_of_k_mse": None, "variance": None,
               "expected_full_mse": float((pred[:, i, valid[i]] - truth[i, valid[i]]).square().mean()), "partitions": {}}
        if n:
            x, y = pred[:, i, take], truth[i, take]
            norms = ((x - y).square().mean(1)).sqrt()
            # Empirical energy score E||X-y|| - 1/2 E||X-X'||, norm / sqrt(D).
            pair = torch.cdist(x, x) / n**.5
            row.update(energy_score=float(norms.mean() - .5 * pair.mean()), expected_masked_mse=float(norms.square().mean()),
                       best_of_k_mse=float(norms.square().min()), variance=float(x.var(0, unbiased=False).mean()))
        for domain, start, stop in (("state", 0, 65*68), ("action", 65*68, pred.shape[-1])):
            row["partitions"][domain] = {}
            for part, mask in (("full", valid), ("masked", hidden), ("visible", valid & ~hidden)):
                domain_mask = mask[i, start:stop]
                errors = pred[:, i, start:stop][:, domain_mask] - truth[i, start:stop][domain_mask]
                row["partitions"][domain][part] = {"count": int(errors.numel()), "sse": float(errors.square().sum()),
                    "sae": float(errors.abs().sum()), "sample_variance_sum": float(pred[:, i, start:stop][:, domain_mask].var(0, unbiased=False).sum()) if errors.numel() else 0.,
                    "target_count": int(domain_mask.sum())}
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate(model, loader, device, normalization, *, route, seed, held_out=False, samples=8, output_dir=None, export_all=False, heartbeat=None, reference_mask_seed=None):
    was_training = model.training
    with isolated_rng():
        model.eval()
        diagnostics = Diagnostics(normalization)
        ensemble = []
        overlap_flags = []
        try:
            for ordinal, cpu in enumerate(loader):
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in cpu.items()}
                if route == "A":
                    sm = torch.zeros_like(batch["physical_state"], dtype=torch.bool)
                    am = torch.zeros_like(batch["action"], dtype=torch.bool)
                    names = ["unconditional_posterior"] * len(sm)
                else:
                    sm, am, names = make_physical_masks(cpu, seed, held_out=held_out)
                    if held_out and reference_mask_seed is not None:
                        reference_sm, reference_am, _ = make_physical_masks(cpu, reference_mask_seed)
                        overlap_flags.extend((sm.eq(reference_sm).flatten(1).all(1) & am.eq(reference_am).flatten(1).all(1)).tolist())
                    sm, am = sm.to(device), am.to(device)
                count = samples if route in {"posterior_sample", "standard_normal"} else 1
                outputs = [route_output(model, batch, sm, am, route, epsilon_for(model, cpu, k, seed, device)) for k in range(count)]
                diagnostics.add(batch, outputs[0], sm, am, names, export_dir=(output_dir / "all_predictions") if export_all and output_dir else None)
                if route != "A":
                    ensemble.extend({**identity, "mask_name": name, **score} for identity, name, score in
                        zip(batch_ids(cpu), names, ensemble_scores(outputs, batch, sm, am)))
                if heartbeat:
                    heartbeat(ordinal + 1)
            details = diagnostics.finish()
        finally:
            model.train(was_training)
    details.update(route=route, samples=samples if route in {"posterior_sample", "standard_normal"} else 1,
                   distribution_diagnostics_draw=0, ensemble=ensemble, held_out=held_out, epsilon_seed=seed)
    if overlap_flags:
        details["heldout_coordinate_overlap"] = {"reference_seed": reference_mask_seed, "seen": sum(overlap_flags),
            "new": len(overlap_flags) - sum(overlap_flags), "note": "full_action and some sampled gaps can equal fixed fixtures"}
        for row, same in zip(details["windows"], overlap_flags):
            row["same_coordinates_as_fixed"] = same
    full = details["partitions"]["full"]
    terms = [full["state"]["mse"], full["action"]["mse"], full["contact"]["bce"]]
    summary = {"route": route, "total_loss": float(np.mean([v for v in terms if v is not None])),
        "state_rmse": full["state"]["rmse"], "action_rmse": full["action"]["rmse"], "contact_bce": full["contact"]["bce"],
        "max_abs": max(full[d]["max_abs"] or 0 for d in ("state", "action")),
        "state_max_abs": full["state"]["max_abs"], "action_max_abs": full["action"]["max_abs"],
        "state_abs_p99": full["state"]["p99"], "action_abs_p99": full["action"]["p99"],
        "partitions": details["partitions"], "mask_families": details["mask_families"],
        "diagnostic_draw": 0, "samples": details["samples"]}
    if ensemble:
        families = []
        for family in details["mask_families"].values():
            terms = [family["masked"][d]["mse"] for d in ("state", "action") if family["masked"][d]["count"]]
            if terms:
                families.append(float(np.mean(terms)))
        summary["masked_mse"] = float(np.mean(families))
        available = [r for r in ensemble if r["masked_count"]]
        for key in ("energy_score", "expected_masked_mse", "best_of_k_mse", "variance", "expected_full_mse"):
            summary[key] = float(np.mean([r[key] for r in available])) if available else None
        summary["expected_single_sample"] = {}
        for domain in ("state", "action"):
            summary["expected_single_sample"][domain] = {}
            for part in PARTS:
                parts = [r["partitions"][domain][part] for r in ensemble]
                n = sum(r["count"] for r in parts)
                count = sum(r["target_count"] for r in parts)
                summary["expected_single_sample"][domain][part] = {"sample_target_count": n, "target_count": count,
                    "mse": sum(r["sse"] for r in parts) / n if n else None,
                    "rmse": (sum(r["sse"] for r in parts) / n)**.5 if n else None,
                    "mae": sum(r["sae"] for r in parts) / n if n else None,
                    "sample_variance": sum(r["sample_variance_sum"] for r in parts) / count if count else None}
        summary["energy_by_family"] = {name: float(np.mean([r["energy_score"] for r in available if r["mask_name"] == name]))
                                      for name in sorted({r["mask_name"] for r in available})}
    summary["selection_score"] = summary["total_loss"] if route == "A" else (summary["energy_score"] if route == "standard_normal" else summary["masked_mse"])
    if overlap_flags:
        summary["heldout_coordinate_overlap"] = details["heldout_coordinate_overlap"]
        summary["coordinate_groups"] = {}
        for label, same in (("seen", True), ("new", False)):
            selected = [r for r, flag in zip(details["windows"], overlap_flags) if flag == same]
            energy = [r["energy_score"] for r, flag in zip(ensemble, overlap_flags) if flag == same and r["energy_score"] is not None]
            summary["coordinate_groups"][label] = {"fixture_count": len(selected),
                "masked": {d: merge_moments(r[d]["masked"] for r in selected) for d in ("state", "action")},
                "energy_score": float(np.mean(energy)) if energy else None}
    if output_dir:
        atomic_write_json(output_dir / "diagnostics.json", details)
        atomic_write_json(output_dir / "summary.json", summary)
        render_traces(output_dir, details)
        summary["diagnostics_path"] = str(output_dir / "diagnostics.json")
    return summary, details


def render_traces(directory, details):
    """Signed linear-axis traces: negative velocities must not use a log plot."""
    import html
    from .util import atomic_write_text
    for identity, trace in details["traces"].items():
        for domain in ("state", "action"):
            candidates = [entry for entry in details["top_elements"][domain] if str(entry["fixture_index"]) == identity]
            features = list(dict.fromkeys([entry["feature_index"] for entry in candidates[:5]] + ([29, 55, 56, 57] if domain == "state" else [0])))[:5]
            values = trace["domains"][domain]
            for space, prefix in (("normalized", ""), ("physical", "physical_")):
                pred, truth = np.asarray(values[prefix + "prediction"]), np.asarray(values[prefix + "target"])
                valid = np.asarray(values["valid"])
                content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{len(features)*180+40}"><rect width="100%" height="100%" fill="white"/>']
                content.append(f'<text x="20" y="22">fixture {identity}, {domain}, {space}; blue target / red prediction; grey hidden</text>')
                for panel, f in enumerate(features):
                    a, b = truth[valid, f], pred[valid, f]
                    low, high = min(a.min(), b.min()), max(a.max(), b.max())
                    span = max(float(high-low), 1e-8)
                    y0 = 50 + panel*180
                    name = STATE_NAMES[f] if domain == "state" else f"action_{f}"
                    content.append(f'<text x="15" y="{y0}">{html.escape(name)} [{low:.5g}, {high:.5g}]</text>')
                    for t, hidden in enumerate(np.asarray(values["mask"])[valid, f]):
                        if hidden:
                            content.append(f'<rect x="{60+t*900/max(len(a)-1,1):.2f}" y="{y0+8}" width="{900/max(len(a)-1,1):.2f}" height="130" fill="#eeeeee"/>')
                    for series, color in ((a, "#08519c"), (b, "#cb181d")):
                        points = ' '.join(f'{60+t*900/max(len(a)-1,1):.2f},{y0+138-130*(v-low)/span:.2f}' for t, v in enumerate(series))
                        content.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{points}"/>')
                content.append('</svg>')
                atomic_write_text(directory / f"trace_{identity}_{domain}_{space}.svg", ''.join(content))


@torch.no_grad()
def ablations(model, batch, sm, am, stage, donors):
    """Fixed cross-window donor map supplied by caller, never a batch flip."""
    condition = None if stage == "A" else model.encode_condition(batch, sm, am)
    q = model.encode_posterior_distribution(batch) if stage in {"A", "C"} else None
    g, l = (q.global_mean, q.local_mean) if q else (condition.global_latent, condition.local_latents)
    rows = []
    for intervention in ("baseline", "global_zero", "global_swap", "local_zero", "local_swap", "memory_zero", "memory_swap", "film_zero", "film_swap"):
        gg, ll = g, l
        if intervention.startswith("global_"):
            gg = torch.zeros_like(g) if intervention.endswith("zero") else g[donors]
        if intervention.startswith("local_"):
            ll = torch.zeros_like(l) if intervention.endswith("zero") else l[donors]
        cc = condition
        if stage == "B":
            cc = replace(condition, global_latent=gg, local_latents=ll)
            fused = model.fuse_condition_only(cc)
        else:
            fused = model.fuse_latents((gg, ll), cc)
        if intervention.startswith("memory_") and cc is not None:
            fused = replace(fused, condition_memory=torch.zeros_like(cc.memory) if intervention.endswith("zero") else cc.memory[donors])
        hooks = []
        if intervention == "film_zero":
            hooks = [head.register_forward_hook(lambda module, args, value: torch.zeros_like(value)) for head in model.film_heads]
        elif intervention == "film_swap":
            hooks = [head.register_forward_hook(lambda module, args, value: value[donors]) for head in model.film_heads]
        try:
            output = model.decode_from_fused_latents(batch, fused)
        finally:
            for hook in hooks:
                hook.remove()
        error = []
        for d, pred, target, valid in (("state", output.physical_state[..., :68], batch["physical_state"][..., :68], batch["valid_state"]),
                                      ("action", output.action, batch["action"], batch["valid_action"])):
            values = (pred - target).square()[valid]
            error.append({"domain": d, "mse": float(values.mean())})
        rows.append({"intervention": intervention, "errors": error, "applicable": not (stage == "A" and intervention.startswith("memory_"))})
    return {"rows": rows, "recipients": batch_ids(batch), "donor_indices": donors.tolist(),
            "interpretation": "dependency only, not representation sufficiency; C uses posterior mean here"}
