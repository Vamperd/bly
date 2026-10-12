"""Kinematic triptychs for replay65; never runs or advances a physics simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from render_mujoco_trajectory import (G1_ISAACLAB_JOINT_NAMES, build_mujoco_qpos,
    load_trajectory, prepare_runtime_xml, write_json_atomic)
from render_h50a_exact_action_replays import _validate_joint_order, _writer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scored_frames(values, horizon, offset):
    """Window-length inputs pass through; full replays (prefix + window) are sliced."""
    if len(values) == horizon + 1:
        return values
    if len(values) == offset + horizon + 1:
        return values[offset:offset + horizon + 1]
    raise ValueError(f"trajectory has {len(values)} frames; expected {horizon + 1} or {offset + horizon + 1}")


def draw_timeline(cv2, frame, hidden, t, y=457):
    width = frame.shape[1] - 20
    count = len(hidden)
    for j in range(count):
        x0 = 10 + j * width // count
        x1 = max(x0, 10 + (j + 1) * width // count - 1)
        cv2.rectangle(frame, (x0, y), (x1, y + 12), (50, 50, 220) if hidden[j] else (70, 100, 70), -1)
    x = 10 + t * width // count
    cv2.line(frame, (x, y - 4), (x, y + 18), (255, 255, 255), 2)


def video_jobs(window, meta):
    truth = window/"data/source.trajectory.pkl"
    jobs=[]
    baseline=window/"simulations/baseline/data/replay"
    if (baseline/"000001.trajectory.pkl").exists():
        jobs.append(("original_repeatability",[truth,baseline/"000000.trajectory.pkl",baseline/"000001.trajectory.pkl"],
                     ["Source","Original Action repeat 1","Original Action repeat 2"],None))
    for entry in meta["entries"]:
        if not entry["representative"]:
            continue
        name=entry["mask"]; data=window/"data"/name
        jobs.append((name+"_state",[truth,data/"truth_integrated.trajectory.pkl",data/"predicted_integrated.trajectory.pkl"],
            ["Source pose","Truth State integration","Model completed State integration"],data/"prediction.npz"))
        model=window/"simulations"/name/"data/replay/000000.trajectory.pkl"
        if model.exists() and (baseline/"000000.trajectory.pkl").exists():
            jobs.append((name+"_action",[truth,baseline/"000000.trajectory.pkl",model],
                ["Source pose","Original Action replay","Model Action replay"],data/"prediction.npz"))
    return jobs


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True); p.add_argument("--model",type=Path,required=True)
    p.add_argument("--gl",default="egl",choices=("egl","osmesa","glfw"))
    args=p.parse_args(argv)
    os.environ["MUJOCO_GL"]=args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco
    run=args.run.resolve()
    root=json.loads((run/"manifests/replay65.json").read_text())
    render_manifest=run/"manifests/replay65_render.json"
    previous=json.loads(render_manifest.read_text())["videos"] if render_manifest.exists() else []
    previous={v["path"]:v for v in previous}
    outputs=[]
    for relative in root["windows"]:
        window=run/relative
        meta=json.loads((window/"manifests/replay65.json").read_text())
        report=json.loads((window/"manifests/replay65_report.json").read_text())
        with np.load(window/"data/source.replay.npz",allow_pickle=False) as archive:
            if tuple(archive["joint_names"].tolist())!=G1_ISAACLAB_JOINT_NAMES:
                raise ValueError("recorded Isaac joint order does not match the renderer permutation")
        xml=window/"manifests/replay65_render.xml"
        xml_details=prepare_runtime_xml(args.model.resolve(),xml,640,480)
        model=mujoco.MjModel.from_xml_path(str(xml)); _validate_joint_order(model)
        if model.nq!=36:
            raise ValueError("expected G1 MuJoCo model nq=36")
        data=mujoco.MjData(model); renderer=mujoco.Renderer(model,height=480,width=640)
        camera=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
        camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth=135; camera.elevation=-15; camera.distance=3.5
        try:
            horizon=int(meta.get("horizon_steps",64)); offset=int(meta.get("score_offset",0)); frames_total=horizon+1
            for name,paths,labels,prediction in video_jobs(window,meta):
                trajectories=[load_trajectory(path) for path in paths]
                qposes=[scored_frames(build_mujoco_qpos(t),horizon,offset) for t in trajectories]
                if any(q.shape!=(frames_total,36) or not np.isfinite(q).all() for q in qposes) or any(not np.isclose(t["fps"],50) for t in trajectories):
                    raise ValueError(f"video requires {frames_total} finite frames, 50Hz; partial replay cannot masquerade as complete")
                hidden=np.zeros(frames_total,bool)
                if prediction:
                    with np.load(prediction,allow_pickle=False) as values:
                        hidden=values["state_mask"].any(axis=-1) if name.endswith("_state") else np.r_[values["action_mask"].any(axis=-1),False]
                output=window/"videos"/(name+".mp4")
                relative_output=str(output.relative_to(run))
                input_hashes={str(path.relative_to(run)):sha(path) for path in paths}
                old=previous.get(relative_output)
                if output.exists():
                    if old and old["sha256"]==sha(output) and old["input_hashes"]==input_hashes and old["baseline_banner"]==report["banner"]:
                        outputs.append(old)
                        continue
                    raise FileExistsError(f"unverified/changed video is not overwritten: {output}")
                writer=_writer(imageio,output,50.)
                try:
                    for t in range(frames_total):
                        panels=[]
                        for qpos,label in zip(qposes,labels):
                            data.qpos[:]=qpos[t]; data.qvel[:]=0
                            mujoco.mj_forward(model,data)
                            camera.lookat[:]=qposes[0][t,:3]
                            renderer.update_scene(data,camera=camera)
                            frame=renderer.render().copy()
                            cv2.rectangle(frame,(0,0),(640,84),(25,25,25),-1)
                            banner=report["banner"]
                            if name.endswith("_state"):
                                banner="KINEMATIC ONLY / NOT DYNAMICS VALIDATION"
                            elif name.endswith("_action"):
                                mask=name[:-7]
                                if report.get("checks",{}).get(mask,{}).get("model_physical_quality"):
                                    banner="MODEL_INITIALIZATION_INVALID / UNDETERMINED"
                            cv2.putText(frame,label,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.52,(245,245,245),1)
                            cv2.putText(frame,banner,(10,47),cv2.FONT_HERSHEY_SIMPLEX,.40,(80,190,255),1)
                            episode_frame=int(meta.get("window_start",meta["window"]["window_start"]))+t
                            cv2.putText(frame,f"{meta['route']} | {name} | frame {t}/{horizon} (ep {episode_frame}) | 50 Hz",(10,71),cv2.FONT_HERSHEY_SIMPLEX,.43,(230,230,230),1)
                            draw_timeline(cv2,frame,hidden,t)
                            panels.append(frame)
                        writer.append_data(np.concatenate(panels,axis=1))
                finally:
                    writer.close()
                frames,seconds=count_frames_and_secs(str(output))
                if frames!=frames_total or output.stat().st_size<=0:
                    raise RuntimeError("encoded frame count mismatch")
                outputs.append({"path":str(output.relative_to(run)),"sha256":sha(output),"frames":frames,"seconds":seconds,
                    "input_hashes":input_hashes,"baseline_banner":report["banner"],"xml":xml_details})
                write_json_atomic(render_manifest,{"version":"65-token-replay-render-v1","videos":list({v["path"]:v for v in [*previous.values(),*outputs]}.values()),"rendering":True,"quality_pass":None})
        finally:
            renderer.close()
    write_json_atomic(run/"manifests/replay65_render.json",{"version":"65-token-replay-render-v1","videos":outputs,
        "camera":"shared recorded root track; no predicted root correction","quality_pass":None})
    print(f"Rendered {len(outputs)} videos under {run}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
