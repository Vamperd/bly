from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import default_collate
from cvae_sa import replay65 as replay
from cvae_sa.models import build_model
from cvae_sa.cvae_protocol import digest, sample_identity
from cvae_sa.posterior_t64_protocol import make_physical_masks
from test_posterior_hierarchical_standard_cvae import config, batch


def fixture(slot=7):
    value=batch(1)
    sample={k:v[0] for k,v in value.items()}
    sample.update(stable_window_id=digest(sample_identity(sample)),window_index=0,fixture_index=slot,mask_slot=slot)
    return default_collate([sample])


def trajectory(offset=0.,n=64):
    states=np.zeros((n+1,70),np.float32); states[:,66]=-1; states[:,67]=1
    quat=np.zeros((n+1,4),np.float32); quat[:,0]=1
    return dict(physics_state_v3=states,physical_state=states,joint_pos=np.zeros((n+1,29),np.float32)+offset,
        joint_vel=np.zeros((n+1,29),np.float32),root_pos=np.tile([0.,0.,1.],(n+1,1)).astype(np.float32),root_quat=quat,
        root_lin_vel=np.zeros((n+1,3),np.float32),root_ang_vel=np.zeros((n+1,3),np.float32),
        body_pos=np.zeros((n+1,3,3),np.float32),raw_action=np.zeros((n,29),np.float32))


VALID_READBACK={"payload_sha256":"p","initialization_file_sha256":"f","application_complete":True,"errors":{
    k:0. for k in ("joint_pos_max_abs_rad","joint_vel_max_abs_rad_s","root_pos_max_abs_m","root_orientation_error_deg",
    "root_lin_vel_max_abs_m_s","root_ang_vel_max_abs_rad_s","body_pos_max_abs_m","runtime_context_max_abs",
    "previous_raw_action_max_abs","current_raw_action_max_abs","previous_processed_action_max_abs","initial_joint_target_max_abs")}}
VALID_RUNTIME={"contract_verified":True,"midrun_reset":False}
MAPPING=dict(nominal_default_joint_pos=np.zeros(29,np.float32),action_scale=np.ones(29,np.float32),
    action_offset=np.zeros(29,np.float32),action_clip=np.empty(0,np.float32),wrapper_action_clip=np.float32(np.nan))


def write_worker(child,trajectories,valid=True):
    """Fake replay-action-mask worker output, sealed like simulate_group."""
    (child/"manifests").mkdir(parents=True); (child/"data/replay").mkdir(parents=True)
    steps=trajectories[0]["raw_action"].shape[0]
    replay._write_npz(child/"data/raw_actions.npz",raw_actions=np.stack([t["raw_action"] for t in trajectories],axis=1))
    for i,value in enumerate(trajectories):
        replay._write_npz(child/f"data/replay/{i:06d}.replay.npz",**value,processed_action=value["raw_action"],
                          joint_names=np.asarray([f"joint_{j}" for j in range(29)]))
        replay._write_trajectory(child/f"data/replay/{i:06d}.trajectory.pkl",joint_pos=value["joint_pos"],
                                 root_pos=value["root_pos"],root_quat=value["root_quat"],fps=50.)
        (child/f"manifests/exact_initialization_readback_{i:06d}.json").write_text(json.dumps(VALID_READBACK if valid else {"errors":{}}))
        (child/f"data/replay/{i:06d}.runtime.json").write_text(json.dumps({**VALID_RUNTIME,"control_steps":steps} if valid else {}))
    hashes={str(p.relative_to(child)).replace("\\","/"):replay.file_sha256(p) for p in child.rglob("*") if p.is_file()}
    (child/"manifests/replay65_worker_complete.json").write_text(json.dumps({"hashes":hashes}))


class Replay65Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hidden_truth_never_changes_B_or_C_and_no_posterior(self):
        for route in ("B","C"):
            model=build_model(config()).eval(); cpu=fixture()
            sm,am,_=make_physical_masks(cpu,42)
            changed={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
            changed["physical_state"][sm]=9999; changed["action"][am]=-9999
            with patch.object(model,"encode_posterior_distribution",side_effect=AssertionError("posterior called")):
                a=replay.predict(model,cpu,route,42,10)
                b=replay.predict(model,changed,route,42,10)
            np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[1],b[1])
            self.assertEqual(a[0].shape,(65,70)); self.assertEqual(a[1].shape,(64,29))
            self.assertEqual(model.local_chunk_ids(torch.tensor([60,64])).tolist(),[15,15])

    def test_A_no_condition_and_mask_invariant(self):
        model=build_model(config()).eval(); cpu=fixture()
        with patch.object(model,"encode_condition",side_effect=AssertionError("condition called")):
            a=replay.predict(model,cpu,"A",42,10)
            b=replay.predict(model,cpu,"A",81,10)
        np.testing.assert_array_equal(a[0],b[0])

    def test_sampling_fixed_and_does_not_consume_training_rng(self):
        model=build_model(config()).eval(); cpu=fixture()
        before=torch.get_rng_state().clone()
        a=replay.predict(model,cpu,"C",42,10)
        torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
        b=replay.predict(model,cpu,"C",42,11)
        self.assertFalse(np.array_equal(a[5][0].numpy(),b[5][0].numpy()))

    def test_visible_raw_bits_preserved_with_clipping_aliases(self):
        raw=np.full((64,29),3.,np.float32); scale=np.ones(29,np.float32); nominal=np.zeros(29,np.float32)
        clip=np.tile([-1.,1.],(29,1)).astype(np.float32); truth=np.ones_like(raw)
        pred=np.full_like(raw,-5.); mask=np.zeros_like(raw,bool); mask[1:3,5]=True
        executed,achieved,info=replay.execution_actions(raw,truth,pred,mask,nominal,scale,nominal,clip)
        np.testing.assert_array_equal(executed[~mask],raw[~mask])
        self.assertEqual(info["saturated_elements"],2)
        np.testing.assert_array_equal(achieved[mask],-np.ones(2))
        with self.assertRaisesRegex(ValueError,"mapping mismatch"):
            replay.execution_actions(raw,np.zeros_like(raw),pred,mask,nominal,scale,nominal,clip)

    def test_nonfinite_and_terminal_action_rejected(self):
        raw=np.zeros((65,29),np.float32)
        with self.assertRaises(ValueError):
            replay.execution_actions(raw,raw,raw,raw.astype(bool),np.zeros(29),np.ones(29),np.zeros(29))
        with self.assertRaises(ValueError):
            replay.finite("bad",[float("nan")])

    def test_full_masked_visible_zero_target(self):
        s=np.zeros((65,70)); a=np.zeros((64,29)); sm=np.zeros_like(s,bool); am=np.ones_like(a,bool)
        metrics=replay.offline_metrics(s,a,s+2,a+3,sm,am)
        self.assertEqual(metrics["masked"]["state"]["count"],0)
        self.assertIsNone(metrics["masked"]["state"]["rmse"])
        self.assertEqual(metrics["masked"]["action"]["rmse"],3)

    def test_legacy_signature_configuration(self):
        cfg=replay.checkpoint_config({"model_signature":config()})
        self.assertEqual(cfg["data"]["window_transitions"],64)
        self.assertEqual(cfg["model"],config())

    def test_baseline_fail_is_not_model_fail_and_render_job_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"manifests").mkdir(); (root/"data").mkdir()
            meta={"window":{"window_start":100},"initialization":{"payload_sha256":"p","file_sha256":"f"},
                "entries":[{"mask":"full_action","action_model_replay":True,"representative":True}]}
            (root/"manifests/replay65.json").write_text(json.dumps(meta))
            replay._write_npz(root/"data/source.replay.npz",**trajectory())
            replay._write_npz(root/"data/exact_initialization.npz",action_scale=np.ones(29),action_offset=np.zeros(29),action_clip=np.empty(0),wrapper_action_clip=np.float32(np.nan))
            for tag,n in (("baseline",2),("full_action",1)):
                write_worker(root/"simulations"/tag,[trajectory(.3)]*n,valid=False)
            (root/"data/full_action").mkdir()
            replay._write_npz(root/"data/full_action/prediction.npz",predicted_state=trajectory()["physical_state"],completed_state=trajectory()["physical_state"])
            report=replay.report_window(root)
            self.assertFalse(report["baseline_valid"])
            self.assertIsNone(report["quality_pass"])
            self.assertTrue(report["execution_complete"])
            self.assertIn("BASELINE_INVALID",report["banner"])
            self.assertEqual(report["comparisons"]["original_repeatability"]["joint_position_rmse_rad"],0)

    def test_horizon_chunk_helpers(self):
        self.assertEqual(replay.resolve_window_start(64,300,192),(64,None))
        start,shift=replay.resolve_window_start(200,300,192)
        self.assertEqual(start,64); self.assertIn("shifted",shift)
        with self.assertRaises(ValueError):
            replay.resolve_window_start(0,100,192)
        with self.assertRaises(ValueError):
            replay.resolve_window_start(0,300,100)
        ds=SimpleNamespace(refs=[SimpleNamespace(episode_index=e,fixed_start=s) for e,s in
                                 ((0,0),(0,64),(0,128),(0,192),(1,64))])
        self.assertEqual(replay.chunk_indices(ds,1,64,3),[1,2,3])
        with self.assertRaisesRegex(ValueError,"no fixed T64"):
            replay.chunk_indices(ds,4,64,2)
        parts=[np.full((65,2),k,np.float32) for k in range(3)]
        stitched=replay.stitch_states(parts)
        self.assertEqual(stitched.shape,(193,2))
        self.assertEqual(stitched[64,0],0); self.assertEqual(stitched[65,0],1); self.assertEqual(stitched[-1,0],2)

    def test_state_only_continuation_starts_after_selected_window(self):
        enabled = SimpleNamespace(state_only_continuation=True)
        disabled = SimpleNamespace(state_only_continuation=False)
        self.assertFalse(replay.state_only_continuation(enabled, 0))
        self.assertTrue(replay.state_only_continuation(enabled, 1))
        self.assertFalse(replay.state_only_continuation(disabled, 1))

    def test_execution_actions_whole_chunks(self):
        raw=np.zeros((192,29),np.float32); one=np.ones(29,np.float32); zero=np.zeros(29,np.float32)
        mask=np.zeros_like(raw,bool); mask[130:140]=True
        executed,_,info=replay.execution_actions(raw,raw,raw+.5,mask,zero,one,zero)
        self.assertEqual(executed.shape,(192,29)); self.assertEqual(info["replaced_elements"],290)
        np.testing.assert_array_equal(executed[~mask],0)
        with self.assertRaisesRegex(ValueError,"whole"):
            replay.execution_actions(raw[:100],raw[:100],raw[:100],mask[:100],zero,one,zero)

    def test_recapture_report_scores_window_after_prefix_with_absolute_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"manifests").mkdir(); (root/"data/full_action").mkdir(parents=True)
            meta={"window":{"window_start":64},"window_start":64,"horizon_steps":64,"score_offset":64,"replay_steps":128,
                "source_mode":"recapture","source":{"deterministic":True},
                "initialization":{"payload_sha256":"p","file_sha256":"f"},
                "entries":[{"mask":"full_action","action_model_replay":True,"representative":True}]}
            (root/"manifests/replay65.json").write_text(json.dumps(meta))
            replay._write_npz(root/"data/source.replay.npz",**trajectory(n=128))
            replay._write_npz(root/"data/exact_initialization.npz",**MAPPING)
            write_worker(root/"simulations/baseline",[trajectory(n=128)]*2)
            drift=trajectory(n=128); drift["joint_pos"][70:]+=.3
            write_worker(root/"simulations/full_action",[drift])
            states=trajectory()["physical_state"]
            replay._write_npz(root/"data/full_action/prediction.npz",predicted_state=states,completed_state=states)
            report=replay.report_window(root)
            self.assertTrue(report["baseline_valid"])
            self.assertIs(report["quality_pass"],False)
            self.assertEqual(report["model_physical_quality"],"FAIL_ABSOLUTE_THRESHOLDS")
            self.assertEqual(report["model_quality"]["full_action"]["mode"],"absolute_thresholds")
            comparison=report["comparisons"]["source_to_full_action"]
            self.assertEqual(comparison["aligned_state_count"],65)
            self.assertEqual(comparison["prefix_max_abs"]["joint_pos"],0)
            self.assertEqual(comparison["first_crossing_episode_frame"]["joint_position_rmse_rad"],70)
            meta["source"]["deterministic"]=False
            (root/"manifests/replay65.json").write_text(json.dumps(meta))
            self.assertFalse(replay.report_window(root)["baseline_valid"])

    def test_baseline_thresholds(self):
        metrics={k:0. for k in replay.THRESHOLDS}; metrics["foot_contact_accuracy"]=1.
        self.assertTrue(replay.passes(metrics))
        metrics["root_orientation_max_deg"]=6
        self.assertFalse(replay.passes(metrics))

    def test_model_quality_is_relative_to_valid_baseline(self):
        baseline={key:1e-2 for key in replay.MODEL_QUALITY_METRICS}
        model={key:1.1e-2 for key in replay.MODEL_QUALITY_METRICS}
        result=replay.model_quality_against_baseline(baseline,model)
        self.assertTrue(result["pass"])
        model["body_mpjpe_m"]=1.3e-2
        result=replay.model_quality_against_baseline(baseline,model)
        self.assertFalse(result["pass"])

    def test_prepared_hash_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"manifests").mkdir()
            (root/"input").write_text("one")
            (root/"manifests/prepared_hashes.json").write_text(json.dumps({"input":replay.file_sha256(root/"input")}))
            replay.verify_prepared(root)
            (root/"input").write_text("two")
            with self.assertRaisesRegex(ValueError,"changed"):
                replay.verify_prepared(root)

    def test_prepare_actual_model_checkpoint_and_artifact_roundtrip(self):
        """Real model + serialization + preparation; only the external HDF source is mocked."""
        from cvae_sa.posterior_hierarchical_standard_cvae import _new_checkpoint
        from cvae_sa.cvae_protocol import CHECKPOINT, Fixtures
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dataset_run=root/"dataset"; source_run=root/"training"
            (dataset_run/"data").mkdir(parents=True); (dataset_run/"manifests").mkdir()
            (source_run/"checkpoints").mkdir(parents=True)
            np.savez(dataset_run/"data/normalization.npz",physical_state_mean=np.zeros(70),physical_state_std=np.ones(70),action_mean=np.zeros(29),action_std=np.ones(29))
            for name in ("dataset_manifest.json","episodes.jsonl"):
                (dataset_run/"manifests"/name).write_text("{}")
            truth=trajectory(); value=batch(1)
            value["physical_state"]=torch.from_numpy(truth["physical_state"]).unsqueeze(0)
            value["action"]=torch.zeros(1,64,29)
            sample={k:v[0] for k,v in value.items()}
            class Dataset:
                refs=[SimpleNamespace(episode_index=0,fixed_start=0)]
                episodes=[{"steps":64}]
                def __getitem__(self,index): return sample
                def close(self): pass
            ds=Dataset(); fixtures=Fixtures(ds,[0],expand=True)
            model=build_model(config()); cfg={"model":model.config,"data":{"window_transitions":64,"max_windows":1}}
            checkpoint=_new_checkpoint(model,torch.optim.AdamW(model.parameters()),None,cfg,stage="B",step=2)
            checkpoint.update(format_version=CHECKPOINT,config=cfg,dataset_identity=replay.data_identity(dataset_run,fixtures.manifest()),training_contract={"training_mask_seed":42,"mask_mode":"fixed"})
            path=source_run/"checkpoints/best.pt"; torch.save(checkpoint,path)
            init={"joint_names":np.asarray([f"joint_{i}" for i in range(29)]),
                "nominal_default_joint_pos":np.zeros(29,np.float32),"action_scale":np.ones(29,np.float32),
                "action_offset":np.zeros(29,np.float32),"action_clip":np.empty(0,np.float32),
                "wrapper_action_clip":np.float32(np.nan),"physics_state_v3":truth["physical_state"][0]}
            truth["action_target_canonical"]=np.zeros((64,29),np.float32)
            truth["joint_names"]=init["joint_names"]
            args=argparse.Namespace(output_run=root/"output",checkpoint=path,dataset_run=dataset_run,route="B",window_index=0,selection=None,
                masks=list(replay.REPRESENTATIVES),mask_seed=None,sample_seed=3,sample_index=0,simulation_seed=7,action_mode="masked-completion",device="cpu",command="prepare",
                horizon_steps=64)
            with patch.object(replay,"make_dataset",return_value=(ds,[0])),patch.object(replay,"load_source",return_value=(truth,init,{}, {"motion_file":"unused","motion_file_sha256":"unused"})):
                replay.prepare(args)
            replay.verify_prepared(args.output_run)
            manifest=json.loads((args.output_run/"manifests/replay65.json").read_text())
            self.assertTrue(manifest["exact_training_fixture"])
            self.assertEqual(args.mask_seed,42)
            child=args.output_run/manifest["windows"][0]
            report=replay.report_window(child)
            self.assertIsNone(report["baseline_valid"])
            self.assertEqual(len(json.loads((child/"manifests/replay65.json").read_text())["entries"]),8)
            with np.load(child/"data/full_action/prediction.npz") as data:
                self.assertEqual(data["completed_state"].shape,(65,70))
                np.testing.assert_array_equal(data["completed_state"],truth["physical_state"])
            with self.assertRaises(FileExistsError):
                replay.prepare(args)


def fake_capture_run(run,replays,start=64,horizon=64):
    """Capture run after the Isaac worker; the baseline child is pre-sealed so no simulator is launched."""
    for folder in ("manifests","data/source","markers","plots","logs"):
        (run/folder).mkdir(parents=True,exist_ok=True)
    steps=start+horizon
    source=trajectory(n=steps)
    replay._write_npz(run/"data/source/000000.replay.npz",**source,processed_action=source["raw_action"])
    replay._write_trajectory(run/"data/source/000000.trajectory.pkl",joint_pos=source["joint_pos"],
                             root_pos=source["root_pos"],root_quat=source["root_quat"],fps=50.)
    (run/"data/source/000000.runtime.json").write_text(json.dumps({**VALID_RUNTIME,"control_steps":steps}))
    (run/"manifests/exact_initialization_readback_capture.json").write_text(json.dumps(VALID_READBACK))
    replay._write_npz(run/"data/exact_initialization.npz",**MAPPING)
    replay._write_npz(run/"data/dataset_source.replay.npz",**trajectory(.01,n=steps))
    replay._write_npz(run/"data/normalization.npz",unused=np.zeros(1))
    (run/"manifests/source_schema.json").write_text("{}")
    (run/"markers/replay65_capture.ok").write_text("ok\n")
    meta={"window":{"window_index":3,"stable_window_id":"w3","motion_key":"m","window_start":start},"window_start":start,
          "horizon_steps":horizon,"simulation_seed":7,"motion_file":"motion.pkl","motion_file_sha256":"x",
          "initialization":{"payload_sha256":"p","file_sha256":"f"}}
    (run/"manifests/capture.json").write_text(json.dumps(meta))
    write_worker(run/"simulations/baseline",replays)
    return SimpleNamespace(output_run=run,checkpoint=run.parent/"ckpt/checkpoints/x.pt",dataset_run=run.parent/"dataset",
                           window_index=3,horizon_steps=horizon,simulation_seed=7)


class Replay65CaptureCompareTest(unittest.TestCase):
    def test_capture_determinism_gate_and_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/"capture"
            args=fake_capture_run(run,[trajectory(n=128)]*2)
            complete=replay.capture(args)
            self.assertTrue(complete["deterministic"])
            self.assertEqual((run/"markers/replay65_capture.status").read_text().strip(),"DETERMINISTIC")
            gate=json.loads((run/"manifests/capture_determinism.json").read_text())
            self.assertEqual(gate["scored_region"],[64,128])
            self.assertGreater(gate["dataset_to_capture"]["scored"]["joint_position_rmse_rad"],0)
            request=json.loads((run/"manifests/replay65_capture_request.json").read_text())
            self.assertEqual(request["steps"],128)
            loaded=replay.load_capture(run)
            self.assertEqual(loaded["canonical"].shape,(128,29))
            self.assertEqual(replay.capture(args)["hashes"],complete["hashes"])
            (run/"manifests/capture.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"capture artifact changed"):
                replay.verify_capture(run)

    def test_capture_non_deterministic_simulator(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/"capture"
            drift=trajectory(n=128); drift["root_pos"][100:,0]+=1e-3
            complete=replay.capture(fake_capture_run(run,[trajectory(n=128),drift]))
            self.assertFalse(complete["deterministic"])
            gate=json.loads((run/"manifests/capture_determinism.json").read_text())
            self.assertFalse(gate["within_limits"]["capture_to_original_2"]["root_pos"])
            self.assertTrue(gate["within_limits"]["capture_to_original_1"]["root_pos"])

    def test_capture_rejects_different_arguments_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fake_capture_run(Path(tmp)/"capture",[trajectory(n=128)]*2)
            args.window_index=4
            with self.assertRaisesRegex(ValueError,"different|started with"):
                replay.capture(args)

    def test_first_fall_frame(self):
        value=trajectory(n=64)
        self.assertIsNone(replay.first_fall_frame(value))
        value["root_pos"][40:,2]=.2
        self.assertEqual(replay.first_fall_frame(value),40)
        tilted=trajectory(n=64); angle=np.deg2rad(70)
        tilted["root_quat"][20:]=[np.cos(angle/2),np.sin(angle/2),0,0]
        self.assertEqual(replay.first_fall_frame(tilted),20)

    @staticmethod
    def prepared_run(root,route,mask,model,source=None,window_id="w3"):
        window=root/"windows/w000003"
        (window/"manifests").mkdir(parents=True); (window/"data").mkdir()
        (root/"manifests").mkdir(exist_ok=True)
        (root/"manifests/replay65.json").write_text(json.dumps({"windows":["windows/w000003"]}))
        (root/"manifests/prepared_hashes.json").write_text("{}")
        source=source or trajectory(n=128)
        replay._write_npz(window/"data/source.replay.npz",**source)
        replay._write_trajectory(window/"data/source.trajectory.pkl",joint_pos=source["joint_pos"],
                                 root_pos=source["root_pos"],root_quat=source["root_quat"],fps=50.)
        masks=[mask] if route=="A" else list(dict.fromkeys([mask,"state_gap_16"]))
        meta={"route":route,"window":{"stable_window_id":window_id,"window_start":64},"window_start":64,
              "horizon_steps":64,"score_offset":64,"simulation_seed":7,"source_mode":"recapture",
              "entries":[{"mask":m,"action_model_replay":True} for m in masks]}
        (window/"manifests/replay65.json").write_text(json.dumps(meta))
        for m in masks:
            write_worker(window/"simulations"/m,[model])
        return root

    def test_compare_routes_on_shared_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); fallen=trajectory(n=128); fallen["root_pos"][100:,2]=.2
            runs=[self.prepared_run(tmp/"a","A","state_gap_16",trajectory(.01,n=128)),
                  self.prepared_run(tmp/"b","B","full_action",trajectory(.02,n=128)),
                  self.prepared_run(tmp/"c","C","full_action",fallen)]
            args=SimpleNamespace(runs=runs,labels=["A","B","C"],mask="full_action",layout="both",
                                 output_run=tmp/"compare",skip_render=True,model=None,gl="egl")
            manifest=replay.compare(args)
            self.assertEqual([r["mask"] for r in manifest["routes"]],["state_gap_16","full_action","full_action"])
            self.assertEqual([r["fall_frame"] for r in manifest["routes"]],[None,None,36])
            self.assertEqual(len(manifest["routes"][0]["per_frame"]["joint_position_rmse_rad"]),65)
            self.assertAlmostEqual(manifest["routes"][1]["metrics"]["joint_position_rmse_rad"],.02,places=6)
            self.assertTrue((tmp/"compare/plots/compare_errors.svg").is_file())
            with self.assertRaises(FileExistsError):
                replay.compare(args)

    def test_compare_rejects_different_window_or_missing_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            a=self.prepared_run(tmp/"a","A","full_action",trajectory(n=128))
            b=self.prepared_run(tmp/"b","B","full_action",trajectory(n=128),window_id="other")
            args=SimpleNamespace(runs=[a,b],labels=["A","B"],mask="full_action",layout="both",
                                 output_run=tmp/"out1",skip_render=True,model=None,gl="egl")
            with self.assertRaisesRegex(ValueError,"same window"):
                replay.compare(args)
            c=self.prepared_run(tmp/"c","C","state_gap_16",trajectory(n=128))
            args.runs, args.output_run=[a,c], tmp/"out2"
            with self.assertRaisesRegex(ValueError,"no Action replay for full_action"):
                replay.compare(args)

    def test_cli_parses_capture_and_compare(self):
        with patch.object(replay,"capture") as capture, patch.object(replay,"compare") as compare, \
                patch.object(replay,"render") as render, \
                patch.object(replay.sys,"platform","win32"):
            replay.main(["capture","--checkpoint","c.pt","--dataset-run","d","--output-run","o","--window-index","844"])
            replay.main(["compare","--runs","a","b","c","--output-run","o","--skip-render"])
            replay.main(["render","--run","r","--model","m","--robot-profile","g1_plus_locked29",
                         "--allow-render-code-change"])
        self.assertEqual(capture.call_args[0][0].horizon_steps,192)
        self.assertEqual(compare.call_args[0][0].labels,["A","B","C"])
        self.assertEqual(compare.call_args[0][0].layout,"both")
        self.assertEqual(render.call_args[0][0].robot_profile,"g1_plus_locked29")


if __name__=="__main__":
    unittest.main()
