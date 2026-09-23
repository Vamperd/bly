from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
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


def trajectory(offset=0.):
    states=np.zeros((65,70),np.float32); states[:,66]=-1; states[:,67]=1
    quat=np.zeros((65,4),np.float32); quat[:,0]=1
    return dict(physics_state_v3=states,physical_state=states,joint_pos=np.zeros((65,29),np.float32)+offset,
        joint_vel=np.zeros((65,29),np.float32),root_pos=np.tile([0.,0.,1.],(65,1)).astype(np.float32),root_quat=quat,
        root_lin_vel=np.zeros((65,3),np.float32),root_ang_vel=np.zeros((65,3),np.float32),
        body_pos=np.zeros((65,3,3),np.float32),raw_action=np.zeros((64,29),np.float32))


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
            replay._write_npz(root/"data/recorded_hdf.replay.npz",**trajectory())
            replay._write_npz(root/"data/exact_initialization.npz",action_scale=np.ones(29),action_offset=np.zeros(29),action_clip=np.empty(0),wrapper_action_clip=np.float32(np.nan))
            for tag,n in (("baseline",2),("full_action",1)):
                child=root/"simulations"/tag; (child/"manifests").mkdir(parents=True); (child/"data/replay").mkdir(parents=True)
                replay._write_npz(child/"data/raw_actions.npz",raw_actions=np.zeros((64,n,29)))
                for i in range(n):
                    replay._write_npz(child/f"data/replay/{i:06d}.replay.npz",**trajectory(.3),processed_action=np.zeros((64,29)))
                    (child/f"manifests/exact_initialization_readback_{i:06d}.json").write_text(json.dumps({"errors":{}}))
                    (child/f"data/replay/{i:06d}.runtime.json").write_text("{}")
                hashes={str(p.relative_to(child)):replay.file_sha256(p) for p in child.rglob("*") if p.is_file()}
                (child/"manifests/replay65_worker_complete.json").write_text(json.dumps({"hashes":hashes}))
            (root/"data/full_action").mkdir()
            replay._write_npz(root/"data/full_action/prediction.npz",predicted_state=trajectory()["physical_state"],completed_state=trajectory()["physical_state"])
            report=replay.report_window(root)
            self.assertFalse(report["baseline_valid"])
            self.assertIsNone(report["quality_pass"])
            self.assertTrue(report["execution_complete"])
            self.assertIn("BASELINE_INVALID",report["banner"])
            self.assertEqual(report["comparisons"]["original_repeatability"]["joint_position_rmse_rad"],0)

    def test_baseline_thresholds(self):
        metrics={k:0. for k in replay.THRESHOLDS}; metrics["foot_contact_accuracy"]=1.
        self.assertTrue(replay.passes(metrics))
        metrics["root_orientation_max_deg"]=6
        self.assertFalse(replay.passes(metrics))

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
                masks=list(replay.REPRESENTATIVES),mask_seed=None,sample_seed=3,sample_index=0,simulation_seed=7,action_mode="masked-completion",device="cpu",command="prepare")
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


if __name__=="__main__":
    unittest.main()
