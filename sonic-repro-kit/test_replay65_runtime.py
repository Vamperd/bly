import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
from replay65_runtime import audit_and_freeze, compare_runtime, suppress_interval_event
from render_replay65 import video_jobs


class Events:
    active_terms={"startup":["mass"],"reset":["reset_scene"],"interval":["push"]}
    def __init__(self):
        self.cfg=NS(func="random_push")
    def get_term_cfg(self,name):
        return self.cfg
    def set_term_cfg(self,name,cfg):
        self.cfg=cfg


class ReplayRuntimeTest(unittest.TestCase):
    def runtime(self,delay=0):
        cfg=NS(min_delay=0,max_delay=delay)
        actuator=NS(cfg=cfg,joint_names=["j"])
        robot=NS(actuators={"legs":actuator},cfg=NS(spawn=NS(articulation_props=NS(solver_position_iteration_count=4,solver_velocity_iteration_count=1))))
        raw=NS(event_manager=Events(),scene={"robot":robot},physics_dt=.005,step_dt=.02,cfg=NS(decimation=4,sim=NS(gravity=[0,0,-9.81])))
        contract={"simulation":{"sim_dt":.005,"control_dt":.02,"decimation":4,"gravity_w":[0,0,-9.81],"solver_position_iteration_count":4,"solver_velocity_iteration_count":1},
            "active_events":{"interval":[]},"actuators":{"legs":{"type":"SimpleNamespace","joint_names":["j"],"min_delay":0,"max_delay":delay}}}
        return raw,{"replay65_contract":np.asarray(json.dumps(contract))}

    def test_runtime_interval_frozen_without_replacing_physics_parameters(self):
        raw,values=self.runtime()
        report=audit_and_freeze(raw,values)
        self.assertTrue(report["contract_verified"])
        self.assertIs(raw.event_manager.cfg.func,suppress_interval_event)
        self.assertEqual(report["suppressed_interval_events"],["push"])

    def test_unknown_delay_cannot_claim_exact_initialization(self):
        raw,values=self.runtime(2)
        report=audit_and_freeze(raw,values)
        self.assertFalse(report["contract_verified"])
        self.assertEqual(report["hidden_state"]["actuator_queue"],"unknown")

    def test_source_interval_events_fail_closed(self):
        raw,values=self.runtime()
        contract=json.loads(values["replay65_contract"].item()); contract["active_events"]["interval"]=["push"]
        with self.assertRaisesRegex(ValueError,"source has interval"):
            audit_and_freeze(raw,{"replay65_contract":np.asarray(json.dumps(contract))})

    def test_timing_mismatch_stops_before_rollout(self):
        raw,values=self.runtime(); raw.step_dt=.04
        with self.assertRaisesRegex(ValueError,"contract mismatch"):
            audit_and_freeze(raw,values)

    def test_unknown_fields_not_reported_as_verified(self):
        checks=compare_runtime({}, {})
        self.assertTrue(all(value is None for value in checks.values()))

    def test_state_jobs_available_without_simulation(self):
        jobs=video_jobs(Path("missing"),{"entries":[{"mask":"state_gap_16","representative":True}]})
        self.assertEqual(len(jobs),1)
        self.assertEqual(jobs[0][0],"state_gap_16_state")


if __name__=="__main__":
    unittest.main()
