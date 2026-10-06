#!/usr/bin/env python3
"""Pure CPU receipt protocol tests; no simulator or torch import."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('com_native_driver', HERE/'native_driver.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
GOOD = {'status':'engineering_probe_completed', 'num_envs':4096, 'policy_steps':600,
        'physics_steps':2400, 'actor_tensors_unchanged':True, 'checkpoint_unchanged':True,
        'source_files_unchanged':True, 'target_contract_unchanged':True,
        'formal_updates_consumed':0, 'optimizer_steps':0, 'qualification':None,
        'source_files_sha256':{'src/wheeled_tasks/chassis/root_kinematics.py':'pinned'}}
COUNTS = {name:1 for name in driver.SEMANTIC_PATHS}
COUNTS['native_data_updates'] = 2400

class ReceiptTests(unittest.TestCase):
    def test_ast(self):
        ast.parse((HERE/'native_driver.py').read_text())

    def test_success(self):
        for mode, variant in [('semantic','candidate'), ('throughput','candidate'), ('throughput','baseline')]:
            self.assertEqual(driver.coverage_status(GOOD, mode, variant, COUNTS), (True, []))

    def test_independent_failed_report_checks(self):
        failures = {'status':'failed', 'num_envs':4095, 'policy_steps':599, 'physics_steps':2399,
                    'actor_tensors_unchanged':False, 'checkpoint_unchanged':False,
                    'source_files_unchanged':False, 'target_contract_unchanged':False,
                    'formal_updates_consumed':1, 'optimizer_steps':1,
                    'qualification':'qualified', 'source_files_sha256':{}}
        for field, wrong in failures.items():
            with self.subTest(field=field):
                report = deepcopy(GOOD); report[field] = wrong
                complete, missing = driver.coverage_status(report,'semantic','candidate',COUNTS)
                self.assertFalse(complete); self.assertTrue(missing)
        report = deepcopy(GOOD); report['status'] = 'running'
        self.assertFalse(driver.coverage_status(report,'semantic','candidate',COUNTS)[0])

    def test_every_semantic_path_and_native_clock_rejected_when_missing(self):
        for name in driver.SEMANTIC_PATHS:
            with self.subTest(name=name):
                counts = dict(COUNTS); counts[name] = 0
                complete, missing = driver.coverage_status(GOOD,'semantic','candidate',counts)
                self.assertFalse(complete); self.assertIn('observed_'+name, missing)
        counts = dict(COUNTS); counts['native_data_updates'] = 2399
        self.assertIn('at_least_2400_native_updates',driver.coverage_status(GOOD,'semantic','candidate',counts)[1])

    def test_postclose_preserves_terminal_preclose_bytes(self):
        with tempfile.TemporaryDirectory(prefix='com-sidecar-cpu-') as temp:
            path = Path(temp); preclose = path/'root_com_candidate_receipt.json'
            terminal = {'mode_complete':True, 'written_before_app_close':True}
            preclose.write_text(json.dumps(terminal)+'\n'); expected = preclose.read_bytes()
            driver.write_postclose_receipt(path, 0, None)
            self.assertEqual(preclose.read_bytes(), expected)
            postclose = json.loads((path/'root_com_candidate_postclose_receipt.json').read_text())
            self.assertTrue(postclose['preclose_receipt_present'])
            self.assertEqual(postclose['preclose_receipt_sha256'],driver.digest(preclose))
            self.assertTrue(postclose['acceptance_must_use_preclose_receipt'])

    def test_postclose_cannot_manufacture_acceptance_or_create_directory(self):
        with tempfile.TemporaryDirectory(prefix='com-sidecar-cpu-') as temp:
            path = Path(temp)
            driver.write_postclose_receipt(path,0,None)
            self.assertFalse((path/'root_com_candidate_receipt.json').exists())
            postclose = json.loads((path/'root_com_candidate_postclose_receipt.json').read_text())
            self.assertFalse(postclose['preclose_receipt_present'])
            self.assertNotIn('mode_complete',postclose)
            absent = path/'absent'; driver.write_postclose_receipt(absent,0,None)
            self.assertFalse(absent.exists())

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReceiptTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (HERE/'cpu_driver_protocol.json').write_text(json.dumps({
        'scope':'stdlib_only_CPU_no_native_no_GPU_no_PPO',
        'tests_run':result.testsRun, 'successful':result.wasSuccessful(),
        'failures':len(result.failures), 'errors':len(result.errors),
        'report_rejection_cases':13, 'semantic_missing_rejection_cases':12,
        'preclose_preservation_test':True, 'missing_preclose_no_acceptance_test':True,
        'driver_sha256':driver.digest(HERE/'native_driver.py'),
        'test_sha256':driver.digest(__file__), 'qualification':None, 'ppo_speedup':None},indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
