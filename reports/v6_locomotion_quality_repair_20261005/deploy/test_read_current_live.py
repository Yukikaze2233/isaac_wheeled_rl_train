"""Publication-accounting checks without SSH, GPU or a running worker."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("live_reader", Path(__file__).with_name("read_current_live.py"))
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

class AccountingTests(unittest.TestCase):
    def snapshot(self, charged=110, actor=100, attempted=110):
        progress = dict(successful_updates=110, attempted_updates=attempted, actor_updates_in_block=100, parent_updates=23868)
        account = dict(attempts={"speed/attempt_002/train": dict(consumed_updates=charged, actor_updates=actor)},
            consumed_updates=1000+charged, domain_consumed_updates={"speed":charged})
        return progress, account

    def read(self, **kwargs):
        p,a=self.snapshot(**kwargs)
        return reader.accounting_snapshot(p,a,"speed/attempt_002/train")

    def test_aligned_completed(self):
        result=self.read()
        self.assertEqual(result["status"],"aligned_completed_updates")
        self.assertEqual(result["learning_progress_global_completed"],23978)

    def test_reserved_work_is_not_actor_learning(self):
        result=self.read(charged=111,attempted=111)
        self.assertEqual(result["status"],"inflight_reservation_charged_not_completed")
        self.assertEqual(result["worker_actor_completed_updates"],100)
        self.assertEqual(result["worker_inflight_reservations"],1)

    def test_supervisor_lag_is_not_terminal(self):
        self.assertEqual(self.read(charged=109,actor=99)["status"],"supervisor_ledger_lags_worker_progress")

    def test_non_atomic_snapshot_keeps_two_clocks(self):
        self.assertEqual(self.read(charged=111,actor=101)["status"],"non_atomic_publication_snapshot")

    def test_invalid_ledger_sum_is_rejected(self):
        p,a=self.snapshot();a["consumed_updates"]+=1
        with self.assertRaises(ValueError):reader.accounting_snapshot(p,a,"speed/attempt_002/train")

    def test_impossible_worker_reservation_is_rejected(self):
        with self.assertRaises(ValueError):self.read(attempted=112)

if __name__=="__main__":unittest.main()
