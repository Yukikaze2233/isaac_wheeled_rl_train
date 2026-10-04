"""Small offline checks of baseline identity, clock, and denominator rejection."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('native_reader',Path(__file__).with_name('read_native_tail.py'))
reader=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def endpoint(worker, timestamp):
    groups={}
    for name in reader.STAND+reader.MOTION+reader.RELEASE:
        groups[name]=dict(frames=worker*24,observed_frames=worker*24,
            episodes=worker,failures=worker//5,timeouts=worker-worker//5,boundary_truncations=worker//10,
            **{key:.1 if worker==50 else .2 for key in reader.AVERAGES})
        groups[name]['tilt_max_deg_since_worker_birth']=99.
    return dict(event_step=reader.CONSUMED_ORIGIN+worker-1,worker_successful_updates=worker,
        actor_updates_since_restart=worker-reader.CRITIC_UPDATES,
        task_actor_update=reader.ACTOR_ORIGIN+worker-reader.CRITIC_UPDATES,
        native_event_wall_time_utc=timestamp,groups=groups,groups_with_complete_counters=135,
        all_groups_counts=dict(frames=worker*4096*24,observed_frames=worker*4096*24,
            episodes=worker*10,failures=worker*3,timeouts=worker*7,boundary_truncations=worker*2))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.baseline=endpoint(50,'2026-10-04T21:44:00+00:00')
        self.baseline.update(source_commit=reader.SOURCE_COMMIT,queried_at_utc='2026-10-04T21:44:01+00:00',
            event=dict(file=reader.DIRECTORY+'/events.out.tfevents.test'))
        self.report=dict(source_commit=reader.SOURCE_COMMIT,event=copy.deepcopy(self.baseline['event']),
            queried_at_utc='2026-10-04T21:55:01+00:00',progress=dict(num_envs=4096),expected_groups=135,
            latest_cumulative_behavior=endpoint(70,'2026-10-04T21:55:00+00:00'))

    def test_valid_difference_is_pure_actor_and_maxima_are_not_differenced(self):
        result=reader.baseline_difference(self.baseline,self.report,{})
        self.assertEqual(result['actor_updates_since_restart_inclusive'],[41,60])
        self.assertEqual(result['all_groups_counts']['observed_frames'],20*4096*24)
        group=result['groups']['stand_305mm']
        self.assertAlmostEqual(group['planar_speed_mean_m_s'],.45)
        self.assertNotIn('tilt_max_deg_since_worker_birth',group)
        self.assertIsNone(group['stationary_10s']['samples'])
        self.assertIsNone(result['exploration_std_observed_tail']['mean'])

    def test_cross_attempt_and_cross_source_rejected(self):
        for field in ('event','source_commit'):
            with self.subTest(field=field):
                bad=copy.deepcopy(self.baseline)
                if field=='event':bad['event']['file']=bad['event']['file'].replace('attempt_002','attempt_003')
                else:bad[field]='different-frozen-source'
                with self.assertRaisesRegex(ValueError,'Cross-'):
                    reader.baseline_difference(bad,self.report,{})

    def test_counter_rollback_rejected(self):
        self.report['latest_cumulative_behavior']['groups']['stand_305mm']['failures']=0
        with self.assertRaisesRegex(ValueError,'counter rollback'):
            reader.baseline_difference(self.baseline,self.report,{})

    def test_insufficient_frame_denominator_rejected(self):
        self.report['latest_cumulative_behavior']['groups']['stand_305mm']['frames']=50*24
        with self.assertRaisesRegex(ValueError,'Insufficient frame denominator'):
            reader.baseline_difference(self.baseline,self.report,{})

    def test_nonincreasing_clock_rejected(self):
        self.report['latest_cumulative_behavior']=endpoint(50,'2026-10-04T21:55:00+00:00')
        with self.assertRaisesRegex(ValueError,'clocks did not increase'):
            reader.baseline_difference(self.baseline,self.report,{})


if __name__=='__main__':unittest.main()
