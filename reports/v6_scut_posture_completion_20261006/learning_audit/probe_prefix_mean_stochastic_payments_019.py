#!/usr/bin/env python3
"""Identity-bound native actual-payment observation of source712/seal2579.

CPU --preflight-only never imports an App. Explicit native execution reuses the
unchanged55 wrapper, its actor, native sampling, resume/reset and600-step loop.
This entry performs no PPO, optimizer, reward, command or controller edit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import types

BASE_SHA = "55f4511284ae09010ba9d79af2a747b70b1f9da3df6b4f713323340b64648b9d"
HELPER_SHA = "51580e8bd7c08025edd69588e70e62a2970cb6afb5f97f355d85ddf48460a8ac"
INITIAL_BIRTH_SHA = "06ddce7de22f8fa1c064a1797b20cb4ac9bc4e6a30cf8a89b77b28958f60c0dc"
INITIAL_ENV_RNG_SHA = "66e20269960768bdf7277de2a0c311e8ae61fc899f3b821d0e48840b58a32921"
INITIAL_SAMPLING_SHA = "102a3abeec4215c42410d077c7fbb77ae4a417e4f5be37f5cb6dacb3d2afd53b"
FINAL_SAMPLING_SHA = "3c7290917984f56ba8b61e70f80245f89e08525ad3324c0a56938ed620f24cb1"


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bound_module(path, expected, name):
    """Hash source before executing it, without reading or writing cached pyc."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Artifact SHA differs: " + str(path))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def artifact_binding(base_path, helper_path):
    require(digest(base_path) == BASE_SHA and digest(helper_path) == HELPER_SHA,
            "Original wrapper or payment helper SHA differs")
    return {str(Path(p).resolve()): digest(p) for p in (Path(__file__), base_path, helper_path)}


def verify_artifacts(binding):
    require(all(digest(path) == expected for path, expected in binding.items()),
            "Payment artifact changed during observation")


def native_context(report, *, final=False):
    require(report.get("initial_physical_birth_sha256") == INITIAL_BIRTH_SHA
            and report.get("initial_env_rng_sha256") == INITIAL_ENV_RNG_SHA,
            "Native birth/environment RNG differs from authenticated2579 comparison")
    if final:
        require(report.get("sampling_rng_initial_sha256") == INITIAL_SAMPLING_SHA
                and report.get("sampling_rng_final_sha256") == FINAL_SAMPLING_SHA,
                "Native private sampling stream differs from authenticated2579 comparison")


def guarded_writer(write_json, args, binding):
    def write(path, value):
        try:
            verify_artifacts(binding)
            if Path(path).name == "report.json" and value.get("status") == "native600_completed_preclose":
                native_context(value, final=True)
                ledger = json.loads((args.output / "payment_summary.json").read_text())
                require(ledger.get("complete") is True and ledger.get("native_lane_intervals") == 4096 * 600
                        and ledger.get("event_producer_calls") == 600
                        and value.get("actual_payment_complete") is True
                        and value.get("payment_summary_sha256") == digest(args.output / "payment_summary.json")
                        and ledger.get("identity_sha256") == value["identity_sha256"],
                        "Completed native actual-payment admission differs")
                require(not (args.output / "payment_observer_failure.json").exists(), "Payment cleanup failure present")
                value.update(payment_artifacts_unchanged=True,
                             initial_birth_and_private_stream_match_original2579=True,
                             process_exit_not_observed_by_this_preclose_report=True)
            return write_json(path, value)
        except BaseException:
            # This can fail after observer.finish. Persist before base finally's
            # App.close, which may exit before the driver exception handler.
            write_json(args.output / "payment_admission_failure.json", dict(
                complete=False, source_admission=False, app_close_returned=None,
                status="payment_preclose_publication_rejected", error=traceback.format_exc(),
                artifact_sha256=binding, PPO_updates=0, optimizer_steps=0))
            raise
    return write


class ClosingStream:
    """The base finally closes this before Kit.close can exit the process."""
    def __init__(self, original, cleanup):
        self.original, self.cleanup = original, cleanup

    def write(self, value):
        return self.original.write(value)

    def close(self):
        try:
            self.cleanup()
        finally:
            self.original.close()


def observer_class(base, helper, args, identity, report, binding, write_json):
    class WithPayments(base.NativeObserver):
        def __init__(self, env, output):
            super().__init__(env, output)
            self.payment = None
            self.payment_complete = self.payment_cleanup_done = False
            self.payment_begin = self.payment_step = None
            self.prior_begin = env.recovery.begin_policy
            self.prior_step = env.step
            self.sparse = ClosingStream(self.sparse, self._before_close)
            try:
                native_context(report)
                before_cpu_rng = self.t.get_rng_state().clone()
                before_env_rng = env.generator.get_state().clone()
                self.payment = helper.NativePaymentObserver(env, lambda: self.live, lambda: self.hold_start,
                    helper.original_goal_ids(env), policy_steps=base.POLICY_STEPS)

                def begin():
                    result = self.prior_begin()  # Native and base observer begin once.
                    self.payment.begin(env.tick + 1)
                    return result

                def step(actions):
                    result = self.prior_step(actions)  # Original physics/control/reward once.
                    require(isinstance(result, tuple) and len(result) == 4, "Native step return ABI differs")
                    self.payment.capture(result[1], result[2], result[3])
                    return result  # Same tuple and every original return object.

                self.payment_begin, self.payment_step = begin, step
                env.recovery.begin_policy, env.step = begin, step
                require(self.t.equal(before_cpu_rng, self.t.get_rng_state())
                        and self.t.equal(before_env_rng, env.generator.get_state()),
                        "Payment observer setup changed native CPU/environment RNG")
            except BaseException:
                self._before_close()
                self.sparse.original.close()
                raise

        def _restore(self):
            errors = []
            if self.payment is not None:
                try:
                    self.payment.close()
                except BaseException:
                    errors.append(traceback.format_exc())
            for owner, name, installed, prior in (
                (self.env.recovery, "begin_policy", self.payment_begin, self.prior_begin),
                (self.env, "step", self.payment_step, self.prior_step)):
                if installed is None:
                    continue
                if getattr(owner, name) is installed:
                    setattr(owner, name, prior)
                elif getattr(owner, name) is not prior:
                    errors.append("Payment hook ownership changed: " + name)
            return errors

        def _before_close(self):
            if self.payment_cleanup_done:
                return
            errors = self._restore()
            self.payment_cleanup_done = True
            if not self.payment_complete or errors:
                # Failure receipt precedes App.close; it cannot be skipped by
                # Kit's process exit. Never turn exit0 into admission.
                write_json(args.output / "payment_observer_failure.json", dict(
                    status="payment_observation_incomplete_or_cleanup_failed", complete=False,
                    completed_policy_steps=self.payment.tick if self.payment else 0,
                    event_calls=self.payment.event_calls if self.payment else 0,
                    unfinished_interval=self.payment.active is not None if self.payment else None,
                    callback_restore_errors=errors, artifact_sha256=binding,
                    error_active_at_preclose=traceback.format_exc(), app_close_returned=None,
                    source_admission=False, PPO_updates=0, optimizer_steps=0))

        def finish(self, output):
            require(self.payment is not None and self.t.equal(self.payment.first_closed, ~self.live),
                    "Payment and base first-cohort closure disagree")
            ledger = self.payment.summary()
            verify_artifacts(binding)
            require(base.verify_inputs(args, identity), "Original source/seal/wrapper changed before payment seal")
            errors = self._restore()
            require(not errors, "Payment callback restore failed: " + str(errors))
            ledger.update(artifact_sha256=binding, identity_sha256=digest(args.output / "identity.json"),
                original_wrapper_sha256=BASE_SHA, source_commit=base.SOURCE_COMMIT,
                checkpoint_sha256=base.CHECKPOINT_SHA, checkpoint_clocks=base.CLOCKS,
                native_step_returns_preserved=True, producer_outputs_preserved=True,
                analytic_std_entropy_is_not_payment=True, advantages_and_PPO_gradients="not_observed")
            write_json(output / "payment_summary.json", ledger)
            self.payment_complete = self.payment_cleanup_done = True
            try:
                original = super().finish(output)
            except BaseException:
                write_json(output / "payment_observer_failure.json", dict(
                    status="original_base_observer_finish_failed_preclose", complete=False,
                    native_payments_reconciled=True, source_admission=False, app_close_returned=None,
                    error=traceback.format_exc(), artifact_sha256=binding, PPO_updates=0, optimizer_steps=0))
                raise
            original.update(payment_summary_file="payment_summary.json",
                            payment_summary_sha256=digest(output / "payment_summary.json"),
                            actual_payment_complete=True, payment_hooks_restored_before_app_close=True)
            return original

    return WithPayments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("mean", "stochastic"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path,
                        default=Path(__file__).with_name("source712_native_probe_manifest_019.json"))
    parser.add_argument("--base-wrapper", type=Path,
                        default=Path(__file__).with_name("probe_prefix_mean_stochastic_019.py"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    helper_path = Path(__file__).with_name("observe_prefix_native_payments_019.py")
    binding = artifact_binding(args.base_wrapper, helper_path)
    base = load_bound_module(args.base_wrapper, BASE_SHA, "bound_prefix_base_019")
    helper = load_bound_module(helper_path, HELPER_SHA, "bound_prefix_payments_019")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="native-prefix-actual-payments-driver-019-v1", status="starting", arm=args.arm,
        started_at_utc=datetime.now(timezone.utc).isoformat(), artifact_sha256=binding,
        qualification=None, acceptance="not_assessed", formal_updates_consumed=0,
        optimizer_steps=0, optimizer_instances=0, PPO_updates=0,
        policy_steps_requested=600, physics_steps_requested=2400, requested_seconds=12.,
        reward_controller_command_overrides=[], evaluation_projection=False, clock_advance=False)
    original_write, original_observer = base.write_json, base.NativeObserver
    try:
        config, manifest, control, saved, identity = base.authenticate(args)
        identity.update(payment_observer_artifact_sha256=binding,
                        original_wrapper_sha256=BASE_SHA, actual_payment_source="computed_native_extras_and_once_only_events")
        original_write(args.output / "identity.json", identity)
        report["identity_sha256"] = digest(args.output / "identity.json")

        guarded_write = guarded_writer(original_write, args, binding)
        base.write_json = guarded_write
        base.NativeObserver = observer_class(base, helper, args, identity, report, binding, original_write)
        if args.preflight_only:
            verify_artifacts(binding)
            require(base.verify_inputs(args, identity)
                    and base.snapshot_sources(args.source_root.resolve(), config, identity["archive_source_files"])
                    == identity["source_files_sha256"], "CPU preflight immutable input changed")
            report.update(status="preflight_passed_scene_not_run", native_scene_run=False,
                runtime_preflight_passed=False, actual_payment_complete=False,
                policy_steps_completed=0, physical_steps_completed=0)
            guarded_write(args.output / "report.json", report)
        else:
            guarded_write(args.output / "report.json", report)
            base.run_native(args, config, manifest, control, saved, identity, report)
        return 0
    except BaseException:
        if report.get("status") != "native600_completed_preclose":
            report.update(status="failed", error=traceback.format_exc())
            original_write(args.output / "report.json", report)
        else:
            original_write(args.output / "payment_postclose_or_admission_error.json",
                           dict(error=traceback.format_exc(), source_admission=False))
        return 1
    finally:
        base.write_json, base.NativeObserver = original_write, original_observer


if __name__ == "__main__":
    raise SystemExit(main())
