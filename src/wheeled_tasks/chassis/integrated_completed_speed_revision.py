"""Named v26 completed-speed → v27 terrain boundary; no REC rebase.

The predecessor keeps its own immutable lineage identity. A SHA-bound entry
certificate authenticates that old REC and speed completion before mapping
the *new* worker metadata. Neither old checkpoints nor completion records are
rewritten, and the ordinary account remains the only consumption authority.
"""
from copy import deepcopy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import integrated_lineage as lineage
from .task_campaign import CAMPAIGN_ID, account_updates

SCHEMA = "v6-completed-speed-predecessor-v1"
ENTRY_SCHEMA = "v6-completed-speed-predecessor-entry-v1"
DIRECTORY = "revisions/v26_completed_speed_to_v27"
MODE = "inherit_v26_completed_speed"
SOURCE029_COMMIT = "e5bd9c1367b875d7f229b9a1930647013d023372"


def static_pin(root):
    from . import integrated_production as production
    files = production.artifact_files(26)
    return dict(schema=SCHEMA, campaign_plan_file=files["plan"],
        campaign_plan_sha256=lineage.digest(Path(root) / files["plan"]),
        lineage_manifest_file=files["lineage"],
        lineage_manifest_sha256=lineage.digest(Path(root) / files["lineage"]),
        source_revision_commit=SOURCE029_COMMIT,
        transition="completed_speed_to_terrain_only")


def parent_contract_allowed(source, target, root):
    """Native resume accepts only the exact old speed contract for v27 terrain."""
    from . import integrated_production as production
    if (production.revision_number(target) != 27
            or target["task_campaign"]["domain"] != "terrain"):
        return False
    pin = static_pin(root)
    _, manifest = lineage._json(pin["lineage_manifest_file"], pin["lineage_manifest_sha256"], root)
    row = next(r for r in manifest["domains"] if r["name"] == "speed")
    _, expected = lineage._json(row["contract_file"], row["contract_sha256"], root)
    return source == expected


def close_native_completed_speed(job, source_root):
    """Recover only the old normal speed terminal record, without a new worker.

    The caller holds the existing account flock. This calls the unchanged
    native terminal path, with both worker-start methods explicitly disabled.
    A stopped/incomplete worker or an uncharged update can never enter it.
    """
    from . import integrated_production as production
    if job.plan.get("contract_id") != "v6-task-campaign-v26":
        raise ValueError("Zero-PPO terminal closure supports only the original v26 binding")
    job.assert_no_live_workers()
    before = account_updates(job.plan, job.records)
    if production.reconcile_records(job.plan, job.root, job.records) != job.records:
        raise ValueError("Native closure cannot hide uncharged worker progress")
    row = next(r for r in job.plan["domains"] if r["name"] == "speed")
    binding, selected = production.select_source(job.plan, source_root, job.root,
        dict(job.report, active_joint_domain="speed"),
        previous_selection=job.report.get("joint_learning_sources", {}).get("speed"))
    if not selected["domain_complete"] or selected["critic_warmup_remaining"] != 0:
        raise ValueError("Zero-PPO terminal closure requires a completed speed full seal")
    checkpoint = lineage._checkpoint(binding, selected["checkpoint"], "speed", row["sha256"])
    lineage._finished(binding, checkpoint, dict(status="joint_domain_finished_unassessed",
        candidate_checkpoint=selected["checkpoint"]["path"], candidate_checkpoint_sha256=selected["checkpoint"]["sha256"]))
    endpoint = job.plan["domain_completion"]["schedule"]["end_actor_updates"]["speed"]
    if not (selected["fresh_domain_actor_updates"] >= endpoint
            or before["domain_consumed_updates"]["speed"] == 20000 or before["consumed_updates"] == job.plan["total_ceiling"]):
        raise ValueError("Speed terminal seal reached neither its actor endpoint nor charged cap")
    other = {d: deepcopy(r) for d, r in job.report["domains"].items() if d != "speed"}
    def forbidden(*_args, **_kwargs):
        raise ValueError("Zero-PPO native closure forbids worker construction or launch")
    execute, train_command = job.execute, job.train_command
    job.execute = job.train_command = forbidden
    try:
        job.run_integrated_domain(row)
    finally:
        job.execute, job.train_command = execute, train_command
    if (account_updates(job.plan, job.records) != before
            or {d: r for d, r in job.report["domains"].items() if d != "speed"} != other):
        raise ValueError("Native terminal closure changed paid clocks or another domain")
    return dict(status="closed_native_completed_speed", checkpoint=selected["checkpoint"],
        account=before, PPO_updates=0, next_domain_dispatched=False, qualification=None)


def predecessor_binding(binding):
    """Read the old original source, without enabling a general lineage alias."""
    plan = binding.read()
    pin = plan.get("completed_speed_predecessor")
    if plan.get("lineage_id") != "source030-joint-ordinary-stair-v1" or pin != static_pin(binding.source_root):
        raise ValueError("v27 requires its exact named completed-speed predecessor")
    _, old_manifest = lineage._json(pin["lineage_manifest_file"], pin["lineage_manifest_sha256"], binding.source_root)
    old_root = Path(old_manifest["source_root"]).resolve(strict=True)
    old, _ = lineage.load_plan(old_root / pin["lineage_manifest_file"],
        pin["lineage_manifest_sha256"], old_root, binding.run_root)
    _, old_plan = lineage._json(pin["campaign_plan_file"], pin["campaign_plan_sha256"], old_root)
    if (old_plan.get("contract_id") != "v6-task-campaign-v26"
            or old_plan["integrated_continuation"]["lineage_manifest"] != dict(
                file=pin["lineage_manifest_file"], sha256=pin["lineage_manifest_sha256"])):
        raise ValueError("Completed-speed predecessor is not frozen v26")
    return old, old_plan


def _pin(path):
    return dict(file=str(Path(path).resolve()), sha256=lineage.digest(path))


def _read_pin(pin, root):
    return lineage._json(pin["file"], pin["sha256"], root)[1]


def _retain(path, content):
    """Publish complete evidence without overwriting a prior certificate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Boundary evidence already exists with different bytes")
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".pending.", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError("Boundary evidence already exists with different bytes")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _wrapper_path(experiment, name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Native wrapper reference must name a retained local file")
    return experiment / name


def _source029_job(run_root, new_root):
    """Retain the current owner and every exact-command resume to its revision."""
    experiment = Path(run_root).resolve(strict=True).parent
    current_path = experiment / "job.json"
    current = json.loads(current_path.read_text())
    current_runner = str(Path(new_root) / "scripts/run_task_campaign.py")
    old_path = _wrapper_path(experiment, current.get("previous_job")) if current_runner in current["command"] else current_path
    old_job = json.loads(old_path.read_text())
    adopted, resumes, seen = old_job, [], {old_path}
    while adopted.get("source_revision_receipt") is None:
        path = _wrapper_path(experiment, adopted.get("previous_job"))
        if path in seen or path.is_symlink():
            raise ValueError("Native resume chain repeats or links a retained job")
        seen.add(path)
        previous = json.loads(path.read_text())
        if previous.get("command") != old_job.get("command"):
            raise ValueError("Native same-source resume changed its frozen command")
        resumes.append((path, previous))
        adopted = previous
    receipt_path = _wrapper_path(experiment, adopted.get("source_revision_receipt"))
    prior_path = _wrapper_path(experiment, adopted.get("previous_job"))
    receipt = json.loads(receipt_path.read_text())
    prior = json.loads(prior_path.read_text())
    _verify_source029(experiment, old_job, receipt, receipt_path, prior, prior_path,
        resumed_jobs=resumes, receipt_source_path=receipt_path, prior_source_path=prior_path)
    return dict(job=_pin(old_path), revision=_pin(receipt_path), previous_job=_pin(prior_path),
        **{f"resume_job_{i:03d}": _pin(path) for i, (path, _) in enumerate(resumes)})


def _verify_source029(experiment, job, receipt, receipt_path, prior, prior_path, *,
                      resumed_jobs=(), receipt_source_path=None, prior_source_path=None):
    adopted, seen = job, set()
    for path, previous in resumed_jobs:
        if (adopted.get("source_revision_receipt") is not None
                or path != _wrapper_path(experiment, adopted.get("previous_job"))
                or path in seen or previous.get("command") != job.get("command")):
            raise ValueError("Retained native resume chain differs from its exact command and links")
        seen.add(path)
        adopted = previous
    if (not isinstance(adopted.get("source_revision_receipt"), str)
            or not adopted["source_revision_receipt"]
            or receipt_source_path is not None and receipt_source_path != _wrapper_path(experiment, adopted["source_revision_receipt"])
            or prior_source_path is not None and prior_source_path != _wrapper_path(experiment, adopted.get("previous_job"))):
        raise ValueError("Native resume chain does not retain its source revision pair")
    if (receipt.get("schema") != "task-source-revision-v1"
            or receipt.get("source_commit") != SOURCE029_COMMIT
            or receipt.get("new_contract_sha256") != receipt.get("old_contract_sha256")):
        raise ValueError("Boundary predecessor must retain the actual source029 code-only receipt")
    module_path = Path(__file__).resolve().parents[3] / "scripts/chassis_remote_job.py"
    spec = importlib.util.spec_from_file_location("_completed_speed_wrapper", module_path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    wrapper.validate_source_revision(experiment, job["command"], prior, receipt_path, previous_job_path=prior_path)


def _authenticated(certificate, binding, *, before_paths=None):
    from . import integrated_production as production
    from . import integrated_runtime as runtime
    old, old_plan = predecessor_binding(binding)
    if (certificate.get("schema") != ENTRY_SCHEMA or certificate.get("new_lineage_id") != binding.read()["lineage_id"]
            or certificate.get("new_lineage_manifest_sha256") != binding.sha256
            or certificate.get("predecessor") != static_pin(binding.source_root)
            or certificate.get("PPO_updates_by_migration") != 0 or certificate.get("qualification") is not None):
        raise ValueError("Completed-speed certificate differs from its frozen boundary")
    if before_paths is None:
        before = {k: _read_pin(v, binding.run_root) for k, v in certificate["before"].items()}
    else:
        # First admission authenticates the current authoritative files before
        # retaining snapshots; a rejected parent cannot poison a future retry.
        before = {}
        for key, pin in certificate["before"].items():
            path = Path(before_paths[key])
            if lineage.digest(path) != pin["sha256"]:
                raise ValueError("Boundary source changed before evidence was frozen")
            before[key] = json.loads(path.read_text())
    ledger, state = before["account"], before["state"]
    totals = account_updates(old_plan, ledger["attempts"])
    if (ledger.get("campaign_id") != CAMPAIGN_ID or ledger.get("run_directory") != str(binding.run_root)
            or ledger.get("contract_sha256") != certificate["predecessor"]["campaign_plan_sha256"]
            or ledger.get("rollback_refunds_consumed_updates") is not False
            or any(ledger.get(k) != v or state.get(k) != v for k, v in totals.items())
            or certificate["account_at_boundary"] != totals
            or state.get("contract_sha256") != ledger["contract_sha256"]):
        raise ValueError("Completed-speed entry refunds or substitutes the old consumption account")
    entry = production.recovery_entry_admission(old, state)
    if certificate["old_entry"] != entry:
        raise ValueError("Completed-speed entry replaces its authenticated REC origin")
    record = state["domains"]["speed"]
    reference = dict(path=record.get("candidate_checkpoint"), sha256=record.get("candidate_checkpoint_sha256"))
    _, selected = production.select_source(old_plan, old.source_root, binding.run_root,
        dict(state, active_joint_domain="speed"), previous_selection=ledger.get("joint_learning_sources", {}).get("speed"))
    if selected["checkpoint"] != reference:
        raise ValueError("Completed-speed record is not the latest authenticated native own state")
    row = next(r for r in old.read()["domains"] if r["name"] == "speed")
    checkpoint = lineage._checkpoint(old, reference, "speed", row["contract_sha256"])
    lineage._finished(old, checkpoint, record)
    endpoint = old_plan["domain_completion"]["schedule"]["end_actor_updates"]["speed"]
    schedule = record.get("schedule_completion", {})
    if (not selected["domain_complete"] or selected["critic_warmup_remaining"] != 0
            or schedule.get("actual_actor_update") != selected["fresh_domain_actor_updates"]
            or schedule.get("end_actor_update") != endpoint
            or not (selected["fresh_domain_actor_updates"] >= endpoint
                    or totals["domain_consumed_updates"]["speed"] == 20000
                    or totals["consumed_updates"] == old_plan["total_ceiling"])
            or certificate["speed_selection"] != selected
            or certificate["speed_record"] != record
            or certificate["predecessor_component_clocks"] != runtime.source_clocks(old, selected)):
        raise ValueError("Speed has not reached its normal actor endpoint or charged cap")
    for domain in ("terrain", "jump", "stepjump"):
        if ledger.get("joint_learning_sources", {}).get(domain) is not None:
            raise ValueError("v27 entry cannot migrate an already started old downstream domain")
        for path in (binding.run_root / domain).glob("attempt_*/train/**/*.checkpoint.json"):
            meta = json.loads(path.read_text()).get("integrated_lineage")
            if meta is not None and meta.get("lineage_id") != binding.read()["lineage_id"]:
                raise ValueError("Completed-speed entry cannot replace an old downstream joint seal")
    evidence_paths = before_paths or {key: pin["file"] for key, pin in certificate["before"].items()}
    resume_keys = sorted(key for key in before if key.startswith("source029_resume_job_"))
    if resume_keys != [f"source029_resume_job_{i:03d}" for i in range(len(resume_keys))]:
        raise ValueError("Completed-speed resume evidence has a missing wrapper link")
    _verify_source029(binding.run_root.parent, before["source029_job"], before["source029_revision"],
        Path(evidence_paths["source029_revision"]), before["source029_previous_job"],
        Path(evidence_paths["source029_previous_job"]),
        resumed_jobs=[(Path(certificate["before"][key]["source_file"]), before[key]) for key in resume_keys],
        receipt_source_path=Path(certificate["before"]["source029_revision"]["source_file"])
            if "source_file" in certificate["before"]["source029_revision"] else None,
        prior_source_path=Path(certificate["before"]["source029_previous_job"]["source_file"])
            if "source_file" in certificate["before"]["source029_previous_job"] else None)
    actual_old_root = Path(before["source029_revision"]["old_source_root"]).resolve()
    old_contract = old_plan["integrated_continuation"]["lineage_manifest"]
    if (actual_old_root != old.source_root
            or before["source029_revision"]["old_contract_sha256"] != certificate["predecessor"]["campaign_plan_sha256"]
            or old_contract["sha256"] != old.sha256):
        raise ValueError("Source029 receipt belongs to another predecessor")
    target = json.loads((binding.source_root / production.artifact_files(27)["plan"]).read_text())
    maximum = min(20000 - totals["domain_consumed_updates"]["terrain"],
                  target["total_ceiling"] - totals["consumed_updates"]) - 10
    expected_course = dict(schema="ordinary-large-stair-introduction-v1", component="terrain",
        origin_actor_update=0, actor_allowance=min(target["domain_completion"]["schedule"]["end_actor_updates"]["terrain"], maximum))
    if certificate["course_introduction"] != expected_course or expected_course["actor_allowance"] < 1000:
        raise ValueError("Ordinary large-stair course lacks its bounded real remaining actor allowance")
    return old, checkpoint, selected


def authenticate_entry(binding, entry):
    pin = entry.get("completed_speed_predecessor_entry", {})
    certificate = _read_pin(pin, binding.run_root)
    file = Path(pin["file"]).resolve(strict=True)
    if file != (binding.run_root / DIRECTORY / "entry.json").resolve(strict=True):
        raise ValueError("Completed-speed entry is outside its immutable boundary directory")
    old, checkpoint, selected = _authenticated(certificate, binding)
    expected = dict(schema="v6-integrated-lineage-entry-v1", plan_sha256=binding.sha256,
        lineage_id=binding.read()["lineage_id"], checkpoint=deepcopy(certificate["old_entry"]["checkpoint"]),
        record=deepcopy(certificate["old_entry"]["record"]), completed_speed_predecessor_entry=deepcopy(pin))
    if entry != expected:
        raise ValueError("New entry substitutes the old authenticated REC certificate")
    return certificate, old, checkpoint, selected


def entry_from_report(binding, report):
    pin = report.get("completed_speed_predecessor_entry", {})
    certificate = _read_pin(pin, binding.run_root)
    entry = dict(schema="v6-integrated-lineage-entry-v1", plan_sha256=binding.sha256,
        lineage_id=binding.read()["lineage_id"], checkpoint=deepcopy(certificate["old_entry"]["checkpoint"]),
        record=deepcopy(certificate["old_entry"]["record"]), completed_speed_predecessor_entry=deepcopy(pin))
    authenticate_entry(binding, entry)
    return entry


def parent(binding, entry, reference, record):
    certificate, old, checkpoint, selected = authenticate_entry(binding, entry)
    if reference != selected["checkpoint"] or record != certificate["speed_record"]:
        raise ValueError("v27 terrain requires its exact completed v26 speed parent")
    return checkpoint, selected


def open_account(job, source_root):
    """Called only under the supervisor's existing exclusive account lock."""
    from . import integrated_production as production
    binding = production.binding_for(job.plan, source_root, job.root)
    old, old_plan = predecessor_binding(binding)
    plan_sha = lineage.digest(job.args.contract)
    ledger = json.loads(job.claim.read_text())
    state_path = job.root / "campaign_state.json"
    state = json.loads(state_path.read_text())
    directory = job.root / DIRECTORY
    certificate_path = directory / "entry.json"
    totals = account_updates(job.plan, ledger["attempts"])
    if (ledger.get("campaign_id") != CAMPAIGN_ID or ledger.get("run_directory") != str(job.root)
            or ledger.get("rollback_refunds_consumed_updates") is not False
            or ledger.get("contract_sha256") not in (static_pin(source_root)["campaign_plan_sha256"], plan_sha)
            or any(ledger.get(k) != v for k, v in totals.items())):
        raise ValueError("v27 account is not its non-refundable v26 predecessor")
    job.assert_no_live_workers()
    records = production.reconcile_records(job.plan, job.root, ledger["attempts"])
    if records != ledger["attempts"]:
        raise ValueError("Completed-speed boundary contains unaccounted work; seal the old owner first")
    if not certificate_path.exists():
        if ledger["contract_sha256"] == plan_sha:
            raise ValueError("v27 account lacks its immutable predecessor entry")
        source_job = _source029_job(job.root, source_root)
        entry = production.recovery_entry_admission(old, state)
        record = state["domains"]["speed"]
        reference = dict(path=record.get("candidate_checkpoint"), sha256=record.get("candidate_checkpoint_sha256"))
        _, selected = production.select_source(old_plan, old.source_root, job.root,
            dict(state, active_joint_domain="speed"), previous_selection=ledger.get("joint_learning_sources", {}).get("speed"))
        if selected["checkpoint"] != reference:
            raise ValueError("Completed-speed record is not its latest native own state")
        from . import integrated_runtime as runtime
        before_bytes = dict(account=job.claim.read_bytes(), state=state_path.read_bytes(),
            **{"source029_" + key: Path(pin["file"]).read_bytes() for key, pin in source_job.items()})
        before_paths = dict(account=job.claim, state=state_path,
            **{"source029_" + key: pin["file"] for key, pin in source_job.items()})
        before = {key: dict(file=str(directory / ("before_" + key + ".json")),
                           sha256=hashlib.sha256(value).hexdigest(),
                           **({"source_file": str(before_paths[key])} if key.startswith("source029_") else {}))
                  for key, value in before_bytes.items()}
        maximum = min(20000-totals["domain_consumed_updates"]["terrain"], job.plan["total_ceiling"]-totals["consumed_updates"])-10
        certificate = dict(schema=ENTRY_SCHEMA, predecessor=static_pin(source_root),
            new_lineage_id=binding.read()["lineage_id"], new_lineage_manifest_sha256=binding.sha256,
            before=before, old_entry=entry, speed_selection=selected, speed_record=deepcopy(record),
            predecessor_component_clocks=runtime.source_clocks(old, selected), account_at_boundary=totals,
            course_introduction=dict(schema="ordinary-large-stair-introduction-v1", component="terrain", origin_actor_update=0,
                actor_allowance=min(job.plan["domain_completion"]["schedule"]["end_actor_updates"]["terrain"], maximum)),
            PPO_updates_by_migration=0, qualification=None)
        # All real parent and source checks happen before the claim changes.
        _authenticated(certificate, binding, before_paths=before_paths)
        for key, value in before_bytes.items():
            _retain(directory / ("before_" + key + ".json"), value)
        _retain(certificate_path, production._bytes(certificate))
    certificate = json.loads(certificate_path.read_text())
    _authenticated(certificate, binding)
    pin = _pin(certificate_path)
    old_totals = certificate["account_at_boundary"]
    for key in ("consumed_updates", "actor_updates_consumed"):
        if totals[key] < old_totals[key]:
            raise ValueError("v27 restart refunds the retained completed-speed account")
    for name, record in _read_pin(certificate["before"]["account"], job.root)["attempts"].items():
        if records.get(name) != record:
            raise ValueError("v27 restart rewrites an old paid attempt")
    if state["domains"]["speed"] != certificate["speed_record"]:
        raise ValueError("v27 restart rewrites the completed v26 speed record")
    identity = dict(lineage_id=binding.read()["lineage_id"],
        plan_sha256=binding.sha256, origin_checkpoint_sha256=certificate["old_entry"]["checkpoint"]["sha256"])
    if ledger["contract_sha256"] == plan_sha and state.get("completed_speed_predecessor_entry") != pin:
        # Native persist commits the authoritative claim before state/report.
        # Recover only this exact initial one-way cut, before any new work.
        old_state = _read_pin(certificate["before"]["state"], job.root)
        old_account = _read_pin(certificate["before"]["account"], job.root)
        expected = dict(campaign_id=CAMPAIGN_ID, run_directory=str(job.root), contract_sha256=plan_sha,
            attempts=deepcopy(old_account["attempts"]), rollback_refunds_consumed_updates=False,
            integrated_continuation=identity, joint_learning_sources=deepcopy(old_state.get("joint_learning_sources", {})),
            **certificate["account_at_boundary"])
        if state != old_state or ledger != expected or records != old_account["attempts"]:
            raise ValueError("v27 partial commit is not the exact frozen initial claim→state boundary")
        state = dict(deepcopy(old_state), completed_speed_predecessor_entry=pin, integrated_continuation=identity)
    if ledger["contract_sha256"] == plan_sha and (ledger.get("integrated_continuation") != identity
            or state.get("integrated_continuation") != identity):
        raise ValueError("v27 restart changed its retained REC or new lineage identity")
    job.records = records
    job.report.update(deepcopy(state))
    for domain, selection in ledger.get("joint_learning_sources", {}).items():
        previous = job.report.setdefault("joint_learning_sources", {}).get(domain)
        if previous is not None and (previous["checkpoint_consumed_updates"] > selection["checkpoint_consumed_updates"]
                or previous["checkpoint_consumed_updates"] == selection["checkpoint_consumed_updates"]
                and previous["checkpoint"]["sha256"] != selection["checkpoint"]["sha256"]):
            raise ValueError("v27 recovery loses its durable selected learning-state floor")
        job.report["joint_learning_sources"][domain] = deepcopy(selection)
    job.report.update(contract_sha256=plan_sha, completed_speed_predecessor_entry=pin,
        integrated_continuation=identity, revision_migration=dict(
            schema="v26-completed-speed-to-v27-account-v1", predecessor_entry=pin,
            consumption="preserved_no_refund", qualification=None))
    job.persist()


def already_completed_speed(binding, report):
    entry = entry_from_report(binding, report)
    certificate, _, _, _ = authenticate_entry(binding, entry)
    if report["domains"]["speed"] != certificate["speed_record"]:
        raise ValueError("v27 must retain the exact old speed completion")
    return True


def bind_course(config, receipt, binding):
    """Bind the introduction once; later joint stages keep that same clock base."""
    from .ordinary_large_stair import row_settings
    certificate, _, _, _ = authenticate_entry(binding, receipt["entry_admission"])
    introduction = certificate["course_introduction"]
    result = deepcopy(config)
    found = 0
    for spec in result["skill_specs"].values():
        settings = spec.get("ordinary_large_stair_curriculum")
        if settings is not None:
            spec["ordinary_large_stair_curriculum"] = row_settings(settings["row_index"],
                origin_actor_update=introduction["origin_actor_update"], actor_allowance=introduction["actor_allowance"])
            found += 1
    if found != 8:
        raise ValueError("v27 downstream runtime must bind all eight ordinary stair rows")
    result["integrated_runtime_candidate"]["course_introduction"] = deepcopy(introduction)
    return result
