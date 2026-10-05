#!/usr/bin/env python3
"""Collect five completed frozen native probes without changing their contents."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from prepare_revision_018 import DOMAINS, OUT, load_evidence, read


def main():
    metadata = read(OUT / "source_candidate_018.json")
    verification = OUT.parent / "verification"
    domains, provenance = {}, {}
    for domain in DOMAINS:
        path = verification / f"native_{domain}_018_final.report.json.gz"
        compressed = path.read_bytes()
        raw = gzip.decompress(compressed)
        report = json.loads(raw)
        assert report["status"] == "engineering_probe_completed"
        assert report["domain"] == domain
        assert report["formal_updates_consumed"] == report["optimizer_steps"] == 0
        assert report["qualification"] is None
        domains[domain] = report
        provenance[domain] = dict(
            report=str(path), compressed_sha256=hashlib.sha256(compressed).hexdigest(),
            report_sha256=hashlib.sha256(raw).hexdigest(),
            host="Kaiser" if domain in ("terrain", "recovery") else "local",
            scope="one complete native engineering probe per domain; not behavior qualification")
    aggregate = dict(
        schema="v18-five-native-engineering-admission-v1", status="passed",
        source_commit=metadata["source_commit"], archive_sha256=metadata["archive_sha256"],
        assembled_at_utc=datetime.now(timezone.utc).isoformat(),
        formal_updates_consumed=0, qualification=None,
        scope="five unchanged full reports; separate engineering proof, no PPO or capability credit",
        domains=domains, provenance=provenance)
    path = verification / "native_population_018.json"
    pending = path.with_suffix(".json.pending")
    assert not path.exists(), "Final evidence already exists; inspect instead of replacing"
    with pending.open("x") as stream:
        json.dump(aggregate, stream, indent=2); stream.write("\n")
    # The actual admission loader checks runtime schema, archived source bytes,
    # baseline PT and contract, population, clocks and native effect per domain.
    try:
        evidence = load_evidence(native_path=pending)
    except Exception:
        aggregate["status"] = "failed_admission_not_deployed"
        rejected = path.with_suffix(".json.rejected")
        with rejected.open("x") as stream:
            json.dump(aggregate, stream, indent=2); stream.write("\n")
        pending.unlink()
        raise
    pending.rename(path)
    print(json.dumps(dict(status="passed", domains=list(domains),
        source_commit=metadata["source_commit"], evidence_sha256=evidence["evidence_sha256"])))


if __name__ == "__main__":
    main()
