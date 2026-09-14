#!/usr/bin/env python3
"""Deploy a committed B1 code tree and stable periodic A snapshot into an isolated experiment."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile

from common import CODE_DIRECTORY, ControlSSH, add_ssh_arguments, absolute_remote_path
from deploy import committed_files
from pipeline_common import B_CONTRACT, remote_capture
from wheeled_algo.v40_job import JobError, json_bytes

UNPACK = r'''
import hashlib,json,pathlib,tarfile
p=PARAMS
root=pathlib.Path(p['root'])
bundle=root/'bundle.tar.gz'
if hashlib.sha256(bundle.read_bytes()).hexdigest()!=p['sha256']: raise ValueError('bundle SHA mismatch')
with tarfile.open(bundle,'r:gz') as archive:
    entries=archive.getmembers()
    names=[e.name for e in entries]
    manifest=json.load(archive.extractfile('pipeline.json'))
    if len(names)!=len(set(names)) or set(names)!=set(manifest['files'])|{'pipeline.json'}:
        raise ValueError('bundle member mismatch')
    for entry in entries:
        relative=pathlib.PurePosixPath(entry.name)
        if not entry.isfile() or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unconfined entry')
        if entry.name=='pipeline.json': continue
        if relative.parts[0] not in ('isaac_wheeled_rl_train','parent'): raise ValueError('wrong member root')
        data=archive.extractfile(entry).read()
        if {'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}!=manifest['files'][entry.name]:
            raise ValueError('member SHA mismatch')
        path=root/entry.name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as out: out.write(data)
    with (root/'pipeline.json').open('x') as out: json.dump(manifest,out,indent=2)
print(json.dumps({'deployed':True,'code_dir':str(root/'isaac_wheeled_rl_train'),'experiment':str(root),'training_started':False}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--a-run", required=True)
    parser.add_argument("--checkpoint", help="original periodic filename; default latest stable at capture time")
    parser.add_argument("--experiment")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--inspect-parent", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    add_ssh_arguments(parser)
    args = parser.parse_args()
    client = ControlSSH(args.host, args.control_path, args.ssh_port, timeout=180)
    absolute_remote_path(args.a_run)
    if args.inspect_parent:
        print(json.dumps(remote_capture(client, args.a_run, args.checkpoint), indent=2))
        return
    if not args.experiment or not args.bundle:
        parser.error("experiment and local bundle path required")
    experiment = Path(absolute_remote_path(args.experiment))
    if experiment.parent != Path("/home/kaiser/robot-rl-sim60/experiments") or not experiment.name.startswith("b1-pilot-"):
        raise JobError("use a new /home/kaiser/robot-rl-sim60/experiments/b1-pilot-<commit> container")
    commit, files = committed_files(args.commit)
    required = (B_CONTRACT, "src/wheeled_algo/v40_stage_transfer.py", "scripts/round3/pipeline_pilot.py",
                "scripts/round3/pipeline_common.py", "scripts/round3/pipeline_follow.py", "scripts/round3/pipeline_handoff.py")
    if any(CODE_DIRECTORY + "/" + name not in files for name in required):
        raise JobError("commit does not yet contain complete B1 pipeline/env/transfer code")
    entry = files[CODE_DIRECTORY + "/scripts/train_v40.py"]
    if b"--stage-transfer" not in entry or b"--source-checkpoint-sha256" not in entry:
        raise JobError("committed entry lacks stage-transfer flags")
    target = json.loads(files[CODE_DIRECTORY + "/" + B_CONTRACT])
    if target.get("round3", {}).get("stage") != "B1":
        raise JobError("target is not B1")
    parent = remote_capture(client, args.a_run, args.checkpoint)
    manifest = {"schema_version": 1, "classification": "experiment_not_promoted", "git_commit": commit,
                "parent": parent, "target_contract_sha256": hashlib.sha256(json.dumps(target, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
    if not args.deploy:
        print(json.dumps({**manifest, "dry_run": True, "experiment": str(experiment), "training_started": False}, indent=2))
        return
    for name, record in parent["files"].items():
        with client.download(args.a_run + "/" + name) as stream:
            data = stream.read(record["size"] + 1)
        if len(data) != record["size"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise JobError("source copy SHA mismatch: " + name)
        files["parent/" + name] = data
    after = remote_capture(client, args.a_run, parent["checkpoint"])
    if after != parent:
        raise JobError("source parent changed during copy")
    manifest["files"] = {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()}
    files["pipeline.json"] = json_bytes(manifest)
    with args.bundle.open("xb") as target_file, gzip.GzipFile(fileobj=target_file, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o644
                archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    script = "from pathlib import Path;p=Path(" + repr(str(experiment)) + ");p.parent.mkdir(parents=True,exist_ok=True);p.mkdir()"
    client.exec(shlex.join(["python3", "-c", script]))
    client.upload(args.bundle, str(experiment / "bundle.tar.gz"))
    script = UNPACK.replace("PARAMS", repr({"root": str(experiment), "sha256": digest}), 1)
    print(client.exec(shlex.join(["python3", "-c", script]), timeout_ms=180000))


if __name__ == "__main__":
    main()
