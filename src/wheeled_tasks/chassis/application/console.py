#!/usr/bin/env python3
"""Kaiser terminal frontend for one existing formal training line."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import subprocess
import sys
import termios
import time
import tty
import unicodedata
import uuid

from wheeled_algo.training_control import (  # noqa: E402
    atomic_json, identity_alive, process_identity, read_json, utc_now,
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def option(argv, name):
    if argv.count(name) != 1 or argv.index(name) + 1 == len(argv):
        raise ValueError(f"启动参数必须且只能包含一次 {name}")
    return argv[argv.index(name) + 1]


def replace_option(argv, name, value):
    option(argv, name)
    argv[argv.index(name) + 1] = str(value)


def same_process(left, right):
    # exec changes argv, but not the kernel process identity.
    return bool(left and right) and all(left.get(k) == right.get(k)
        for k in ("pid", "uid", "start_ticks", "boot_id"))


class Controller:
    def __init__(self, config):
        self.path = Path(config).expanduser().resolve()
        self.config = read_json(self.path)
        if not self.config or self.config.get("schema") != "v7ctl-v1":
            raise ValueError(f"找不到有效配置：{self.path}")
        self.directory = self.path.parent
        self.session_path = self.directory / "session.json"

    def current(self):
        session = read_json(self.session_path, {})
        run = Path(session.get("run_directory", self.config["run_directory"]))
        status = read_json(run / "control/status.json", {})
        alive = identity_alive(status.get("identity"))
        launching = same_process(session.get("identity"),
            process_identity(session.get("identity", {}).get("pid")))
        return run, status, session, alive, launching

    def ledger(self):
        return read_json(self.config["ledger"])

    def latest_checkpoint(self, *, verify=False):
        rows = [row for row in self.ledger()["checkpoints"].values()
                if row["status"] == "registered"
                and row["metadata"]["attempt_id"] == self.config["attempt_id"]]
        rows.sort(key=lambda row: (row["metadata"]["paid_updates_attempt"], row["path"]), reverse=True)
        for row in rows:
            path = Path(row["path"])
            seal_path = path.with_suffix(".checkpoint.json")
            if not path.is_file() or not seal_path.is_file():
                if verify:
                    raise ValueError(f"最新登记检查点缺失；不静默回退：{path}")
                continue
            seal = read_json(seal_path)
            if verify and (digest(path) != row["checkpoint_sha256"]
                           or digest(seal_path) != row["seal_sha256"]
                           or seal.get("training_budget") != row["metadata"]
                           or seal.get("checkpoint_sha256") != row["checkpoint_sha256"]):
                raise ValueError(f"检查点校验失败；不静默回退：{path}")
            return {**row, "learning_updates": seal.get("learning_updates")}
        return None

    def snapshot(self):
        run, status, session, alive, launching = self.current()
        progress = read_json(run / "progress.json", {})
        completion = read_json(run / "completion.json", {})
        ledger = self.ledger()
        attempt = ledger["attempts"][self.config["attempt_id"]]
        paid = sum(row["attempt_id"] == self.config["attempt_id"] for row in ledger["reservations"])
        phase = status.get("phase", "starting") if alive else "starting" if launching else "stopped"
        if not alive and not launching:
            phase = "completed" if completion.get("status") == "completed" else "stopped"
            if self.config.get("blocked_reasons"):
                phase = "blocked"
        latest = self.latest_checkpoint()
        learning = status.get("learning_updates") if alive else None
        if learning is None and "parent_updates" in progress:
            learning = progress["parent_updates"] + progress.get("successful_updates", 0)
        if learning is None:
            learning = latest.get("learning_updates") if latest else None
        pending = read_json(run / "control/request.json", {})
        allocation = attempt["identity"]["requested_updates"]
        return {"phase": phase, "host": os.uname().nodename, "run_directory": str(run),
                "identity": status.get("identity") if alive else session.get("identity") if launching else None,
                "learning_updates": learning, "latest_checkpoint": latest,
                "paid_updates_global": len(ledger["reservations"]), "paid_updates_attempt": paid,
                "attempt_allocation": allocation,
                "remaining_updates": None if allocation is None else max(0, allocation - paid),
                "pending_action": pending.get("action") if alive and pending.get("id") != status.get("acknowledged_request") else None,
                "progress_updated_at": progress.get("updated_at"),
                "timing": progress.get("latest_timing", {}),
                "blocked_reasons": self.config.get("blocked_reasons", []),
                "log": session.get("log", self.config.get("log")),
                "deadline": ledger["campaign_policy"].get("absolute_deadline_utc"),
                "control": status, "completion": completion.get("status")}

    def request(self, action, timeout=60.):
        with lock(self.directory / "command.lock"):
            run, status, _, alive, _ = self.current()
            if not alive:
                raise ValueError("没有支持边界控制的存活训练进程；不会向旧 PID 发信号")
            if action == "pause" and status["phase"] == "paused":
                return status
            if status["phase"] in ("stopping", "completed", "failed", "export_failed"):
                raise ValueError("训练正在退出或已经失败，请等待完整保存结果")
            pending = read_json(run / "control/request.json", {})
            if (pending.get("identity") == status["identity"] and pending.get("action") == action
                    and pending.get("id") != status.get("acknowledged_request")):
                request = pending
            else:
                request = {"id": uuid.uuid4().hex, "identity": status["identity"],
                           "action": action, "requested_at": utc_now()}
                atomic_json(run / "control/request.json", request)
        until = time.monotonic() + timeout
        wanted = {"pause": "paused", "resume": "running", "stop": "stopped"}[action]
        while time.monotonic() < until:
            status = read_json(run / "control/status.json", {})
            if status.get("acknowledged_request") == request["id"]:
                if action != "stop" and status.get("phase") == wanted:
                    return status
                if action == "stop" and not identity_alive(request["identity"]):
                    completion = read_json(run / "completion.json", {})
                    if completion.get("status") != "stopped":
                        raise ValueError(f"退出未完整成功：{completion.get('status', '缺少完成回执')}")
                    self.latest_checkpoint(verify=True)
                    return status
            if not identity_alive(request["identity"]):
                raise ValueError("训练进程已退出；尚未确认此次请求完整完成，请查看 status/logs")
            time.sleep(.2)
        raise TimeoutError("请求仍在等待安全边界／保存，未强杀训练。请用 v7ctl status 查看。")

    def plan(self):
        if self.config.get("blocked_reasons"):
            raise ValueError("启动待修复：" + "；".join(self.config["blocked_reasons"]))
        launch = self.config["launch"]
        cwd = Path(launch["cwd"])
        for filename, expected in launch["files_sha256"].items():
            if digest(cwd / filename) != expected:
                raise ValueError(f"冻结源／启动依赖已变化：{filename}")
        if "src/wheeled_algo/training_control.py" not in launch["files_sha256"]:
            raise ValueError("冻结训练入口尚未接入安全边界控制")
        argv = list(launch["argv"])
        if "--transfer" in argv or any(flag in argv for flag in (
                "--consumed-updates", "--stage-actor-update", "--global-actor-update-offset",
                "--actor-updates-consumed", "--learning-rate-scale")):
            raise ValueError("普通启动不能转移权重、重置课程时钟或覆盖学习率／消费")
        if option(argv, "--rng-mode") != "inherit":
            raise ValueError("完整续训要求继承 RNG")
        if option(argv, "--budget-attempt-id") != self.config["attempt_id"]:
            raise ValueError("启动的预算身份不匹配")
        if Path(option(argv, "--budget-ledger")).resolve() != Path(self.config["ledger"]).resolve():
            raise ValueError("启动的预算账本不匹配")
        review_path = Path(option(argv, "--budget-review-file"))
        if digest(review_path) != option(argv, "--budget-review-sha256"):
            raise ValueError("预算审查文件 SHA 不匹配")
        review, ledger = read_json(review_path), self.ledger()
        for name, entry in ledger["attempts"].items():
            if review["attempts"].get(name) != entry["identity"]:
                raise ValueError("预算审查未保留现有训练／评测记录")
        deadline = review["absolute_deadline_utc"]
        if deadline is not None and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
            raise ValueError("训练截止时间已到；CLI 不延长预算")
        latest = self.latest_checkpoint(verify=True)
        if latest is None:
            raise ValueError("没有本次正式训练的已登记完整检查点")
        contract = Path(option(argv, "--contract"))
        contract_sha = digest(cwd / contract)
        seal = read_json(Path(latest["path"]).with_suffix(".checkpoint.json"))
        if seal["contract_sha256"] != contract_sha:
            raise ValueError("普通续训不能更改合同；请先完成已审查的迁移")
        if read_json(cwd / contract).get("formal_stage_optimizer") and option(argv, "--stage") != "terrain":
            raise ValueError("阶段／优化器继承不明确，不能自动启动")
        remaining = self.snapshot()["remaining_updates"]
        cap = review["paid_update_cap"]
        if cap is not None:
            available = cap - len(ledger["reservations"])
            remaining = available if remaining is None else min(remaining, available)
        if remaining is not None and remaining <= 0:
            raise ValueError("既有额度已用完，CLI 不新增额度")
        # A finite invocation window is not a new campaign allocation. Keep
        # its reviewed launch value when the main attempt has no budget cap.
        window = int(option(argv, "--updates")) if remaining is None else remaining
        if not 1 <= window <= 100000:
            raise ValueError("启动窗口必须为1到100000次；持续运行入口另行核验")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = Path(self.config["run_directory"]).parent / f"{self.config['attempt_id']}_resume_{stamp}_{uuid.uuid4().hex[:6]}"
        replace_option(argv, "--run-dir", run)
        replace_option(argv, "--resume", latest["path"])
        replace_option(argv, "--updates", window)
        return {"argv": argv, "cwd": str(cwd), "runtime": launch["runtime"],
                "run_directory": str(run), "checkpoint": latest,
                "remaining_updates": remaining, "execution_window_updates": window,
                "physical_episodes_restart": True}

    def start(self, timeout=60., dry_run=False):
        run, status, _, alive, launching = self.current()
        if alive:
            if status["phase"] == "paused" or read_json(run / "control/request.json", {}).get("action") == "pause":
                return self.request("resume", timeout)
            return {"phase": status["phase"], "message": "现有进程保持运行，不重复启动"}
        if launching:
            return {"phase": "starting", "message": "已有启动进程，不重复启动"}
        with lock(self.directory / "command.lock"):
            if any(self.current()[3:]):
                raise ValueError("另一个命令已启动训练")
            for process in Path("/proc").glob("[0-9]*"):
                identity = process_identity(int(process.name))
                if not identity or identity["uid"] != os.getuid():
                    continue
                argv = identity["argv"]
                if any(Path(arg).name == "train_chassis.py" for arg in argv) and "--budget-ledger" in argv:
                    if Path(option(argv, "--budget-ledger")).resolve() == Path(self.config["ledger"]).resolve():
                        raise ValueError(f"此正式训练账本已有进程 PID {identity['pid']}，不会重复启动")
            plan = self.plan()
            if dry_run:
                return plan
            # The budget worker remains the final lock/identity authority. Check
            # its existing flock before launching another heavyweight simulator.
            ledger = Path(self.config["ledger"])
            attempt_hash = hashlib.sha256(json.dumps(self.config["attempt_id"],
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            lease = ledger.parent / (ledger.name + ".attempt-" + attempt_hash + ".lock")
            with lock(lease):
                pass
            log = self.directory / (Path(plan["run_directory"]).name + ".log")
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
            env.update(OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
            command = ["/bin/bash", "-c", 'source "$1"; shift; exec "$@"',
                       "v7ctl", plan["runtime"], *plan["argv"]]
            with log.open("xb") as output:
                child = subprocess.Popen(command, cwd=plan["cwd"], env=env,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True)
            identity = process_identity(child.pid)
            session = {**plan, "identity": identity, "started_at": utc_now(), "log": str(log)}
            atomic_json(self.session_path, session)
            atomic_json(self.directory / (Path(plan["run_directory"]).name + ".launch.json"), session)
            return {"phase": "starting", "pid": child.pid, "run_directory": plan["run_directory"],
                    "message": "已启动；真实 PPO 更新以 status 和训练日志为准"}


LABELS = {"running": ("训练中", "32"), "paused": ("驻留暂停", "33"),
          "starting": ("初始化", "36"), "saving": ("保存中", "33"),
          "stopping": ("保存退出中", "33"), "stopped": ("已停止", "37"),
          "blocked": ("待修复 · 未启动", "33"), "completed": ("阶段完成", "32"),
          "failed": ("运行失败", "31"), "export_failed": ("导出失败", "31")}


def width(text):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def shorten(text, limit):
    text = str(text)
    if width(text) <= limit:
        return text
    while width(text) > max(0, limit - 1):
        text = text[:-1]
    return text + "…"


def render(snapshot, *, color=None, columns=None, message=""):
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ if color is None else color
    columns = max(32, min(columns or shutil.get_terminal_size((88, 24)).columns, 100))
    inner = columns - 4
    paint = lambda s, code: f"\033[{code}m{s}\033[0m" if color else s
    lines = [paint("╭" + "─" * (columns - 2) + "╮", "36")]

    def row(text="", code=None):
        text = shorten(text, inner)
        padded = text + " " * (inner - width(text))
        lines.append(paint("│ ", "36") + (paint(padded, code) if code else padded) + paint(" │", "36"))

    row("V7  /  KAISER     训练控制台", "1;36")
    label, tone = LABELS.get(snapshot["phase"], (snapshot["phase"], "37"))
    row(f"● {label}     {datetime.now().strftime('%m-%d %H:%M:%S')}", tone)
    row()
    row(f"学习进度  {snapshot['learning_updates'] or '—'}     已消费  {snapshot['paid_updates_global']:,}", "1")
    total, done = snapshot["attempt_allocation"], snapshot["paid_updates_attempt"]
    if total is None:
        row(f"本任务已消费  {done:,} updates · 无预算上限", "36")
        row("启动窗口沿用已冻结命令；历史消费继续累计。")
    else:
        size = max(6, min(30, inner - 24))
        filled = min(size, int(size * done / max(1, total)))
        row(f"{'━' * filled}{'─' * (size - filled)}  {done:,} / {total:,}", "36")
        row(f"本段剩余  {snapshot['remaining_updates']:,} updates")
    cp = snapshot.get("latest_checkpoint")
    row(f"已保存    {cp['learning_updates'] if cp else '—'}  ·  {(cp or {}).get('checkpoint_sha256', '—')[:12]}")
    if cp and snapshot.get("learning_updates") and cp.get("learning_updates"):
        gap = snapshot["learning_updates"] - cp["learning_updates"]
        if gap > 0:
            row(f"保存间隔  {gap} 次更新（中断后不能退款）", "33")
    identity = snapshot.get("identity") or {}
    row(f"主机      {snapshot['host']}     PID {identity.get('pid', '—')}")
    row(f"当前目录  {Path(snapshot['run_directory']).name}")
    if snapshot.get("pending_action"):
        row(f"待处理    {snapshot['pending_action']} · 等待更新边界", "33")
    row()
    for reason in snapshot["blocked_reasons"]:
        row("! " + reason, "33")
    if snapshot["blocked_reasons"]:
        row()
    row("p 暂停    r 启动/继续    s 保存退出    q 关闭面板", "1")
    row("暂停保留显存；保存退出释放显存，重启重建物理回合。", "2")
    row("历史消费始终保留；按当前已迁移配置检查限制。", "2")
    if message:
        row(message, "33")
    lines.append(paint("╰" + "─" * (columns - 2) + "╯", "36"))
    return "\n".join(lines)


def dashboard(controller):
    original = termios.tcgetattr(sys.stdin.fileno())
    message = ""
    try:
        tty.setcbreak(sys.stdin.fileno())
        print("\033[?1049h\033[?25l", end="", flush=True)
        while True:
            print("\033[H\033[J" + render(controller.snapshot(), message=message), end="", flush=True)
            if not select.select([sys.stdin], [], [], 2.)[0]:
                continue
            key = sys.stdin.read(1).lower()
            if key in ("q", "\x03", "\x04"):
                break
            try:
                if key in ("p", "s", "r"):
                    # Submit without blocking the dashboard for a long rollout.
                    if key == "r":
                        result = controller.start(timeout=.05)
                    else:
                        result = controller.request("pause" if key == "p" else "stop", timeout=.05)
                    message = result.get("message", "请求已完成")
            except TimeoutError:
                message = "请求已提交；等待更新边界，训练不会被强杀。"
            except (ValueError, OSError) as error:
                message = str(error)
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original)
        print("\033[?25h\033[?1049l", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="V7 训练控制：安全边界暂停、完整学习状态续训")
    parser.add_argument("--config", default=os.environ.get("V7CTL_CONFIG", "~/.config/v7ctl/config.json"))
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    parser.add_argument("command", nargs="?", choices=("status", "pause", "start", "resume", "stop", "logs", "plan"))
    parser.add_argument("--timeout", type=float, default=60., help="等待回执秒数；超时不强杀")
    parser.add_argument("--follow", "-f", action="store_true", help="持续读取日志")
    args = parser.parse_args()
    try:
        controller = Controller(args.config)
        if args.command is None and sys.stdin.isatty() and sys.stdout.isatty() and not args.json:
            dashboard(controller)
            return 0
        command = args.command or "status"
        if command == "logs":
            path = controller.snapshot()["log"]
            if not path:
                raise ValueError("没有日志路径")
            return subprocess.call(["tail", "-n", "80", *(["-F"] if args.follow else []), "--", path])
        if command == "status":
            value = controller.snapshot()
        elif command == "plan":
            value = controller.plan()
        elif command in ("start", "resume"):
            value = controller.start(timeout=args.timeout)
        else:
            value = controller.request(command, timeout=args.timeout)
        if args.json:
            print(json.dumps(value, ensure_ascii=False, indent=2))
        elif command == "plan":
            print(shlex.join(value["argv"]))
            print("完整学习状态续训；物理回合重新初始化。")
        else:
            print(render(controller.snapshot(), message=value.get("message", "")))
        return 0
    except (ValueError, OSError, KeyError, TimeoutError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False) if args.json else f"V7 · {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已退出控制台；未停止训练。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
