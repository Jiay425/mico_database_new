"""Unattended, checkpointed migration of microbe_abundance_standard."""
from __future__ import annotations

import subprocess
import sys
import time
import zlib
import os
import shlex

import paramiko

MYSQLDUMP = r"D:\MySQL\MySQL Server 8.0\bin\mysqldump.exe"
MAX_ID = 30_150_521
STEP = 1_000_000


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} must be supplied through the environment")
    return value


def _local_args() -> list[str]:
    return [
        MYSQLDUMP, "-h", os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "-P", os.environ.get("MYSQL_PORT", "3306"),
        "-u", os.environ.get("MYSQL_USERNAME", "root"),
        "--password=" + _required("MYSQL_PASSWORD"), "--single-transaction", "--quick",
        "--skip-triggers", "--no-create-info", "--hex-blob", "--set-gtid-purged=OFF",
        os.environ.get("MYSQL_DATABASE", "patient_data_manager"), "microbe_abundance_standard",
    ]


def connect():
    ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        _required("REMOTE_SSH_HOST"), username=_required("REMOTE_SSH_USER"),
        password=_required("REMOTE_SSH_PASSWORD"), timeout=30,
        look_for_keys=False, allow_agent=False,
    )
    return ssh


def remote_max_id() -> int:
    ssh = connect()
    sudo_password = _required("REMOTE_SUDO_PASSWORD")
    mysql_password = _required("REMOTE_MYSQL_PASSWORD")
    command = (
        f"printf '%s\\n' {shlex.quote(sudo_password)} | sudo -S -p '' docker exec mico-mysql "
        f"mysql -N -uroot {shlex.quote('-p' + mysql_password)} patient_data_manager "
        "-e 'SELECT COALESCE(MAX(standard_abundance_id),0) FROM microbe_abundance_standard'"
    )
    _, out, _ = ssh.exec_command(command, timeout=180)
    value = out.read().decode().strip().splitlines()[-1]
    ssh.close()
    return int(value)


def run_chunk(lower: int, upper: int) -> None:
    where = f"standard_abundance_id > {lower} AND standard_abundance_id <= {upper}"
    proc = None; channel = None; ssh = None
    try:
        ssh = connect()
        channel = ssh.get_transport().open_session()
        mysql_password = _required("REMOTE_MYSQL_PASSWORD")
        channel.exec_command(
            "sudo -S -p '' bash -lc " + shlex.quote(
                "gzip -dc | docker exec -i mico-mysql mysql --binary-mode=1 -uroot "
                + shlex.quote("-p" + mysql_password) + " patient_data_manager"
            )
        )
        channel.sendall((_required("REMOTE_SUDO_PASSWORD") + "\n").encode())
        proc = subprocess.Popen(_local_args() + [f"--where={where}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        compressor = zlib.compressobj(6, zlib.DEFLATED, 31)
        raw = sent = 0
        while True:
            chunk = proc.stdout.read(1024 * 1024)
            if not chunk: break
            raw += len(chunk)
            payload = compressor.compress(chunk)
            if payload: channel.sendall(payload); sent += len(payload)
        payload = compressor.flush()
        if payload: channel.sendall(payload); sent += len(payload)
        channel.shutdown_write()
        source_error = proc.stderr.read().decode(errors="replace")
        source_exit = proc.wait()
        remote_exit = channel.recv_exit_status()
        remote_error = channel.recv_stderr(10 * 1024 * 1024).decode(errors="replace")
        print(f"chunk {lower + 1}-{upper}: raw={raw} compressed={sent} source={source_exit} remote={remote_exit}", flush=True)
        if source_error: print(source_error, file=sys.stderr, flush=True)
        if remote_error: print(remote_error, file=sys.stderr, flush=True)
        if source_exit or remote_exit: raise RuntimeError(f"chunk failed: {lower + 1}-{upper}")
    finally:
        if proc is not None and proc.poll() is None: proc.terminate()
        if channel is not None: channel.close()
        if ssh is not None: ssh.close()


def main() -> None:
    lower = remote_max_id()
    while lower < MAX_ID:
        upper = min(lower + STEP, MAX_ID)
        for attempt in range(1, 8):
            try:
                run_chunk(lower, upper)
                break
            except Exception as exc:
                print(f"chunk {lower + 1}-{upper} attempt {attempt} failed: {exc}", flush=True)
                time.sleep(min(60, attempt * 10))
                lower = remote_max_id()
                if lower >= MAX_ID: break
                upper = min(lower + STEP, MAX_ID)
        else:
            raise RuntimeError("seven retries failed; checkpoint preserved")
        lower = remote_max_id()
    print("Remote standard abundance resume complete", flush=True)


if __name__ == "__main__": main()
