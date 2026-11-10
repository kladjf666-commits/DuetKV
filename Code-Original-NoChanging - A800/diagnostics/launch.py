"""Reserve the benchmark card and optionally hand it back at a formal action boundary."""
import argparse
import contextlib
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def formal_owner(lock):
    stat = os.fstat(lock.fileno())
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if len(fields)<6 or fields[1:4]!=["FLOCK", "ADVISORY", "WRITE"]:
            continue
        major, minor, inode = fields[5].split(":")
        if (int(major,16), int(minor,16), int(inode)) != (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino):
            continue
        pid = int(fields[4])
        if pid <= 1:
            continue
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if len(argv)>1 and Path(os.fsdecode(argv[0])).name.startswith("python") and any(Path(os.fsdecode(i)).name=="a800_formal.py" for i in argv[1:]):
            return pid
    raise SystemExit("Could not verify the formal queue owner. Use this probe after the GPU is idle.")


def process_status(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], int(fields[19])  # state (field 3), start time in ticks (field 22)
    except (FileNotFoundError, ProcessLookupError):
        return "Z", None


def still_running(pid, birth):
    state, started = process_status(pid)
    return birth is not None and started == birth and state not in ("Z", "X")


@contextlib.contextmanager
def queue_signaller(lock, pid, birth):
    fd = None
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            fd = os.pidfd_open(pid)
        except OSError as exc:
            if exc.errno not in (errno.ENOSYS, errno.EPERM, errno.EACCES):
                raise

    def send(sig):
        # Recheck identity even when falling back to POSIX signals on older Python builds.
        if not still_running(pid, birth):
            raise ProcessLookupError(f"Original scheduler {pid} has exited")
        if formal_owner(lock) != pid:
            raise RuntimeError("Formal lock owner changed; refusing to signal another process")
        if fd is not None:
            try:
                signal.pidfd_send_signal(fd, sig)
                return
            except OSError as exc:
                if exc.errno not in (errno.ENOSYS, errno.EPERM, errno.EACCES):
                    raise
        os.kill(pid, sig)

    try:
        yield send
    finally:
        if fd is not None:
            os.close(fd)


@contextlib.contextmanager
def reserve_formal(lock, handoff):
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if not handoff:
            raise SystemExit("Formal queue active. Use --after-current to run after its current action and resume it.")
    else:
        yield
        return
    pid = formal_owner(lock)
    initial, birth = process_status(pid)
    if birth is None or initial in ("Z", "X"):
        raise SystemExit("Formal scheduler ended before handoff. Run the same command again.")
    if initial in ("T","t"):
        raise SystemExit("Formal queue already paused externally; leaving its state alone.")
    with queue_signaller(lock, pid, birth) as send:
        paused = False
        try:
            send(signal.SIGSTOP)
            paused = True
            while True:
                status, started = process_status(pid)
                if started != birth or status in ("Z", "X"):
                    raise SystemExit("Formal scheduler exited during handoff. Retry the probe.")
                if status in ("T","t"):
                    break
                time.sleep(.1)
            # Existing worker completes normally. No competition between it and the probe.
            children = Path(f"/proc/{pid}/task/{pid}/children")
            if children.exists():
                active = [(int(n), process_status(int(n))[1]) for n in children.read_text().split()]
                if active:
                    print(f"Formal scheduler {pid} holding at the current action; waiting for its worker. It will resume automatically.",flush=True)
                while any(still_running(n, stamp) for n, stamp in active):
                    time.sleep(1)
            yield
        finally:
            if paused:
                try:
                    send(signal.SIGCONT)
                    print(f"Resumed formal scheduler {pid}.",flush=True)
                except ProcessLookupError:
                    pass


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--after-current",action="store_true")
    ap.add_argument("--prompt-mma",action="store_true",help="Test the SM80 packed-prompt candidate")
    ap.add_argument("--resume",action="store_true",help="Reuse completed measurements from the prompt-policy failure")
    ap.add_argument("--late",action="store_true")
    args=ap.parse_args()
    env=dict(os.environ)
    base=Path(env.get("PHASEKV_BENCH","/data/phasekv_a800_s1"))
    py=Path(env.get("A800_MAIN_PYTHON") or Path(env.get("A800_ENVS", str(base/".envs-offline")))/"main/bin/python")
    if not py.is_file():
        raise SystemExit(f"Missing existing offline main environment: {py}")
    model=Path(env.get("PHASEKV_MODEL","/data/Llama-3.1-8B-Instruct/snapshots/master"))
    if not (model/"model.safetensors.index.json").is_file():
        raise SystemExit(f"Missing model: {model}")
    for line in (HERE/"SHA256SUMS").read_text().splitlines():
        expected,name=line.split("  ",1)
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=expected:
            raise SystemExit(f"Probe package checksum mismatch: {name}")
    gpu=env.get("CUDA_VISIBLE_DEVICES","0")
    if not gpu or "," in gpu or "/" in gpu:
        raise SystemExit("Select exactly one GPU with CUDA_VISIBLE_DEVICES")
    env.update(CUDA_VISIBLE_DEVICES=gpu,PHASEKV_MODEL=str(model),HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1",PYTHONUNBUFFERED="1",OMP_NUM_THREADS=env.get("OMP_NUM_THREADS","4"))
    formal=Path(env.get("A800_FORMAL_OUT","/data/phasekv-results/a800_formal_v1"))
    formal.mkdir(parents=True,exist_ok=True)
    output_name = "a800_prompt_mma" if args.prompt_mma else "a800_reason"
    root=Path(env.get("A800_PROBE_OUT", str(Path("/data/phasekv-results") / output_name)))
    root.mkdir(parents=True,exist_ok=True)
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    with contextlib.ExitStack() as locks:
        handoff_lock = locks.enter_context(Path(f"/tmp/phasekv-a800-handoff-{gpu}.lock").open("a"))
        try:
            fcntl.flock(handoff_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A probe is already waiting or running on this GPU.")
        job = locks.enter_context((formal/"queue.lock").open("a"))
        locks.enter_context(reserve_formal(job,args.after_current))
        card = locks.enter_context(Path(f"/tmp/phasekv-a800-single-{gpu}.lock").open("a"))
        try:
            fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another GPU benchmark owns the card; no results were started.")
        out=root/(time.strftime("%Y%m%d_%H%M%S")+"_"+str(os.getpid()))
        out.mkdir()
        print(f"Starting B3 probe. Log and summary: {out}",flush=True)
        cmd=[str(py),"-u",str(HERE/"probe.py"),"--out",str(out)]+(["--late"] if args.late else [])
        if args.resume:
            cmd.append("--resume")
        if args.prompt_mma:
            cmd.append("--prompt-mma")
        with (out/"run.log").open("w",encoding="utf-8",buffering=1) as log:
            with subprocess.Popen(cmd,env=env,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,errors="replace",bufsize=1) as child:
                try:
                    for line in child.stdout:
                        print(line,end="",flush=True)
                        log.write(line)
                    return child.wait()
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait()


if __name__=="__main__":
    sys.exit(main())
