"""Run model-written code in a disposable copy of the task directory.

The model's script never touches the real task dir: it runs in a temp copy with
CPU/memory/file-size rlimits and a wall-clock timeout. Only whitelisted files
(answer.txt, changed non-test python modules for coding tasks) are copied back.

This is not a container: it protects task data and tests from accidental damage
by a confused model, not from a determined attacker. The competition image
already runs with --read-only, --cap-drop ALL and --pids-limit.

Fallback: if the task dir is larger than SANDBOX_MAX_MB (default 256), we run in
place with the same rlimits to avoid filling the 2g /tmp tmpfs.
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile

MAX_COPY_MB = float(os.environ.get("SANDBOX_MAX_MB", "256"))
TAIL_CHARS = 4000
MEM_BYTES = int(float(os.environ.get("SANDBOX_MEM_MB", "4096")) * 1024 * 1024)
FSIZE_BYTES = int(float(os.environ.get("SANDBOX_FSIZE_MB", "32")) * 1024 * 1024)

# Runs in the child *after* fork+exec, so it is safe with a threaded parent
# (preexec_fn is not). Then execs the real script/tests.
_LIMITER = r"""
import os, resource, sys
cpu, mem, fsize = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
def lim(which, value):
    try:
        resource.setrlimit(which, (value, value))
    except (ValueError, OSError):
        pass
lim(resource.RLIMIT_CPU, cpu)
lim(resource.RLIMIT_AS, mem)
lim(resource.RLIMIT_FSIZE, fsize)
if sys.platform.startswith("linux"):
    lim(resource.RLIMIT_NPROC, 128)
os.execv(sys.argv[4], sys.argv[4:])
"""

TESTISH = re.compile(r"^(test_.*\.py|.*_test\.py|conftest\.py|pytest\.ini|tox\.ini|setup\.cfg|run_tests\.py)$")


def _is_testish(rel: str) -> bool:
    base = os.path.basename(rel)
    return bool(TESTISH.match(base)) or "tests" in rel.split(os.sep)


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def _tail(path: str, limit: int = TAIL_CHARS) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def run_limited(argv, cwd: str, timeout: float = 30, env: dict | None = None):
    """Run argv with rlimits + a wall timeout. Returns (returncode, output tail)."""
    fd, log_path = tempfile.mkstemp(prefix="run_", suffix=".log", dir=tempfile.gettempdir())
    os.close(fd)
    cpu = max(1, int(timeout) + 2)
    wrapped = [sys.executable, "-c", _LIMITER, str(cpu), str(MEM_BYTES), str(FSIZE_BYTES)] + list(argv)
    base_env = {"PATH": os.environ.get("PATH", ""), "HOME": "/tmp", "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    if env:
        base_env.update(env)
    try:
        with open(log_path, "wb") as out:
            p = subprocess.Popen(wrapped, cwd=cwd, env=base_env, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except OSError:
                    pass
                p.wait()
                rc = 124
        text = _tail(log_path)
        if rc == 124:
            text += "\n[timeout]"
        return rc, text
    finally:
        try:
            os.remove(log_path)
        except OSError:
            pass


class Sandbox:
    """A throw-away working copy of one task directory."""

    def __init__(self, task_dir: str, timeout: float = 30):
        self.orig = os.path.abspath(task_dir)
        self.timeout = timeout
        self.base = tempfile.mkdtemp(prefix="sbx_", dir=tempfile.gettempdir())
        self.isolated = _dir_size(self.orig) <= MAX_COPY_MB * 1024 * 1024
        if self.isolated:
            self.work = os.path.join(self.base, "task")
            try:
                shutil.copytree(self.orig, self.work, symlinks=False,
                                ignore=shutil.ignore_patterns("__pycache__", ".git"))
            except (OSError, shutil.Error):
                self.isolated, self.work = False, self.orig
        else:
            self.work = self.orig
        # Snapshot test-ish files in memory so we can undo tampering even in
        # the in-place fallback (and after a script deletes them).
        self.tests = {}
        for root, _, files in os.walk(self.orig):
            for fn in files:
                p = os.path.join(root, fn)
                if _is_testish(os.path.relpath(p, self.orig)):
                    try:
                        with open(p, "rb") as f:
                            self.tests[os.path.relpath(p, self.orig)] = f.read()
                    except OSError:
                        pass
        self._rm_answer()

    @property
    def dir(self) -> str:
        return self.work

    def _rm_answer(self) -> None:
        try:
            os.remove(os.path.join(self.work, "answer.txt"))
        except OSError:
            pass

    def run(self, code: str, timeout: float | None = None):
        """Run a python script in the sandbox. Returns (rc, output tail)."""
        tmpdir = self.base if self.isolated else tempfile.gettempdir()
        fd, path = tempfile.mkstemp(prefix="script_", suffix=".py", dir=tmpdir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            return run_limited([sys.executable, path], self.work, timeout or self.timeout)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def run_command(self, argv, timeout: float = 20):
        return run_limited(argv, self.work, timeout=timeout)

    def read_answer(self) -> str:
        try:
            with open(os.path.join(self.work, "answer.txt"), encoding="utf-8", errors="replace") as f:
                return f.read().strip()
        except OSError:
            return ""

    def restore_tests(self, extra=()) -> bool:
        """Restore all test-ish files from the pre-run snapshot. True if any changed."""
        changed = False
        wanted = set(self.tests) | {os.path.normpath(p) for p in extra}
        for rel in wanted:
            rel = os.path.normpath(rel)
            if rel.startswith("..") or os.path.isabs(rel):
                continue
            src = os.path.join(self.orig, rel)
            dst = os.path.join(self.work, rel)
            want = self.tests.get(rel)
            if want is None:
                try:
                    with open(src, "rb") as f:
                        want = f.read()
                except OSError:
                    continue
            try:
                with open(dst, "rb") as f:
                    have = f.read()
            except OSError:
                have = None
            if have != want:
                changed = True
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(want)
        return changed

    def changed_python(self):
        """Relative paths of non-test .py files that differ from the original."""
        out = []
        if not self.isolated:
            return out
        for root, dirs, files in os.walk(self.work):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, self.work)
                if _is_testish(rel):
                    continue
                orig = os.path.join(self.orig, rel)
                try:
                    with open(p, "rb") as f:
                        new = f.read()
                except OSError:
                    continue
                try:
                    with open(orig, "rb") as f:
                        old = f.read()
                except OSError:
                    old = None
                if new != old:
                    out.append(rel)
        return out

    def commit(self, rel_paths=("answer.txt",)) -> None:
        """Copy whitelisted files from the sandbox back to the real task dir."""
        if not self.isolated:
            return  # script already ran in place
        for rel in rel_paths:
            rel = os.path.normpath(rel)
            if rel.startswith("..") or os.path.isabs(rel):
                continue
            src = os.path.join(self.work, rel)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(self.orig, rel)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copy2(src, dst)

    def cleanup(self) -> None:
        if self.base:
            shutil.rmtree(self.base, ignore_errors=True)


def run_script(code: str, cwd: str, timeout: int = 30):
    """Backward-compatible helper: run one script, commit only answer.txt."""
    sb = Sandbox(cwd, timeout=timeout)
    try:
        rc, out = sb.run(code)
        if rc == 0 and sb.read_answer():
            sb.commit(("answer.txt",))
        return rc, out
    finally:
        sb.cleanup()
