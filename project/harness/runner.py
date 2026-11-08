"""Run a model-written script locally (0 tokens) inside the task directory."""
import os
import subprocess
import sys
import tempfile


def run_script(code: str, cwd: str, timeout: int = 30):
    fd, path = tempfile.mkstemp(suffix=".py", dir="/tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": "/tmp",
           "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
    try:
        p = subprocess.run([sys.executable, path], cwd=cwd, env=env, timeout=timeout,
                           capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr)[-700:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
