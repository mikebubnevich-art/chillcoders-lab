#!/usr/bin/env python3
"""Collect a registry harness report and print its SHA-256 digest.

Pure stdlib, works on macOS, Linux and Windows. The report contains all
JIUWENSWARM_* environment variables and a bounded command-line ancestry.
Prints exactly two lines: compact JSON, then the digest of those exact bytes.
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import os
import subprocess
import sys

MAX_TREE_CHARS = 600
MAX_CHAIN = 16
_CANARY = "hrpt-canary-7c1f"


def _proc_map_linux() -> dict[int, tuple[int, str]] | None:
    """Return {pid: (ppid, cmdline)} from /proc, if available."""
    try:
        mapping: dict[int, tuple[int, str]] = {}
        for entry in glob.glob("/proc/[0-9]*"):
            pid = int(entry.rsplit("/", 1)[1])
            with open(f"{entry}/stat", encoding="utf-8", errors="replace") as fh:
                stat = fh.read()
            head, _, rest = stat.rpartition(")")
            fields = rest.split()
            if len(fields) < 2:
                continue
            cmd = ""
            try:
                with open(f"{entry}/cmdline", "rb") as fh:
                    cmd = b" ".join(fh.read().split(b"\0")).decode("utf-8", "replace").strip()
            except OSError:
                pass
            if not cmd:
                cmd = head.partition("(")[2].strip()
            mapping[pid] = (int(fields[1]), cmd)
        return mapping or None
    except Exception:
        return None


def _proc_map_windows() -> dict[int, tuple[int, str]] | None:
    """Return {pid: (ppid, image_name)} via Toolhelp32 on Windows."""
    if os.name != "nt":
        return None
    try:
        from ctypes import wintypes

        class PE32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        k32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PE32W)]
        k32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PE32W)]
        snap = k32.CreateToolhelp32Snapshot(2, 0)
        if snap in (None, 0, ctypes.c_size_t(-1).value):
            return None
        mapping: dict[int, tuple[int, str]] = {}
        try:
            entry = PE32W()
            entry.dwSize = ctypes.sizeof(PE32W)
            have = k32.Process32FirstW(snap, ctypes.byref(entry))
            while have:
                mapping[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile)
                have = k32.Process32NextW(snap, ctypes.byref(entry))
        finally:
            k32.CloseHandle(ctypes.c_void_p(snap))
        return mapping or None
    except Exception:
        return None


def _proc_map_ps() -> dict[int, tuple[int, str]] | None:
    """Return {pid: (ppid, command)} using ps, for POSIX systems without /proc."""
    try:
        done = subprocess.run(["ps", "-axo", "pid=,ppid=,command="],
                              capture_output=True, text=True, timeout=5)
        if done.returncode != 0:
            return None
        mapping: dict[int, tuple[int, str]] = {}
        for line in done.stdout.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                mapping[int(parts[0])] = (int(parts[1]), parts[2])
        return mapping or None
    except Exception:
        return None


def _ancestry(mapping: dict[int, tuple[int, str]]) -> str:
    """Return bounded ancestor command lines, root first."""
    chain: list[str] = []
    pid = os.getppid()
    seen: set[int] = set()
    while pid and pid not in seen and len(chain) < MAX_CHAIN:
        seen.add(pid)
        entry = mapping.get(pid)
        if entry is None:
            break
        pid, cmd = entry
        if cmd and _CANARY not in cmd:
            chain.append(cmd.strip()[:200])
    return " | ".join(reversed(chain))[:MAX_TREE_CHARS]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    env = {key.upper(): value for key, value in os.environ.items()
           if key.upper().startswith("JIUWENSWARM_")}
    mapping = _proc_map_linux() or _proc_map_windows() or _proc_map_ps() or {}
    report = json.dumps({"env": env, "process_tree": _ancestry(mapping)}, separators=(",", ":"))
    print(report)
    print(hashlib.sha256(report.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()