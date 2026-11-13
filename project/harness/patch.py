"""Coding tasks: the model returns exact text replacements, local code applies them.

Why: a full script that rewrites `mylib.py` costs 400-900 output tokens and can
fail on script syntax/boilerplate unrelated to the bug. A replacement plan is
much smaller and deterministic: we verify that every `old` occurs exactly once,
apply it to copies in the sandbox, then run the original tests. A failed or
unparsable plan falls back to the full-script mode (second call), so the total
stays at most 2 calls.

Safety: only existing `.py` files inside the task dir; test files are rejected at
parse time *and* at apply time; nothing is written until every replacement has
been validated.
"""
import json
import os
import re

MAX_PATCH_CHARS = 8000
MAX_REPLACEMENTS = 40
MAX_CONTEXT_CHARS = 9000
MAX_TEST_CHARS = 3500
JSON_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)
TESTISH = re.compile(r"^(test_.*\.py|.*_test\.py|conftest\.py|pytest\.ini|tox\.ini|setup\.cfg|run_tests\.py)$",
                     re.I)

SYSTEM = (
    "You repair Python tasks by returning exact text replacements. Reply with ONE ```json block, no prose.\n"
    'Schema: {"replacements":[{"file":"mylib.py","old":"<exact text currently in the file>",'
    '"new":"<replacement text>"}]}\n'
    "- Every \"old\" must appear EXACTLY ONCE in its file; copy it verbatim, including indentation.\n"
    "- Do not modify test files; they are read-only.\n"
    "- Adding NEW code (functions, imports, helpers) to existing module files is allowed when the "
    "tests need it — only editing the tests is forbidden.\n"
    "- Use several replacements when there are several bugs.\n"
    "- If exact replacements cannot express the fix, reply with a full ```python script instead "
    "(it runs in the task directory and must write answer.txt)."
)


class PatchError(Exception):
    pass


def _is_testish(rel: str) -> bool:
    base = os.path.basename(rel)
    return bool(TESTISH.match(base)) or "tests" in rel.split(os.sep)


def _rel_files(task_dir, suffix=".py"):
    out = []
    for root, dirs, files in os.walk(task_dir):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git"))
        for fn in sorted(files):
            if fn.endswith(suffix):
                out.append(os.path.relpath(os.path.join(root, fn), task_dir))
    return out


def context(task) -> str:
    """Source of the modules (repairable) and the tests (read-only), size-capped."""
    try:
        d = task["dir"]
        files = _rel_files(d)
        mods = [f for f in files if not _is_testish(f)]
        tests = [f for f in files if _is_testish(f)]
        if not mods:
            return ""
        chunks = []
        left = MAX_CONTEXT_CHARS
        for rel in mods:
            try:
                with open(os.path.join(d, rel), encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            if len(text) > left:
                text = text[:left] + "\n... [truncated]"
            left -= len(text)
            chunks.append(f"--- {rel} ---\n{text}")
            if left <= 0:
                break
        left = MAX_TEST_CHARS
        for rel in tests:
            try:
                with open(os.path.join(d, rel), encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            if len(text) > left:
                text = text[:left] + "\n... [truncated]"
            left -= len(text)
            chunks.append(f"--- {rel} (read-only tests) ---\n{text}")
            if left <= 0:
                break
        return "\n\n".join(chunks)
    except Exception:
        return ""


def spec(task):
    ctx = context(task)
    if not ctx:
        return None
    return SYSTEM, ctx


def _parse(text: str):
    if len(text) > 4 * MAX_PATCH_CHARS:
        raise PatchError("reply too long for a replacement plan")
    m = JSON_RE.search(text)
    raw = (m.group(1) if m else text).strip()
    if not raw:
        raise PatchError("empty reply")
    try:
        obj = json.loads(raw)
    except ValueError:
        i, j = raw.find("{"), raw.rfind("}")
        if i < 0 or j <= i:
            raise PatchError("no JSON object in reply")
        try:
            obj = json.loads(raw[i:j + 1])
        except ValueError as e:
            raise PatchError(f"invalid JSON: {e}")
    if not isinstance(obj, dict):
        raise PatchError("plan must be a JSON object")
    extra = set(obj) - {"replacements"}
    if extra:
        raise PatchError(f"unknown keys {sorted(extra)}; allowed: ['replacements']")
    reps = obj.get("replacements")
    if not isinstance(reps, list) or not reps:
        raise PatchError("replacements must be a non-empty list")
    if len(reps) > MAX_REPLACEMENTS:
        raise PatchError(f"too many replacements ({len(reps)} > {MAX_REPLACEMENTS})")
    out, total = [], 0
    for i, r in enumerate(reps):
        if not isinstance(r, dict):
            raise PatchError(f"replacement #{i} must be an object")
        extra = set(r) - {"file", "old", "new"}
        if extra:
            raise PatchError(f"replacement #{i}: unknown keys {sorted(extra)}")
        for k in ("file", "old", "new"):
            if k not in r:
                raise PatchError(f"replacement #{i}: missing {k!r}")
        path, old, new = r["file"], r["old"], r["new"]
        if not isinstance(path, str) or not path.strip():
            raise PatchError(f"replacement #{i}: file must be a non-empty string")
        if not isinstance(old, str) or not isinstance(new, str):
            raise PatchError(f"replacement #{i}: old/new must be strings")
        if not old:
            raise PatchError(f"replacement #{i}: old must not be empty")
        norm = os.path.normpath(path.strip())
        if os.path.isabs(norm) or norm.startswith(".."):
            raise PatchError(f"replacement #{i}: path {path!r} escapes the task directory")
        if norm.startswith(".") or "__pycache__" in norm.split(os.sep):
            raise PatchError(f"replacement #{i}: hidden/generated path {path!r} is not patchable")
        if not norm.endswith(".py"):
            raise PatchError(f"replacement #{i}: only .py files can be patched")
        if _is_testish(norm):
            raise PatchError(f"replacement #{i}: test file {norm!r} is read-only")
        total += len(old) + len(new)
        out.append({"file": norm, "old": old, "new": new})
    if total > MAX_PATCH_CHARS:
        raise PatchError(f"plan is too large ({total} chars > {MAX_PATCH_CHARS})")
    return out


def parse(text: str):
    """(replacements, None) or (None, short_error). Never raises."""
    try:
        return _parse(text), None
    except PatchError as e:
        return None, str(e)[:300]
    except Exception as e:  # a parser bug must fall back to the script mode
        return None, f"{type(e).__name__}: {e}"[:300]


def apply(work_dir: str, edits):
    """Validate all edits against the current files, then write them.

    Returns {relative_path: original_bytes} for `revert`. Raises PatchError
    before writing anything if any replacement is invalid.
    """
    contents = {}
    for e in edits:
        rel = e["file"]
        path = os.path.join(work_dir, rel)
        if not os.path.isfile(path):
            raise PatchError(f"{rel} not found")
        if path not in contents:
            try:
                with open(path, encoding="utf-8") as f:
                    contents[path] = f.read()
            except (OSError, UnicodeDecodeError) as ex:
                raise PatchError(f"cannot read {rel}: {ex}")
        text = contents[path]
        n = text.count(e["old"])
        if n != 1:
            raise PatchError(f"old text occurs {n} times in {rel}; it must occur exactly once")
        contents[path] = text.replace(e["old"], e["new"], 1)

    originals, written = {}, []
    try:
        for path, text in contents.items():
            with open(path, "rb") as f:
                originals[path] = f.read()
            tmp = path + ".patch_tmp"
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            os.replace(tmp, path)
            written.append(path)
    except OSError as ex:
        _restore(written, originals)
        raise PatchError(f"cannot write patch: {ex}")
    return {os.path.relpath(p, work_dir): data for p, data in originals.items()}


def _restore(paths, originals):
    for p in paths:
        try:
            with open(p, "wb") as f:
                f.write(originals[p])
        except OSError:
            pass


def revert(work_dir: str, originals) -> None:
    """Put back the files touched by `apply` (used before the script retry)."""
    for rel, data in originals.items():
        path = os.path.join(work_dir, rel)
        try:
            with open(path, "wb") as f:
                f.write(data)
        except OSError:
            pass
