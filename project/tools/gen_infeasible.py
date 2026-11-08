"""Synthetic infeasible tasks from feasible fixtures (the final set has 15-25% of them).
make(src_dir, out_dir) -> {task_name: reason}. Used by tests and for the estimator's training set."""
import json, os, shutil

def _load(d):
    with open(os.path.join(d, "task.json")) as f:
        return json.load(f)
def _save(d, m):
    with open(os.path.join(d, "task.json"), "w") as f:
        json.dump(m, f, indent=1)

def make(src_dir, out_dir):
    shutil.rmtree(out_dir, ignore_errors=True); os.makedirs(out_dir)
    made = {}
    def variant(base, tag, reason, edit):
        name = f"{base}__{tag}"
        shutil.copytree(os.path.join(src_dir, base), os.path.join(out_dir, name))
        d = os.path.join(out_dir, name); m = _load(d); m["task_id"] = name
        edit(d, m); _save(d, m); made[name] = reason
    for base in sorted(os.listdir(src_dir)):
        m0 = _load(os.path.join(src_dir, base)); typ = m0["task_type"]
        if typ == "data_analysis":
            variant(base, "csv_deleted", "missing_file", lambda d, m: os.remove(os.path.join(d, "sales.csv")))
            variant(base, "csv_renamed_in_text", "missing_file",
                    lambda d, m: m.update(instruction=m["instruction"].replace("sales.csv", "sales_2025.csv")))
        elif typ == "search":
            variant(base, "unknown_person", "entity_missing",
                    lambda d, m: m.update(instruction=m["instruction"].replace("Alice Smith", "Zed Quark")))
            variant(base, "kb_deleted", "missing_dir", lambda d, m: shutil.rmtree(os.path.join(d, "kb")))
        elif typ in ("coding", "code"):
            variant(base, "module_deleted", "missing_file", lambda d, m: os.remove(os.path.join(d, "mylib.py")))
            variant(base, "tests_deleted", "missing_file", lambda d, m: os.remove(os.path.join(d, "test_mylib.py")))
        elif typ == "math":
            variant(base, "missing_input", "missing_file",
                    lambda d, m: m.update(instruction=m["instruction"].replace(" How many", " Use the starting stock from inventory.csv. How many")))
            variant(base, "needs_internet", "needs_internet",
                    lambda d, m: m.update(instruction=m["instruction"].replace(" How many", " Fetch the shipping rates from https://example.com/rates. How many")))
    return made
