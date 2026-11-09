"""Fake OpenAI-compatible proxy. Returns canned scripts per task so we can test
the PLUMBING (budget, retry, saving, parallelism). It says nothing about real
model quality -- that needs the real proxy."""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPTS = {
 "sales.csv": "import pandas as pd\nd=pd.read_csv('sales.csv')\nr=d[(d['product']=='widget')&(d['month'].between(7,8))]['units'].sum()\nopen('answer.txt','w').write(str(int(r)))",
 "starts with 1669": "open('answer.txt','w').write('1735')",
 "warehouse": "b=1752\nb=b//2\nb*=2\nb=b//3\nb*=2\nb=b//3\nopen('answer.txt','w').write(str(b))",
 "kb/": ("import glob,re\nc={}\nfor f in glob.glob('kb/*.md'):\n  t=open(f).read(); c[t.splitlines()[0][2:].strip()]=t\n"
         "p=c['Alice Smith']; emp=re.search(r'Employer: (.*)',p).group(1).strip()\n"
         "hq=re.search(r'Headquarters: (.*)',c[emp]).group(1).strip()\nopen('answer.txt','w').write(hq)"),
 "mylib.py": ("s=open('mylib.py').read()\n"
              "s=s.replace(\"' '.join(sorted(s.split()))\",\"' '.join(reversed(s.split()))\")\n"
              "s=s.replace('s -= x','s += x')\n"
              "s=s.replace('if x < best:','if x > best:')\n"
              "s=s.replace('x % 2 == 1','x % 2 == 0')\n"
              "open('mylib.py','w').write(s)\nopen('answer.txt','w').write('DONE')"),
}
BREAK_FIRST = os.environ.get("MOCK_BREAK_FIRST") == "1"
NOOP_CODING = os.environ.get("MOCK_CODING_NOOP") == "1"
TAMPER_CODING = os.environ.get("MOCK_CODING_TAMPER") == "1"
USAGE_MAX = os.environ.get("MOCK_USAGE_MAX") == "1"
BREAK_CODING_FIRST = os.environ.get("MOCK_CODING_BREAK_FIRST") == "1"
MOCK_DSL = os.environ.get("MOCK_DSL") == "1"
MOCK_DSL_BAD = os.environ.get("MOCK_DSL_BAD") == "1"
MOCK_PATCH = os.environ.get("MOCK_PATCH") == "1"
MOCK_PATCH_BAD = os.environ.get("MOCK_PATCH_BAD") == "1"
MOCK_FAIL_MODEL = os.environ.get("MOCK_FAIL_MODEL", "")
PATCH_REPLACEMENTS = [
    ("' '.join(sorted(s.split()))", "' '.join(reversed(s.split()))"),
    ("s -= x", "s += x"),
    ("if x < best:", "if x > best:"),
    ("x % 2 == 1", "x % 2 == 0"),
]
DSL_PLANS = {
    "starts with 1752": {"start": 1752, "ops": [
        {"op": "div", "k": 2, "round": "down"}, {"op": "mul", "k": 2},
        {"op": "div", "k": 3, "round": "down"}, {"op": "mul", "k": 2},
        {"op": "div", "k": 3, "round": "down"}]},
    "starts with 1669": {"start": 1669, "ops": [
        {"op": "sub", "amount": 36, "times": 3},
        {"op": "add", "lots": 2, "amount": 87}]},
    "sales.csv": {"filters": [{"col": "product", "op": "eq", "value": "widget"},
                              {"col": "month", "op": "between", "lo": 7, "hi": 8}],
                  "agg": "sum", "col": "units"},
    "kb/": {"start": "Alice Smith", "hops": ["employer", "headquarters"]},
}
TAMPER_SCRIPT = (
    "open('test_mylib.py','w').write(\"import unittest\\n"
    "class T(unittest.TestCase):\\n"
    "    def test_ok(self):\\n"
    "        self.assertTrue(True)\\n\")\n"
    "open('answer.txt','w').write('DONE')"
)
NOOP_SCRIPT = "open('answer.txt','w').write('DONE')"
LOCK = threading.Lock(); STATE = {"calls": 0}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req["messages"]
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        with LOCK: STATE["calls"] += 1
        system = next((m["content"] for m in msgs if m.get("role") == "system"), "").lower()
        fail_this = bool(MOCK_FAIL_MODEL) and req.get("model") == MOCK_FAIL_MODEL
        if MOCK_DSL and "compile" in system:
            if MOCK_DSL_BAD:
                text = "Sorry, I cannot produce a plan."
            else:
                p = next((v for k, v in DSL_PLANS.items() if k in user),
                         {"start": 0, "ops": []})
                text = "```json\n" + json.dumps(p) + "\n```"
            pt = sum(len(m["content"]) for m in msgs) // 3
            ct = req.get("max_tokens", 0) if USAGE_MAX else len(text) // 3
            body = json.dumps({"choices": [{"message": {"content": text}}],
                               "usage": {"prompt_tokens": pt, "completion_tokens": ct}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        if MOCK_PATCH and "exact text replacements" in system:
            if MOCK_PATCH_BAD or fail_this:
                plan = {"replacements": [{"file": "mylib.py", "old": "THIS TEXT DOES NOT EXIST",
                                          "new": "irrelevant"}]}
            else:
                plan = {"replacements": [{"file": "mylib.py", "old": old, "new": new}
                                         for old, new in PATCH_REPLACEMENTS if old in user]}
            text = "```json\n" + json.dumps(plan) + "\n```"
            pt = sum(len(m["content"]) for m in msgs) // 3
            ct = req.get("max_tokens", 0) if USAGE_MAX else len(text) // 3
            body = json.dumps({"choices": [{"message": {"content": text}}],
                               "usage": {"prompt_tokens": pt, "completion_tokens": ct}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        if "mylib.py" in user and fail_this:
            code = "raise SystemExit(3)"
        elif "mylib.py" in user and TAMPER_CODING:
            code = TAMPER_SCRIPT
        elif "mylib.py" in user and NOOP_CODING:
            code = NOOP_SCRIPT
        elif "mylib.py" in user and BREAK_CODING_FIRST and len(msgs) == 2:
            code = "raise SystemExit(3)"
        else:
            code = next((v for k, v in SCRIPTS.items() if k in user), "print('?')")
        if BREAK_FIRST and len(msgs) == 2 and "warehouse" in user:
            code = "raise SystemExit(3)"
        text = f"```python\n{code}\n```"
        pt = sum(len(m["content"]) for m in msgs) // 3
        ct = req.get("max_tokens", 0) if USAGE_MAX else len(text) // 3
        body = json.dumps({"choices": [{"message": {"content": text}}],
                           "usage": {"prompt_tokens": pt, "completion_tokens": ct}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), H).serve_forever()
