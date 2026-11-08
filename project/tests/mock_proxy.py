"""Fake OpenAI-compatible proxy. Returns canned scripts per task so we can test
the PLUMBING (budget, retry, saving, parallelism). It says nothing about real
model quality -- that needs the real proxy."""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPTS = {
 "sales.csv": "import pandas as pd\nd=pd.read_csv('sales.csv')\nr=d[(d['product']=='widget')&(d['month'].between(7,8))]['units'].sum()\nopen('answer.txt','w').write(str(int(r)))",
 "warehouse": "b=1752\nb=b//2\nb*=2\nb=b//3\nb*=2\nb=b//3\nopen('answer.txt','w').write(str(b))",
 "kb/": ("import glob,re\nc={}\nfor f in glob.glob('kb/*.md'):\n  t=open(f).read(); c[t.splitlines()[0][2:].strip()]=t\n"
         "p=c['Alice Smith']; emp=re.search(r'Employer: (.*)',p).group(1).strip()\n"
         "hq=re.search(r'Headquarters: (.*)',c[emp]).group(1).strip()\nopen('answer.txt','w').write(hq)"),
 "mylib.py": ("s=open('mylib.py').read()\ns=s.replace(\"' '.join(sorted(s.split()))\",\"' '.join(reversed(s.split()))\").replace('s -= x','s += x')\n"
              "open('mylib.py','w').write(s)\nopen('answer.txt','w').write('DONE')"),
}
BREAK_FIRST = os.environ.get("MOCK_BREAK_FIRST") == "1"
LOCK = threading.Lock(); STATE = {"calls": 0}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req["messages"]; user = msgs[1]["content"]
        with LOCK: STATE["calls"] += 1
        code = next((v for k, v in SCRIPTS.items() if k in user), "print('?')")
        if BREAK_FIRST and len(msgs) == 2 and "warehouse" in user:
            code = "raise SystemExit(3)"
        text = f"```python\n{code}\n```"
        pt = sum(len(m["content"]) for m in msgs) // 3
        body = json.dumps({"choices": [{"message": {"content": text}}],
                           "usage": {"prompt_tokens": pt, "completion_tokens": len(text) // 3}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), H).serve_forever()
