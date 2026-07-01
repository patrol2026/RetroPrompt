"""Zero-shot baseline: ONE generation per problem (same base prompt as our approach's iter-0),
no backward reasoning, no refinement, no grounding. The vanilla LLM baseline our method improves on.
Usage: python zeroshot.py <benchmark> <model> <provider> <n> <out> [workers]
"""
import sys, os, json, threading
from concurrent.futures import ThreadPoolExecutor
import retroprompt as R

bench, model, provider, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
workers = int(sys.argv[6]) if len(sys.argv) > 6 else (6 if provider == "openrouter" else 2)
R.BENCHMARK = bench
R.MODEL = model
R.PROVIDER = provider
R.MAX_TOKENS = 4096                      # match the backward-reasoning runs
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "codes.json")
codes = json.load(open(path)) if os.path.exists(path) else {}
probs = R.load_problems()[:n]
todo = [(pid, rec) for pid, rec in probs if pid not in codes]
print(f"zero-shot | {bench} | {model} ({provider}) | {len(probs)} problems ({len(todo)} to do, workers={workers})", flush=True)
lock = threading.Lock()


def work(item):
    pid, rec = item
    try:
        code = R.generate_code(R.base_prompt(rec), 0.0)        # single shot, temp=0
    except Exception as e:
        print(f"  !! {pid}: {str(e)[:100]}", flush=True); return
    with lock:
        codes[pid] = code
        json.dump(codes, open(path, "w"))
        if len(codes) % 20 == 0:
            print(f"  {len(codes)}/{len(probs)}", flush=True)


with ThreadPoolExecutor(workers) as ex:
    list(ex.map(work, todo))
print("DONE", bench, model, flush=True)
