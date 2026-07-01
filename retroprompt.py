"""
RetroPrompt: prompt refinement for LLM code generation via backward reasoning.

For each problem:
  1. forward spec : extract the INTENDED requirements from the problem statement.
  2. generate code from the (possibly refined) prompt.
  3. backward spec: back-translate the code into the requirements it ACTUALLY implements.
  4. compare forward vs backward -> {missing, extra}.
  5. if empty: stop.  else: refine the prompt (emphasize missing, forbid extra) & regenerate.
  6. loop up to --max_iters.

Final code is graded against the human-curated HumanEval-NFR gold tests (FULL, FR-only,
and per NFR category). No tests are generated or used for selection.

Self-contained: needs only an Ollama-compatible server + the local executor in this folder.

Examples
--------
  # smoke test (1 problem, verbose trace)
  python retroprompt_gen.py - main --n=1 --candidates=1 --max_iters=3 --verbose=True

  # full curve: pass@k after EACH iteration (FULL + FR-only)
  python retroprompt_gen.py - curve --n=30 --candidates=10 --max_iters=4
"""
import json
import os
import re
import doctest
from concurrent.futures import ThreadPoolExecutor
from math import comb
from string import Template

import fire
import requests

import backward_reasoning

# ----- config (override via env) ---------------------------------------------
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
EXEC = os.environ.get("CODEEXEC_URL", "http://localhost:5097/execute")
MODEL = os.environ.get("MODEL", "llama3:8b-instruct-fp16")
MODEL_GEN = None       # per-role override: code generation. None -> fall back to MODEL
MODEL_BACK = None      # per-role override: backward reasoning + NL spec/compare. None -> fall back to MODEL
PROVIDER = os.environ.get("PROVIDER", "ollama")   # "ollama" | "openrouter"
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PREAMBLE = "from typing import *\nimport sys, math, re, collections, itertools, functools\n\n"
NFR_CATS = ["general", "edge", "performance", "robustness", "maintainability"]
BACKWARD = "oneshot"   # "oneshot" (one LLM call) | "structural" (HoarePrompt-style backward walk)
MAX_TOKENS = 2048      # raise for reasoning models (e.g. gpt-oss) so output isn't truncated
BENCHMARK = "humaneval_nfr"   # "humaneval_nfr" | "code_contests" | "xcodeeval"
IO_TIMEOUT = 10        # per-test timeout (s) for stdin/stdout competitive benchmarks
MAX_TESTS = 50         # cap test cases graded per problem for competitive benchmarks (0 = all)
GROUNDING = "none"     # "none" (pure NL loop) | "public" (gate refinements on public_test_cases)

# ----- data (bundled, offline) -----------------------------------------------
try:                                     # one-shot in-context example (ships via the dataset archive)
    _EX = json.load(open(os.path.join(DATA, "humaneval_like.json")))[0]
    EX_PROMPT, EX_REQ, EX_CODE = _EX["prompt"], _EX["requirements"], _EX["code"]
except FileNotFoundError:
    EX_PROMPT = EX_REQ = EX_CODE = ""


def _load_jsonl(fn):
    return [json.loads(l) for l in open(os.path.join(DATA, fn)) if l.strip()]


def load_humaneval_nfr():
    """[(id, {prompt(stub), gold_tc(full test), general, edge, performance, robustness, maintainability})]"""
    prompts = {r["id"]: r for r in json.load(open(os.path.join(DATA, "humaneval_prompts.json")))}
    nf = json.load(open(os.path.join(DATA, "humaneval_nf.json")))
    out = []
    for r in sorted(nf, key=lambda x: x["id"]):
        rec = {"prompt": prompts[r["id"]]["prompt"], "gold_tc": r["test"]}
        for cat in NFR_CATS:
            rec[cat] = r.get(cat, "")
        # Specine-style public tests for grounding: docstring >>> examples; if none, carve a couple of
        # 'assert f(x)==y' cases from the gold suite AND delete them from the hidden gold (unbiased eval).
        pub = _doctest_public_asserts(rec["prompt"]) or _carve_public_from_gold(rec, k=2)
        rec["public"] = pub or None
        out.append((r["id"], rec))
    return out


def load_code_contests():
    """[(id, {prompt(description), io={inputs,outputs}})] — stdin/stdout competitive problems."""
    out = []
    for i, r in enumerate(_load_jsonl("code_contests.jsonl")):
        pid = r.get("name") or f"cc-{i}"
        out.append((pid, {"prompt": r["description"], "io": r["all_test_cases"],
                          "public": r.get("public_test_cases")}))
    return out


def load_xcodeeval():
    """[(id, {prompt(description+specs), io={inputs,outputs}})] — stdin/stdout competitive problems."""
    out = []
    for i, r in enumerate(_load_jsonl("xCodeEval.jsonl")):
        desc = r["description"]
        if r.get("input_spec"):
            desc += "\n\nInput\n" + r["input_spec"]
        if r.get("output_spec"):
            desc += "\n\nOutput\n" + r["output_spec"]
        if r.get("notes"):
            desc += "\n\n" + r["notes"]
        pid = r.get("src_uid") or f"xce-{i}"
        out.append((pid, {"prompt": desc, "io": r["all_test_cases"],
                          "public": r.get("public_test_cases")}))
    return out


def load_apps():
    """[(id, {prompt(question), io={inputs,outputs}})] — APPS, mostly stdin/stdout competitive problems."""
    out = []
    for i, r in enumerate(_load_jsonl("apps.jsonl")):
        pid = str(r.get("problem_id", f"apps-{i}"))
        prompt = r["question"]
        sc = (r.get("starter_code") or "").strip()
        if sc:                                  # the few call-based problems: include the starter signature
            prompt += "\n\n" + sc
        out.append((pid, {"prompt": prompt, "io": r["all_test_cases"],
                          "public": r.get("public_test_cases")}))
    return out


def load_problems():
    if BENCHMARK == "code_contests":
        return load_code_contests()
    if BENCHMARK == "xcodeeval":
        return load_xcodeeval()
    if BENCHMARK == "apps":
        return load_apps()
    return load_humaneval_nfr()


def base_prompt(rec):
    """The loop's base prompt: a code stub (HumanEval) or a problem description (competitive)."""
    if "io" in rec:
        return rec["prompt"]
    return f"```python\n{rec['prompt']}\n```"


# ----- prompts ----------------------------------------------------------------
REQ_INSTR = ("List the software requirements (functional AND non-functional: input/output, "
             "edge cases, robustness to invalid input, performance, maintainability) this "
             "function must satisfy. One concise bullet each.")
BACK_INSTR = ("Read this code line by line, then list as bullets the requirements/behaviors it "
              "ACTUALLY implements (functional and non-functional). Describe ONLY what the code "
              "does, not what it ought to do.")
_CMP_EXAMPLE = ('Example:\n{"missing": ["Return False for an empty list", '
                '"Return None for non-numeric input instead of crashing"], '
                '"extra": ["Sorts the input, which can reorder the result"]}')


import threading as _threading
import time as _time
_tls = _threading.local()


def tok_reset():
    _tls.tin = 0; _tls.tout = 0; _tls.calls = 0


def tok_get():
    return getattr(_tls, "tin", 0), getattr(_tls, "tout", 0), getattr(_tls, "calls", 0)


def _tok_add(i, o):
    _tls.tin = getattr(_tls, "tin", 0) + int(i or 0)
    _tls.tout = getattr(_tls, "tout", 0) + int(o or 0)
    _tls.calls = getattr(_tls, "calls", 0) + 1


# ----- per-COMPONENT accounting: forward-requirements / code-gen / backward-reasoning / compare ---
# process-global, thread-safe; tracks input tokens, output tokens, #calls and wall-time per step.
# (model selection still uses gen vs back: 'gen' -> MODEL_GEN, everything else -> MODEL_BACK.)
_role_lock = _threading.Lock()
_ROLES = ("forward", "gen", "backward", "compare")
ROLE_ACC = {r: {"in": 0, "out": 0, "calls": 0, "time": 0.0} for r in _ROLES}


def role_reset():
    with _role_lock:
        for b in ROLE_ACC.values():
            b["in"] = b["out"] = b["calls"] = 0; b["time"] = 0.0


def role_get():
    with _role_lock:
        return {k: dict(v) for k, v in ROLE_ACC.items()}


def _role_add(role, tin, tout, dt):
    b = ROLE_ACC.get(role, ROLE_ACC["gen"])
    with _role_lock:
        b["in"] += int(tin or 0); b["out"] += int(tout or 0); b["calls"] += 1; b["time"] += dt


def chat(messages, temperature=0.0, top_p=1.0, max_tokens=None, role="gen"):
    mt = max_tokens or MAX_TOKENS
    # per-role model: 'gen' = code generation, 'back' = backward reasoning + NL spec/compare.
    model = (MODEL_GEN if role == "gen" else MODEL_BACK) or MODEL
    last = None
    for _attempt in range(8):
        if _attempt:                                        # exponential backoff (esp. for 429 rate-limits)
            _time.sleep(min(60, 2 ** _attempt))
        t0 = _time.time()
        try:
            if PROVIDER == "openrouter":
                key = os.environ.get("OPENROUTER_API_KEY")
                if not key:
                    raise RuntimeError("set OPENROUTER_API_KEY for --provider=openrouter")
                r = requests.post(OPENROUTER_URL, headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://anonymous.4open.science/",
                    "X-Title": "retroprompt-nfr",
                }, json={
                    "model": model, "messages": messages,
                    "temperature": temperature, "top_p": top_p, "max_tokens": mt,
                    # pin to one provider for precision consistency (OPENROUTER_PROVIDER, e.g. "Together")
                    **({"provider": {"order": [os.environ["OPENROUTER_PROVIDER"]], "allow_fallbacks": False}}
                       if os.environ.get("OPENROUTER_PROVIDER") else {}),
                }, timeout=900)
                data = r.json()
                if "choices" not in data:
                    raise RuntimeError(f"openrouter error: {str(data)[:300]}")
                u = data.get("usage", {}) or {}
                _tok_add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                _role_add(role, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), _time.time() - t0)
                return data["choices"][0]["message"]["content"] or ""
            else:  # ollama (default)
                r = requests.post(OLLAMA, json={
                    "model": model, "messages": messages, "stream": False,
                    "options": {"temperature": temperature, "top_p": top_p, "num_predict": mt},
                }, timeout=900)
                data = r.json()
                _tok_add(data.get("prompt_eval_count", 0), data.get("eval_count", 0))
                _role_add(role, data.get("prompt_eval_count", 0), data.get("eval_count", 0), _time.time() - t0)
                return data["message"]["content"]
        except Exception as e:
            last = e
    raise last


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    code = blocks[-1] if blocks else text                       # fall back to raw text
    code = re.sub(r"^```(?:python)?\s*\n", "", code.strip())    # strip unclosed leading fence
    code = re.sub(r"\n?```\s*$", "", code)                      # strip dangling trailing fence
    return code.replace("```python", "").replace("```", "").strip()


def _stringify(x):
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        return (x.get("description") or "; ".join(f"{k}: {v}" for k, v in x.items())).strip()
    return str(x).strip()


def parse_diff(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"missing": [_stringify(x) for x in d.get("missing", []) if _stringify(x)],
                    "extra": [_stringify(x) for x in d.get("extra", []) if _stringify(x)]}
        except Exception:
            pass
    return {"missing": [], "extra": []}


# ----- competitive (xCodeEval / CodeContests) forward-requirements example: FR ONLY -----------
REQ_INSTR_FR = ("List the requirements this program must satisfy: the exact input format and how to "
                "parse it, the exact output format, the core behavior/algorithm it must compute, and "
                "the edge cases it must handle. One concise bullet each.")

EX_PROMPT_CC = (
    "Theofanis has a string s of length n consisting of the characters '+' and '-'. In one move he "
    "can flip a single character (turn '+' into '-' or '-' into '+'). Find the minimum number of "
    "moves so that no two adjacent characters of s are equal.\n\n"
    "Input\n"
    "The first line contains an integer t (1 <= t <= 1000) - the number of test cases.\n"
    "Each test case consists of two lines: the first contains an integer n (1 <= n <= 100), the "
    "second contains the string s of length n.\n\n"
    "Output\n"
    "For each test case, print one integer - the minimum number of moves.")

EX_REQ_CC = (
    "- Input parsing\n"
    "  - Read an integer t (number of test cases) from the first line of standard input.\n"
    "  - For each test case read an integer n, then read the string s of length n on the next line.\n"
    "  - Process all t test cases; do not stop early.\n"
    "- Output format\n"
    "  - For each test case print exactly one integer on its own line, in input order.\n"
    "- Core behavior\n"
    "  - The desired result is an alternating string. Compare s against both alternating patterns "
    "(starting with '+' and starting with '-') and output the smaller number of mismatched positions.\n"
    "- Edge cases\n"
    "  - n = 1: the answer is 0 (a single character is always valid).\n"
    "  - s already alternating: the answer is 0.\n"
    "  - Strip trailing whitespace/newlines when reading s so its length matches n.")

EX_CODE_CC = (
    "import sys\n"
    "input = sys.stdin.readline\n"
    "t = int(input())\n"
    "for _ in range(t):\n"
    "    n = int(input())\n"
    "    s = input().strip()\n"
    "    c1 = sum(1 for i, ch in enumerate(s) if ch != '+-'[i % 2])\n"
    "    c2 = sum(1 for i, ch in enumerate(s) if ch != '-+'[i % 2])\n"
    "    print(min(c1, c2))")


def forward_requirements(prompt):
    if BENCHMARK in ("code_contests", "xcodeeval", "apps"):   # competitive: FR-only, stdin/stdout example
        return chat([
            {"role": "user", "content": f"{EX_PROMPT_CC}\n\n{REQ_INSTR_FR}"},
            {"role": "assistant", "content": EX_REQ_CC},
            {"role": "user", "content": f"{prompt}\n\n{REQ_INSTR_FR}"},
        ], role="forward")
    return chat([                                      # humaneval_nfr: FR + NFR, function example
        {"role": "user", "content": f"```python\n{EX_PROMPT}\n```\n\n{REQ_INSTR}"},
        {"role": "assistant", "content": EX_REQ},
        {"role": "user", "content": f"```python\n{prompt}\n```\n\n{REQ_INSTR}"},
    ], role="forward")


def generate_code(refined_prompt, temperature):
    if BENCHMARK in ("code_contests", "xcodeeval", "apps"):   # competitive: full stdin/stdout program
        txt = chat([{"role": "user", "content":
            f"{refined_prompt}\n\nWrite a COMPLETE Python 3 program that reads the input from "
            "standard input and writes the answer to standard output, matching the format above. "
            "Output ONLY the program in ```python ... ```."}],
            temperature=temperature, top_p=0.95)
        return extract_code(txt)
    txt = chat([   # humaneval_nfr: 1-shot function completion
        {"role": "user", "content": f"```python\n{EX_PROMPT}\n```\n\nWrite a python solution. "
                                    "DO NOT include tests. Package code in ```python ... ```."},
        {"role": "assistant", "content": f"```python\n{EX_CODE}\n```"},
        {"role": "user", "content": f"{refined_prompt}\n\nWrite a python solution. DO NOT include "
                                    "tests or explanations. Package code in ```python ... ```."},
    ], temperature=temperature, top_p=0.95)
    return extract_code(txt)


def backward_requirements(code):
    return chat([
        {"role": "user", "content": f"```python\n{EX_CODE}\n```\n\n{BACK_INSTR}"},
        {"role": "assistant", "content": EX_REQ},
        {"role": "user", "content": f"```python\n{code}\n```\n\n{BACK_INSTR}"},
    ], role="backward")


def backward_spec(code, trace=None):
    """Dispatch the backward-reasoning method: one-shot (default) or structural (HoarePrompt-style).
    If `trace` is a list and mode is structural, it is filled with per-statement annotations."""
    if BACKWARD == "structural":
        return backward_reasoning.structural_backward(
            code, lambda p: chat([{"role": "user", "content": p}], role="backward"), trace=trace)
    return backward_requirements(code)


def compare(intended, implemented):
    out = chat([{"role": "user", "content":
        "Compare the INTENDED requirements of a problem with what a candidate solution ACTUALLY "
        f"implements.\n\nINTENDED:\n{intended}\n\nIMPLEMENTED (from the code):\n{implemented}\n\n"
        "Focus ONLY on observable behavior: input/output conditions, edge cases, and robustness "
        "to invalid input. IGNORE differences in stated time/space complexity (Big-O) and any "
        "code-style or maintainability wording.\n"
        'Return ONLY JSON with two lists of SHORT plain-English strings: '
        '{"missing": [intended but NOT implemented], "extra": [implemented but NOT intended]}.\n'
        + _CMP_EXAMPLE}], role="compare")
    return parse_diff(out)


# ----- grading ----------------------------------------------------------------
def run_tests(code, tests, timeout=5):
    code = PREAMBLE + code
    full = code + "\n\n" + Template(tests).safe_substitute(prediction=code)
    try:
        out = requests.post(EXEC, json={"code": full, "timeout": timeout}, timeout=120).json()
        return "Exit Code: 0" in out.get("output", "")
    except Exception:
        return False


def fr_only(gold):
    i = gold.find("# Test Cases Regarding Non-functional Requirements")
    return gold[:i] if i != -1 else gold


def _norm(s):
    """Normalize stdout / expected output for comparison (line-wise rstrip, strip trailing blanks)."""
    return "\n".join(ln.rstrip() for ln in str(s).replace("\r\n", "\n").strip().split("\n"))


def run_io_tests(code, io):
    """Competitive grading: run `code` with each test's stdin, compare stdout to expected. All must match."""
    code = PREAMBLE + code
    pairs = list(zip(io.get("inputs", []), io.get("outputs", [])))
    if MAX_TESTS:
        pairs = pairs[:MAX_TESTS]
    if not pairs:
        return False
    for inp, exp in pairs:
        if isinstance(exp, list):              # xCodeEval sometimes nests an output
            exp = exp[0] if exp else ""
        try:
            out = requests.post(EXEC, json={"code": code, "stdin": str(inp), "timeout": IO_TIMEOUT},
                                timeout=300).json().get("output", "")
        except Exception:
            return False
        if "Exit Code: 0" not in out:          # crash or timeout
            return False
        got = out.rsplit("Exit Code:", 1)[0]   # stdout before the executor trailer
        if _norm(got) != _norm(exp):
            return False
    return True


def io_pass_count(code, io):
    """Number of I/O test cases passed (granular signal for grounding/gating). 0 if io is falsy."""
    if not io:
        return 0
    code = PREAMBLE + code
    pairs = list(zip(io.get("inputs", []), io.get("outputs", [])))
    if MAX_TESTS:
        pairs = pairs[:MAX_TESTS]
    cnt = 0
    for inp, exp in pairs:
        if isinstance(exp, list):
            exp = exp[0] if exp else ""
        try:
            out = requests.post(EXEC, json={"code": code, "stdin": str(inp), "timeout": IO_TIMEOUT},
                                timeout=300).json().get("output", "")
        except Exception:
            continue
        if "Exit Code: 0" in out and _norm(out.rsplit("Exit Code:", 1)[0]) == _norm(exp):
            cnt += 1
    return cnt


def _doctest_public_asserts(prompt):
    """HumanEval public tests = docstring >>> examples rendered as 'assert (call) == (want)' lines."""
    out = []
    for ex in doctest.DocTestParser().parse(prompt):
        if not isinstance(ex, doctest.Example):
            continue
        src = ex.source.strip()
        want = ex.want
        if '"""' in want:                       # strip the docstring-closing artifact on the last example
            want = want.split('"""')[0]
        want = want.strip()
        if not (src and want):
            continue
        a = f"assert ({src}) == ({want})"
        try:                                     # keep only asserts that parse (drop multi-line/non-literal wants)
            compile(a, "<t>", "exec")
            out.append(a)
        except Exception:
            pass
    return out


def _carve_public_from_gold(rec, k=2):
    """No docstring examples: move up to k 'assert f(x)==y' lines from general(->edge) into the public set,
    and DELETE them from the hidden gold (gold_tc + that category) so final grading stays unbiased."""
    carved = []
    for cat in ("general", "edge"):
        kept = []
        for ln in rec[cat].split("\n"):
            s = ln.strip()
            if len(carved) < k and re.match(r"assert\s+\w+\(.*\)\s*==", s) and not s.startswith("assert not"):
                carved.append(s)
                rec["gold_tc"] = rec["gold_tc"].replace(s, "")     # remove from hidden FULL grading
            else:
                kept.append(ln)
        rec[cat] = "\n".join(kept)                                 # remove from hidden per-axis grading
        if len(carved) >= k:
            break
    return carved


def assert_pass_count(code, asserts):
    """Public-test grounding signal for HumanEval: how many of `asserts` the code passes (0 if none)."""
    if not asserts:
        return 0
    body = "\n".join(f"try:\n    {a}\n    _p += 1\nexcept Exception:\n    pass" for a in asserts)
    full = PREAMBLE + code + "\n\n_p = 0\n" + body + "\nprint('PUBPASS', _p)"
    try:
        out = requests.post(EXEC, json={"code": full, "timeout": 10}, timeout=120).json().get("output", "")
    except Exception:
        return 0
    m = re.search(r"PUBPASS (\d+)", out)
    return int(m.group(1)) if m else 0


def grade_full(code, rec):
    """Full grade: HumanEval-NFR gold suite (FR+NFR) or competitive I/O tests."""
    return run_io_tests(code, rec["io"]) if "io" in rec else run_tests(code, rec["gold_tc"])


def grade_fr(code, rec):
    """Functional grade: HumanEval-NFR FR-only, or (for competitive) the same I/O tests."""
    return run_io_tests(code, rec["io"]) if "io" in rec else run_tests(code, fr_only(rec["gold_tc"]))


def passatk(per_problem, k):
    o = []
    for p in per_problem:
        n, c = len(p), sum(bool(x) for x in p)
        o.append(1 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0)
    return sum(o) / len(o) * 100 if o else 0.0


# ----- loop -------------------------------------------------------------------
def step_candidate(st, intended, max_temp, regen_temp, it):
    if st["converged"]:
        return st["last_code"]
    cur = st["base"]
    if st["must"]:
        cur += "\n# Make sure the code ALSO satisfies:\n" + "\n".join(f"# - {m}" for m in st["must"])
    if st["avoid"]:
        cur += "\n# Do NOT do the following:\n" + "\n".join(f"# - {a}" for a in st["avoid"])
    code = generate_code(cur, max_temp if it == 0 else regen_temp)
    # grounding gate: reject a refinement that regresses on the public tests (keep the prior code).
    # Public tests are used ONLY here (never for final grading). No-op if grounding off / no public tests.
    if GROUNDING == "public" and st.get("public"):
        pub = st["public"]
        new_score = assert_pass_count(code, pub) if isinstance(pub, list) else io_pass_count(code, pub)
        if new_score >= st.get("public_score", -1):
            st["public_score"] = new_score
        elif st["last_code"]:
            code = st["last_code"]              # revert to the better prior code
    annotations = [] if BACKWARD == "structural" else None
    bspec = backward_spec(code, trace=annotations)
    diff = compare(intended, bspec)
    st["last_code"] = code
    st["last_backward"] = bspec               # final "what the code does"
    st["last_annotations"] = annotations      # per-statement backward annotations (structural)
    st["last_diff"] = diff                     # {missing, extra} comparison
    if len(diff["missing"]) + len(diff["extra"]) == 0:
        st["converged"] = True
    else:
        for m in diff["missing"]:
            if m not in st["must"]:
                st["must"].append(m)
        for e in diff["extra"]:
            if e not in st["avoid"]:
                st["avoid"].append(e)
    st["last_must"] = list(st["must"])         # refinement constraints after this iteration
    st["last_avoid"] = list(st["avoid"])
    return code


def main(n=2, candidates=2, max_iters=3, out="results/retroprompt", verbose=False, model=None,
         backward="oneshot", provider=None, max_tokens=2048,
         benchmark="humaneval_nfr", io_timeout=10, max_tests=50, grounding="none",
         model_gen=None, model_back=None):
    """Run the loop, grade only the FINAL code per candidate, report pass@k."""
    global MODEL, BACKWARD, PROVIDER, MAX_TOKENS, BENCHMARK, IO_TIMEOUT, MAX_TESTS, GROUNDING
    global MODEL_GEN, MODEL_BACK
    if model:
        MODEL = model
    if provider:
        PROVIDER = provider
    MODEL_GEN = model_gen
    MODEL_BACK = model_back
    BACKWARD = backward
    MAX_TOKENS = max_tokens
    BENCHMARK = benchmark
    IO_TIMEOUT = io_timeout
    MAX_TESTS = max_tests
    GROUNDING = grounding
    print(f"benchmark: {BENCHMARK} | provider: {PROVIDER} | model(gen): {MODEL_GEN or MODEL} | "
          f"model(back): {MODEL_BACK or MODEL} | backward: {BACKWARD} | grounding: {GROUNDING}", flush=True)
    targets = load_problems()[:n]
    os.makedirs(out, exist_ok=True)
    merged = []
    for pid, rec in targets:
        intended = forward_requirements(rec["prompt"])
        base = base_prompt(rec)
        codes, passed = [], []
        for c in range(candidates):
            st = {"base": base, "must": [], "avoid": [], "converged": False, "last_code": "",
                  "public": rec.get("public"), "public_score": -1}
            temp = 0.0 if candidates == 1 else 0.8
            for it in range(max_iters):
                code = step_candidate(st, intended, temp, 0.4, it)
                if verbose:
                    print(f"  {pid} cand{c} iter{it}: converged={st['converged']}")
                if st["converged"]:
                    break
            codes.append(st["last_code"])
            passed.append(grade_full(st["last_code"], rec))
        merged.append({"id": pid, "code": codes, "passed": passed})
        print(f"{pid}: passed = {passed}", flush=True)
    json.dump(merged, open(f"{out}/results_merged.json", "w"), indent=2)
    for k in (1, 3):
        if k <= candidates:
            print(f"pass@{k} = {passatk([m['passed'] for m in merged], k):.2f}%")


def curve(n=30, candidates=10, max_iters=4, out="results/retroprompt", workers=8,
          per_category=False, model=None, backward="oneshot", start=0, resume=None, max_tokens=2048,
          provider=None, benchmark="humaneval_nfr", io_timeout=10, max_tests=50,
          early_stop=True, patience=1, grounding="none", model_gen=None, model_back=None):
    """Report pass@1/pass@3 (FULL + FR-only [+ per-category]) AFTER EACH iteration.

    Runs problems [start : start+n] (sorted by id), so the benchmark can be split across runs,
    e.g. --start=0 --n=82  then  --start=82 --n=82.

    --resume=<prior out dir> reloads each candidate's saved state (must/avoid/converged/last_code)
    from a finished run and continues for `max_iters` MORE iterations (use the same start/n/
    candidates/backward). Reads from `resume`, writes the combined history to `out`.
    """
    global MODEL, BACKWARD, MAX_TOKENS, PROVIDER, BENCHMARK, IO_TIMEOUT, MAX_TESTS, GROUNDING
    global MODEL_GEN, MODEL_BACK
    if model:
        MODEL = model
    if provider:
        PROVIDER = provider
    MODEL_GEN = model_gen      # None -> falls back to MODEL
    MODEL_BACK = model_back
    BACKWARD = backward
    MAX_TOKENS = max_tokens
    BENCHMARK = benchmark
    IO_TIMEOUT = io_timeout
    MAX_TESTS = max_tests
    GROUNDING = grounding
    print(f"benchmark: {BENCHMARK} | provider: {PROVIDER} | model(gen): {MODEL_GEN or MODEL} | "
          f"model(back): {MODEL_BACK or MODEL} | backward: {BACKWARD} | grounding: {GROUNDING} | "
          f"max_tokens: {MAX_TOKENS} | problems [{start}:{start+n}]", flush=True)
    targets = load_problems()[start:start + n]
    os.makedirs(out, exist_ok=True)
    role_reset()                       # track tokens+time for code-gen ('gen') vs backward reasoning ('back')
    pool = ThreadPoolExecutor(workers)

    max_temp = 0.0 if candidates == 1 else 0.8
    keys = [(pi, c) for pi in range(len(targets)) for c in range(candidates)]
    if resume:
        print(f"resuming from {resume} ...", flush=True)
        fs = json.load(open(f"{resume}/forward_specs.json"))
        intended = [fs[targets[pi][0]] for pi in range(len(targets))]
        rows = json.load(open(f"{resume}/curve.json"))["rows"]
        code_log = json.load(open(f"{resume}/codes.json"))
        trace_log = json.load(open(f"{resume}/trace.json"))
        base_iter = len(rows)
        states = {}
        for (pi, c) in keys:
            last = trace_log[f"{pi}:{c}"][-1]
            states[(pi, c)] = {"base": base_prompt(targets[pi][1]),
                               "must": list(last["must"]), "avoid": list(last["avoid"]),
                               "converged": last["converged"], "last_code": last["code"],
                               "last_backward": last["backward_spec"], "last_annotations": last["annotations"],
                               "last_diff": {"missing": last["missing"], "extra": last["extra"]},
                               "last_must": list(last["must"]), "last_avoid": list(last["avoid"]),
                               "public": targets[pi][1].get("public"), "public_score": -1}
    else:
        print(f"forward requirements for {len(targets)} problems ...", flush=True)
        intended = list(pool.map(lambda t: forward_requirements(t[1]["prompt"]), targets))
        states = {(pi, c): {"base": base_prompt(targets[pi][1]),
                            "must": [], "avoid": [], "converged": False, "last_code": "",
                            "public": targets[pi][1].get("public"), "public_score": -1}
                  for (pi, c) in keys}
        code_log = {f"{pi}:{c}": [] for (pi, c) in keys}
        trace_log = {f"{pi}:{c}": [] for (pi, c) in keys}
        rows = []
        base_iter = 0
    json.dump({targets[pi][0]: intended[pi] for pi in range(len(targets))},
              open(f"{out}/forward_specs.json", "w"), indent=2)

    def _score(r):  # early-stop score: pass@1 + pass@3 (nan-safe), improvement in either continues
        a, b = r["full_pass@1"], r["full_pass@3"]
        return (a if a == a else 0.0) + (b if b == b else 0.0)
    best_metric = max([_score(r) for r in rows], default=-1.0)
    no_improve = 0

    hdr = "iter | FULL p@1 | FULL p@3 | FRonly p@1 | FRonly p@3 | converged"
    print("\n" + hdr + "\n" + "-" * len(hdr), flush=True)
    for it in range(base_iter, base_iter + max_iters):
        def do(k):
            try:
                return (k, step_candidate(states[k], intended[k[0]], max_temp, 0.4, it))
            except Exception:
                return (k, states[k]["last_code"])
        codes = dict(pool.map(do, keys))
        for k in keys:
            code_log[f"{k[0]}:{k[1]}"].append(codes[k])
            s = states[k]
            trace_log[f"{k[0]}:{k[1]}"].append({
                "iter": it,
                "id": targets[k[0]][0],
                "code": codes[k],
                "backward_spec": s.get("last_backward"),       # final "what the code does"
                "annotations": s.get("last_annotations"),      # per-statement backward trace (structural)
                "missing": s.get("last_diff", {}).get("missing", []),
                "extra": s.get("last_diff", {}).get("extra", []),
                "must": s.get("last_must", []),
                "avoid": s.get("last_avoid", []),
                "converged": s.get("converged", False),
            })

        def grade(k):
            g = targets[k[0]][1]
            try:
                return (k, grade_full(codes[k], g), grade_fr(codes[k], g))
            except Exception:
                return (k, False, False)
        graded = list(pool.map(grade, keys))
        full = {k: f for (k, f, _) in graded}
        fr = {k: r for (k, _, r) in graded}

        def pk(d, k):
            per = [[d[(pi, c)] for c in range(candidates)] for pi in range(len(targets))]
            return passatk(per, k) if candidates >= k else float("nan")
        row = {"iter": it, "full_pass@1": round(pk(full, 1), 2), "full_pass@3": round(pk(full, 3), 2),
               "fronly_pass@1": round(pk(fr, 1), 2), "fronly_pass@3": round(pk(fr, 3), 2),
               "converged": sum(1 for s in states.values() if s["converged"])}
        rows.append(row)
        print(f"{it:>4} | {row['full_pass@1']:8.2f} | {row['full_pass@3']:8.2f} | "
              f"{row['fronly_pass@1']:10.2f} | {row['fronly_pass@3']:10.2f} | "
              f"{row['converged']}/{len(states)}", flush=True)
        json.dump({"config": {"n": n, "candidates": candidates, "max_iters": max_iters}, "rows": rows},
                  open(f"{out}/curve.json", "w"), indent=2)
        json.dump(code_log, open(f"{out}/codes.json", "w"))
        json.dump(trace_log, open(f"{out}/trace.json", "w"), indent=2)
        # per-component cost so far: forward-reqs / code-gen / backward-reasoning / compare
        ra = role_get()
        json.dump(ra, open(f"{out}/tokens.json", "w"), indent=2)
        print("   [tokens] " + " | ".join(
            f"{r}: in={ra[r]['in']} out={ra[r]['out']} t={ra[r]['time']:.0f}s" for r in _ROLES), flush=True)

        # early stop: halt the sweep once FULL pass@k stops improving (saves API cost)
        if _score(row) > best_metric + 1e-9:
            best_metric, no_improve = _score(row), 0
        else:
            no_improve += 1
            if early_stop and no_improve >= patience:
                print(f">> early stop at iter {it}: FULL pass@k did not improve for {patience} "
                      f"iter(s) (best score={best_metric:.2f}). Stopping to save cost.", flush=True)
                break

    if per_category and BENCHMARK == "humaneval_nfr":
        last = len(next(iter(code_log.values()))) - 1
        print("\nper-category pass@1 (final iter):")
        for cat in NFR_CATS:
            per = [[run_tests(code_log[f"{pi}:{c}"][last], targets[pi][1][cat] or "pass")
                    for c in range(candidates)] for pi in range(len(targets))]
            print(f"  {cat:15} pass@1={passatk(per,1):6.2f}  pass@3={passatk(per,3):6.2f}")
    pool.shutdown()
    print("\nsaved:", f"{out}/curve.json", "and", f"{out}/codes.json")


if __name__ == "__main__":
    fire.Fire({"main": main, "curve": curve})
