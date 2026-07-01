"""SCoT prompting baseline  (Li et al., "Structured Chain-of-Thought Prompting for Code Generation",
TOSEM 2025). Faithful reimplementation:
 - 3 demonstration examples of <requirement, SCoT, code>; the SCoT = an IO structure + a solving
   process written with the three programming structures (sequential / branch / loop), as comments.
 - The verbatim task instruction + the two hints ("Let's think step by step", "Write your code here").
 - Single generation of SCoT-then-code; sampling temperature=0.8, top_p=0.95 (paper Sec 3.6).
The paper is FUNCTION-level (HumanEval/MBPP/MBCPP) -> used as-is for humaneval_nfr. For the competitive
benchmarks (xcodeeval/code_contests) there is no paper-native format; the COMPETITIVE demos below are a
clearly-labelled stdin/stdout adaptation that keeps the SCoT method (IO structure + 3 structures) identical.

Usage: python scot.py <benchmark> <model> <provider> <n> <out> [workers] [candidates]
"""
import sys, os, json, re, threading
from concurrent.futures import ThreadPoolExecutor
import retroprompt as R

# ---- verbatim task instruction (paper, Fig. 3) ----
INSTRUCTION = ("Your task is to complete the following code. You should first write a rough problem-solving "
               "process using three programming structures (i.e., sequential, branch, and loop structures) and "
               "then output the final code.")

# ---- 3 function-level demonstrations <requirement, SCoT(as comments), code> (covers loop/branch/sequential) ----
FUNC_DEMOS = '''def sum_Of_Primes(n):
    """ Write a python function to find the sum of prime numbers between 1 to n. """
    # Let's think step by step
    # Input: n, an integer
    # Output: ans, an integer
    # 1. Initialize a list "prime" of length n+1 with True values.
    # 2. Initialize a variable "p" with 2.
    # 3. While p * p is less than or equal to n:
    # 4.    If prime[p] is True:
    # 5.       Set all the multiples of p (from p*2) to False.
    # 6.    Increment "p" by 1.
    # 7. Sum every index i in [2, n] whose prime[i] is True.
    # 8. Return the sum.
    # Write your code here
    prime = [True] * (n + 1)
    p = 2
    while p * p <= n:
        if prime[p]:
            for i in range(p * 2, n + 1, p):
                prime[i] = False
        p += 1
    return sum(i for i in range(2, n + 1) if prime[i])


def larger_of_two(a, b):
    """ Write a function that returns the larger of two integers a and b (return a if they are equal). """
    # Let's think step by step
    # Input: a, an integer; b, an integer
    # Output: result, an integer
    # 1. If a is greater than b:
    # 2.    Set result to a.
    # 3. Else if b is greater than a:
    # 4.    Set result to b.
    # 5. Else:
    # 6.    Set result to a.
    # 7. Return result.
    # Write your code here
    if a > b:
        result = a
    elif b > a:
        result = b
    else:
        result = a
    return result


def count_vowels(s):
    """ Write a function to count the number of vowels in a given string s. """
    # Let's think step by step
    # Input: s, a string
    # Output: count, an integer
    # 1. Initialize count to 0.
    # 2. Define the set of vowels (both cases).
    # 3. For each character ch in s:
    # 4.    If ch is a vowel:
    # 5.       Increment count by 1.
    # 6. Return count.
    # Write your code here
    count = 0
    vowels = set("aeiouAEIOU")
    for ch in s:
        if ch in vowels:
            count += 1
    return count'''

# ---- 3 competitive (stdin/stdout) demonstrations -- SCoT method preserved, format adapted ----
COMP_DEMOS = '''#QUESTION:
The first line contains an integer n. The second line contains n integers. Print the sum of the even integers.
#SOLUTION:
# Let's think step by step
# Input: read n from stdin, then read n integers
# Output: print the sum of the even integers
# 1. Read all of standard input and split into tokens.
# 2. Parse n and then the list of n integers.
# 3. Initialize total to 0.
# 4. For each integer x in the list:
# 5.    If x is even:
# 6.       Add x to total.
# 7. Print total.
# Write your code here
import sys
data = sys.stdin.read().split()
n = int(data[0])
nums = list(map(int, data[1:1 + n]))
total = 0
for x in nums:
    if x % 2 == 0:
        total += x
print(total)

#QUESTION:
The first line contains an integer n. The second line contains n integers. Print the maximum value.
#SOLUTION:
# Let's think step by step
# Input: read n, then n integers
# Output: print the maximum integer
# 1. Read and parse n and the n integers.
# 2. Initialize best to the first integer.
# 3. For each remaining integer x:
# 4.    If x is greater than best:
# 5.       Set best to x.
# 6. Print best.
# Write your code here
import sys
data = sys.stdin.read().split()
n = int(data[0])
nums = list(map(int, data[1:1 + n]))
best = nums[0]
for x in nums[1:]:
    if x > best:
        best = x
print(best)

#QUESTION:
A single line contains an integer n. For i from 1 to n, print "Even" if i is even, otherwise print "Odd", one per line.
#SOLUTION:
# Let's think step by step
# Input: read a single integer n
# Output: for each i in 1..n print Even or Odd
# 1. Read n.
# 2. For i from 1 to n:
# 3.    If i is even:
# 4.       Print "Even".
# 5.    Else:
# 6.       Print "Odd".
# Write your code here
import sys
n = int(sys.stdin.read().split()[0])
for i in range(1, n + 1):
    if i % 2 == 0:
        print("Even")
    else:
        print("Odd")'''

OUT_NOTE = ("\nFollowing the same format as the examples, first write the problem-solving process (an Input/Output "
            "structure and numbered steps using sequential, branch, and loop structures) as comments, then write "
            "the complete final code. Put the whole answer (comments + code) in a single ```python code block.")


def scot_prompt(rec):
    if R.BENCHMARK in ("code_contests", "xcodeeval"):
        return (INSTRUCTION + "\n\nHere are some demonstration examples:\n\n" + COMP_DEMOS +
                "\n\n#QUESTION:\n" + rec["prompt"] + "\n#SOLUTION:\n# Let's think step by step" + OUT_NOTE)
    # function-level (humaneval_nfr): paper-exact
    return (INSTRUCTION + "\n\nHere are some demonstration examples:\n\n" + FUNC_DEMOS +
            "\n\nInput code:\n" + rec["prompt"].rstrip() + "\n    # Let's think step by step" + OUT_NOTE)


def extract_code(txt):
    import ast
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", txt, re.DOTALL)
    if blocks:   # pick the most code-like fenced block (not a SCoT-only comment block)
        codeyness = lambda b: sum(1 for ln in b.split("\n") if ln.strip() and not ln.strip().startswith("#"))
        return max(blocks, key=codeyness).strip("\n")
    # no fenced block (some models, e.g. gemini): drop prose/markdown, keep comments + code,
    # then trim to the largest parseable prefix so stray trailing prose can't break it.
    keep = []
    for ln in txt.split("\n"):
        s = ln.strip()
        if s.startswith("```") or s.startswith("#QUESTION") or s.startswith("#SOLUTION"):
            continue
        if (s == "" or s.startswith("#") or ln[:1] in " \t"
                or re.match(r'^[A-Za-z_][\w.]*\s*[=:(\[]', s)
                or re.match(r'^(import|from|def|class|for|while|if|elif|else|return|print|try|except|finally|with|raise|break|continue|yield|assert|global|sys|input)\b', s)):
            keep.append(ln)
    lines = "\n".join(keep).strip("\n").split("\n")
    while lines:
        try:
            ast.parse("\n".join(lines)); break
        except Exception:
            lines = lines[:-1]
    return "\n".join(lines)


def gen(rec):
    out = R.chat([{"role": "user", "content": scot_prompt(rec)}], temperature=0.8, top_p=0.95, role="gen")
    return extract_code(out)


if __name__ == "__main__":
    bench, model, provider, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else (6 if provider == "openrouter" else 2)
    cands = int(sys.argv[7]) if len(sys.argv) > 7 else 1     # paper uses 20 for unbiased pass@k; 1 = match our setup
    R.BENCHMARK = bench; R.MODEL = model; R.PROVIDER = provider; R.MAX_TOKENS = 4096
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "codes.json")
    codes = json.load(open(path)) if os.path.exists(path) else {}
    probs = R.load_problems()[:n]
    todo = [(pid, rec) for pid, rec in probs if pid not in codes]
    print(f"SCoT | {bench} | {model} ({provider}) | {len(probs)} problems, cands={cands} ({len(todo)} to do, w={workers})", flush=True)
    lock = threading.Lock()

    def work(item):
        pid, rec = item
        try:
            samples = [gen(rec) for _ in range(cands)]
        except Exception as e:
            print(f"  !! {pid}: {str(e)[:100]}", flush=True); return
        with lock:
            codes[pid] = samples if cands > 1 else samples[0]
            json.dump(codes, open(path, "w"))
            if len(codes) % 20 == 0:
                print(f"  {len(codes)}/{len(probs)}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))
    print("DONE", bench, model, flush=True)
