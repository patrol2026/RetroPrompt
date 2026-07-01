"""
Structural BACKWARD reasoning (HoarePrompt-style, reversed).

HoarePrompt pushes state FORWARD via the strongest-postcondition calculus, statement by
statement from the function head to the end. This module does the dual: it walks the body
BOTTOM-UP, statement by statement, threading a natural-language "continuation" summary
(= what the code AFTER the current statement does) backward until it reaches the function
definition -- at which point the accumulated summary IS "what the whole function does".

It reuses HoarePrompt's two key ideas:
  * per-statement Hoare-triple reasoning (one LLM call per statement), and
  * few-shot k-induction for loops (unroll k times, then generalize).

`structural_backward(code, chat)` returns the natural-language spec of the code, suitable
to drop into the same `compare(forward_spec, backward_spec)` step used by the one-shot variant.
`chat(prompt)` must be a function that takes a single prompt string and returns the model reply
(retroprompt_gen.chat already matches this).
"""
import ast
import re

STOP = "After this point the function returns / ends."

_STMT_PROMPT = """You are summarizing what a Python function does, one statement at a time.

You are given ONE statement and a description of what the code that runs AFTER this statement does
("continuation"). Describe what the code does starting from THIS statement onward.

Your description MUST account for BOTH:
  (a) what this statement does, AND
  (b) everything the continuation does afterward.
Do NOT drop or omit the continuation -- if this statement is trivial (e.g. an assignment or setup),
the behavior is dominated by the continuation, so keep it. Combine (a) and (b) into one concise
description of the full behavior from this point to the end (normal top-to-bottom execution).
Focus on WHAT it does (effect on variables / what is returned / printed), not how.

Statement:
```python
{stmt}
```
What the code AFTER this statement does (KEEP this in your answer):
{cont}

Reply strictly in the format:  Effect: **your description**"""

_IF_PROMPT = """You are summarizing what a Python function does.

An if/else statement runs. You are given the effect of each branch (each already including the
continuation that runs after the if/else block).

Condition: `{cond}`
If the condition is TRUE, the code does:
{if_eff}
If the condition is FALSE, the code does:
{else_eff}

Combine these into one concise natural-language description of what the if/else (and everything
after it) does, conditioned on `{cond}`.
Reply strictly:  Effect: **your description**"""

_LOOP_PROMPT = """You are summarizing what a Python loop does, using k-induction.

Loop:
```python
{loop_code}
```
What ONE iteration of the loop body does (already analyzed statement by statement):
{body_effect}

The loop body unrolled {k} times (to reveal the cumulative pattern):
```python
{unrolled}
```

1) Using the single-iteration effect above and the unrolled iterations, generalize what the loop
   does after ALL its iterations (cumulative effect on the variables / what it accumulates or
   computes / when it stops), not just k.
2) After the loop finishes, the following continuation runs:
{cont}

Give one concise natural-language description of what the loop AND everything after it does.
Reply strictly:  Effect: **your description**"""


def _extract(text):
    m = re.findall(r"Effect:\s*\*\*(.*?)\*\*", text, re.DOTALL)
    return (m[-1] if m else text).strip()


def _src(node):
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparsable>"


def _unroll(loop_node, k):
    body = "\n".join(_src(s) for s in loop_node.body)
    if isinstance(loop_node, ast.While):
        head = f"while {_src(loop_node.test)}:"
    else:
        head = f"for {_src(loop_node.target)} in {_src(loop_node.iter)}:"
    indented = "\n".join("    " + ln for ln in body.splitlines())
    return head + "\n" + "\n\n".join(["# iteration {}\n".format(i + 1) + indented for i in range(k)])


def _rec(trace, node, kind, effect, depth):
    if trace is not None:
        trace.append({"depth": depth, "type": kind, "node": _src(node).splitlines()[0], "effect": effect})


def _loop_effect(loop_node, cont, chat, k, trace=None, depth=0):
    head = (f"while {_src(loop_node.test)}:" if isinstance(loop_node, ast.While)
            else f"for {_src(loop_node.target)} in {_src(loop_node.iter)}:")
    loop_code = head + "\n" + "\n".join("    " + ln for s in loop_node.body for ln in _src(s).splitlines())
    # HoarePrompt-style: recurse INTO the loop body (statement by statement, incl. nested
    # loops/ifs) to get a precise single-iteration effect, then k-induct to the whole loop.
    body_effect = _walk(loop_node.body, "the loop checks its condition again and may run another iteration",
                        chat, k, trace, depth + 1)
    prompt = _LOOP_PROMPT.format(loop_code=loop_code, body_effect=body_effect,
                                 unrolled=_unroll(loop_node, k), k=k, cont=cont)
    eff = _extract(chat(prompt))
    _rec(trace, loop_node, "loop", eff, depth)
    return eff


def _stmt_effect(stmt, cont, chat, trace=None, depth=0):
    eff = _extract(chat(_STMT_PROMPT.format(stmt=_src(stmt), cont=cont)))
    _rec(trace, stmt, "stmt", eff, depth)
    return eff


def _if_effect(if_node, cont, chat, k, trace=None, depth=0):
    if_eff = _walk(if_node.body, cont, chat, k, trace, depth + 1)
    else_eff = _walk(if_node.orelse, cont, chat, k, trace, depth + 1) if if_node.orelse else cont
    prompt = _IF_PROMPT.format(cond=_src(if_node.test), if_eff=if_eff, else_eff=else_eff)
    eff = _extract(chat(prompt))
    _rec(trace, if_node, "if", eff, depth)
    return eff


def _walk(body, cont, chat, k, trace=None, depth=0):
    """Reverse-walk a list of statements, threading `cont` backward; return the summary of `body`+cont."""
    for stmt in reversed(body):
        if isinstance(stmt, (ast.For, ast.While)):
            cont = _loop_effect(stmt, cont, chat, k, trace, depth)
        elif isinstance(stmt, ast.If):
            cont = _if_effect(stmt, cont, chat, k, trace, depth)
        else:  # Assign, AugAssign, Return, Expr, Raise, Pass, etc.
            cont = _stmt_effect(stmt, cont, chat, trace, depth)
    return cont


def structural_backward(code, chat, k=3, trace=None):
    """Return a natural-language description of what `code` does, derived bottom-up.

    If `trace` is a list, it is filled with the per-statement annotations
    ({depth, type, node, effect}) produced during the backward walk.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "The code does not parse."
    func = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
    body = func.body if func else tree.body
    return _walk(body, STOP, chat, k, trace, 0)
