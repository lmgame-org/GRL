"""Utility functions for LaTeX parsing and math answer comparison."""

import re
import math

# Regex patterns for LaTeX parsing
LATEX_SPACE = re.compile(r"\s+")
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
TEXT_RE = re.compile(r"\\text\s*\{([^{}]*)\}")
FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
SQRT_BRACE_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
SQRT_PLAIN_RE = re.compile(r"(?<![A-Za-z0-9_])\\sqrt\s*([A-Za-z0-9_.]+)")
PUNCT_TAIL_RE = re.compile(r"[\.，。．]\s*$")


def remove_dollar_and_paren_wrappers(s: str) -> str:
  """Remove LaTeX dollar signs and parenthesis wrappers from a string."""
  s = s.strip()
  if (s.startswith("$$") and s.endswith("$$") and len(s) >= 4) or (
      s.startswith("$") and s.endswith("$") and len(s) >= 2
  ):
    s = s.strip("$").strip()
  if (s.startswith("\\(") and s.endswith("\\)")) or (
      s.startswith("\\[") and s.endswith("\\]")
  ):
    s = s[2:-2].strip()
  return s


def strip_tex(s: str) -> str:
  """Strip LaTeX formatting and convert to plain text expression."""
  if s is None:
    return ""
  s = str(s)
  s = remove_dollar_and_paren_wrappers(s)
  for _ in range(3):
    m = BOXED_RE.search(s)
    if not m:
      break
    s = BOXED_RE.sub(lambda m: m.group(1), s)
  s = s.replace("\\left", "").replace("\\right", "")
  s = TEXT_RE.sub(lambda m: m.group(1), s)
  s = FRAC_RE.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
  s = SQRT_BRACE_RE.sub(lambda m: f"sqrt({m.group(1)})", s)
  s = SQRT_PLAIN_RE.sub(lambda m: f"sqrt({m.group(1)})", s)
  s = re.sub(r"(\d)\s*sqrt\s*\(\s*(\d+)\s*\)", r"\1*sqrt(\2)", s)
  s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
  s = s.replace("^", "**").replace("\\pm", "±")
  s = s.replace("√", "sqrt").replace("π", "pi")
  s = LATEX_SPACE.sub(" ", s).strip()
  s = PUNCT_TAIL_RE.sub("", s).strip()
  return s


def split_answers(s: str) -> list:
  """Split a string containing multiple answers into a list."""
  s = strip_tex(s)
  if not s:
    return []
  if "±" in s:
    return [s.replace("±", "+"), s.replace("±", "-")]
  parts = [p.strip() for p in s.split(",")]
  return [p for p in parts if p]


def canonical_atomic(a: str) -> str:
  """Convert an atomic answer to canonical form for comparison."""
  a = strip_tex(a)
  a = re.sub(r"\s*([+\-*/^=(),])\s*", r"\1", a)
  a = a.replace(", ", ",")
  return a


def try_eval_numeric(x: str) -> tuple:
  """
  Try to evaluate a string as a numeric expression.
  
  Returns:
    Tuple of (success: bool, value: float/complex)
  """
  expr = strip_tex(x)
  if not expr:
    return False, 0.0
  gate = expr.replace("pi", "").replace("sqrt", "")
  if not re.fullmatch(r"[0-9+\-*/().,\s^jJieE]+", gate):
    return False, 0.0
  expr = expr.replace("^", "**")
  expr = re.sub(r"(?<=\d)i\b", "j", expr)
  if "," in expr:
    return False, 0.0
  ns = {"__builtins__": {}}
  ns.update({"sqrt": math.sqrt, "pi": math.pi, "e": math.e, "j": 1j})
  try:
    val = eval(expr, ns, {})
    if isinstance(val, (int, float, complex)):
      return True, val
  except Exception:
    pass
  return False, 0.0


def set_equal(golds: list, users: list) -> bool:
  """Check if two sets of answers are equal (order-independent)."""
  gset = {canonical_atomic(g) for g in golds}
  uset = {canonical_atomic(u) for u in users}
  if gset == uset:
    return True
  matched = [False] * len(golds)
  for u in users:
    for gi, g in enumerate(golds):
      if matched[gi]:
        continue
      uc = canonical_atomic(u)
      gc = canonical_atomic(g)
      if uc == gc:
        matched[gi] = True
        break
      gok, gv = try_eval_numeric(g)
      uok, uv = try_eval_numeric(u)
      if gok and uok and abs(uv - gv) < 1e-9:
        matched[gi] = True
        break
  return all(matched)


def answers_equal(gold: str, user: str) -> bool:
  """Check if user answer matches the gold answer."""
  g_list = split_answers(gold)
  u_list = split_answers(user)
  if not u_list:
    return False
  return set_equal(g_list, u_list)
