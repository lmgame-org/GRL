from datasets import load_dataset
import re
import math
import random
from grl.agents.agent_utils import all_seed
from grl.agents.base_env import BaseEnv


class AIME24Env(BaseEnv):
  LATEX_SPACE = re.compile(r"\s+")
  BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
  TEXT_RE = re.compile(r"\\text\s*\{([^{}]*)\}")
  FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
  SQRT_BRACE_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
  SQRT_PLAIN_RE = re.compile(r"(?<![A-Za-z0-9_])\\sqrt\s*([A-Za-z0-9_.]+)")
  PUNCT_TAIL_RE = re.compile(r"[\.，。．]\s*$")

  @classmethod
  def _remove_dollar_and_paren_wrappers(cls, s: str) -> str:
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

  @classmethod
  def _strip_tex(cls, s: str) -> str:
    if s is None:
      return ""
    s = str(s)
    s = cls._remove_dollar_and_paren_wrappers(s)
    for _ in range(3):
      m = cls.BOXED_RE.search(s)
      if not m:
        break
      s = cls.BOXED_RE.sub(m.group(1), s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = cls.TEXT_RE.sub(lambda m: m.group(1), s)
    s = cls.FRAC_RE.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    s = cls.SQRT_BRACE_RE.sub(lambda m: f"sqrt({m.group(1)})", s)
    s = cls.SQRT_PLAIN_RE.sub(lambda m: f"sqrt({m.group(1)})", s)
    s = re.sub(r"(\d)\s*sqrt\s*\(\s*(\d+)\s*\)", r"\1*sqrt(\2)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("^", "**").replace("\\pm", "±")
    s = s.replace("√", "sqrt").replace("π", "pi")
    s = cls.LATEX_SPACE.sub(" ", s).strip()
    s = cls.PUNCT_TAIL_RE.sub("", s).strip()
    return s

  @classmethod
  def _split_answers(cls, s: str):
    s = cls._strip_tex(s)
    if not s:
      return []
    if "±" in s:
      return [s.replace("±", "+"), s.replace("±", "-")]
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

  @classmethod
  def _canonical_atomic(cls, a: str) -> str:
    a = cls._strip_tex(a)
    a = re.sub(r"\s*([+\-*/^=(),])\s*", r"\1", a)
    a = a.replace(", ", ",")
    return a

  @classmethod
  def _try_eval_numeric(cls, x: str):
    expr = cls._strip_tex(x)
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

  @classmethod
  def _set_equal(cls, golds, users) -> bool:
    gset = {cls._canonical_atomic(g) for g in golds}
    uset = {cls._canonical_atomic(u) for u in users}
    if gset == uset:
      return True
    matched = [False] * len(golds)
    for u in users:
      for gi, g in enumerate(golds):
        if matched[gi]:
          continue
        uc = cls._canonical_atomic(u)
        gc = cls._canonical_atomic(g)
        if uc == gc:
          matched[gi] = True
          break
        gok, gv = cls._try_eval_numeric(g)
        uok, uv = cls._try_eval_numeric(u)
        if gok and uok and abs(uv - gv) < 1e-9:
          matched[gi] = True
          break
    return all(matched)

  @classmethod
  def _answers_equal(cls, gold: str, user: str) -> bool:
    g_list = cls._split_answers(gold)
    u_list = cls._split_answers(user)
    if not u_list:
      return False
    return cls._set_equal(g_list, u_list)

  def __init__(self, config, **kwargs):
    super(AIME24Env, self).__init__()
    self.config = config
    self.dataset = load_dataset(
        self.config.get("dataset_path", "math-ai/aime24"),
        self.config.get("dataset_config", None),
        split=self.config.get("split", "test"),
    )
    self.current_sample = None
    self.current_unique_id = None
    self.current_question = None
    self.correct_answer = None
    self.step_num = None
    self.render_cache = None

  def extract_answer(self, answer):
    return str(answer).strip()

  def reset(self, seed=None):
    with all_seed(seed):
      question_data = random.choice(self.dataset)
    self.current_sample = question_data
    # AIME24 dataset fields: 'problem' and 'solution'
    self.current_question = question_data.get("problem", "")
    self.correct_solution = question_data.get("solution", "")
    self.correct_answer = self.extract_answer(self.correct_solution)
    self.render_cache = self.current_question
    self.step_num = 0
    return self.render_cache

  def step(self, action):
    is_correct, is_valid = self._check_answer(action)
    reward = 10.0 if is_correct else -0.1
    if is_correct:
      observation = "Correct!"
      done = True
    else:
      observation = "Incorrect. Please think again."
      done = False
    self.step_num += 1
    info = {
        "action_is_effective": True,
        "action_is_valid": is_valid,
        "success": is_correct,
    }
    self.render_cache = observation
    return self.render_cache, reward, done, info

  def _check_answer(self, user_answer):
    user_answer = "" if user_answer is None else str(user_answer)
    gold = self.correct_answer if self.correct_answer is not None else ""
    is_valid = len(self._strip_tex(user_answer)) > 0
    is_correct = self._answers_equal(gold, user_answer)
    return is_correct, is_valid

  def render(self):
    return self.render_cache

  def close(self) -> None:
    self._question = self._answer = None
