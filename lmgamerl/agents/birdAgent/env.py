import json
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

from datasets import load_dataset
from lmgamerl.agents.agent_utils import all_seed
from lmgamerl.agents.base_env import BaseEnv
from pathlib import Path



class BirdEnv(BaseEnv):
    """Environment that evaluates SQL generation by comparing *execution results*
    instead of string‑level exact matching.
    """

    CODE_FENCE_RE = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ── ensure dataset_path is absolute ──────────────────────────────
        raw_path = self.config.get("dataset_path", "")
        if raw_path and not os.path.isabs(raw_path):
            # this file lives at <repo>/lmgamerl/agents/birdAgent/env.py
            # go up to <repo>
            repo_root = Path(__file__).resolve().parents[3]
            abs_path = (repo_root / raw_path).resolve()
            self.config["dataset_path"] = str(abs_path)
        # ─────────────────────────────────────────────────────────────────

        # ── ensure db_root is absolute ───────────────────────────────────
        raw_db = self.config.get("db_root",
                                 "datasets/bird_train/train/train_databases")
        if raw_db and not os.path.isabs(raw_db):
            # go up to <repo>
            repo_root = Path(__file__).resolve().parents[3]
            abs_db = (repo_root / raw_db).resolve()
            self.config["db_root"] = str(abs_db)
        # ─────────────────────────────────────────────────────────────────
        # #-----------------------------------------------------------------
        # # debugging: set db_root to home for testing
        # self.config["dataset_path"] = str(Path.home() / "datasets/bird_train/train/train_with_schema.json")
        # self.config["db_root"] = str(Path.home() / "datasets/bird_train/train/train_databases")
        # #-----------------------------------------------------------------


        # Load dataset from local JSON or HuggingFace repo
        if self.config.get('dataset_path', '').endswith(".json"):
            with open(self.config['dataset_path'], encoding="utf-8") as f:
                self.dataset = [json.loads(line) for line in f]
        else:
            self.dataset = load_dataset(
                self.config.get('dataset_path', ''), 
                split=self.config.get('split', 'train')
            )

        # Runtime state variables
        self.sample: Dict[str, str] | None = None
        self.question: str | None = None
        self.gold_sql: str | None = None
        self.db_id: str | None = None
        self.step_num: int = 0
        self.render_cache: str | None = None
        # Track DB readiness per episode
        self._db_ready: bool = False
        self._last_db_id: str | None = None
        self._db_poll_interval_s: float = 0.1
        # Per-step SQL execution timeout (seconds); default 180s = 3 minutes
        self._step_timeout_s: float = float(self.config.get("step_timeout_seconds", 180))

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Cheap whitespace / case normalisation before execution."""
        sql = sql.strip().rstrip(";")
        sql = re.sub(r"\s+", " ", sql)
        return sql.strip()

    def _db_file(self) -> str:
        return os.path.join(
            self.config['db_root'],
            self.db_id,
            f"{self.db_id}.sqlite"
        )

    def _wait_for_db_ready(self) -> None:
        """
        Wait until the backing SQLite DB file for current db_id exists.
        Do not initialize or write schema here to avoid concurrent initialization
        contention. We assume another prior run has created these DBs.
        """
        # Fast path: already marked ready for this db_id and file exists
        db_path = self._db_file()
        if self._db_ready and self._last_db_id == self.db_id and os.path.exists(db_path):
            return

        # If db file not present, wait indefinitely (user requested no time limit)
        while not os.path.exists(db_path):
            time.sleep(self._db_poll_interval_s)

        # Mark as ready for this db_id
        self._db_ready = True
        self._last_db_id = self.db_id


    def _execute_sql(self, sql: str, timeout_s: float | None = None) -> Tuple[bool, Union[List[Tuple[Any, ...]], str]]:
        """Execute one SQL statement with a per-call timeout.

        Returns (ok, result_or_error). On timeout, returns (False, f"TIMEOUT({timeout_s}s)").
        """
        # Ensure DB file exists before attempting to connect
        self._wait_for_db_ready()
        db_file = self._db_file()

        try:
            # Effective per-statement timeout via progress handler
            eff_timeout = float(self._step_timeout_s if timeout_s is None else timeout_s)
            start_time = time.time()
            timeout_flag = False

            # Keep connection-level busy timeout <= eff_timeout
            with sqlite3.connect(db_file, timeout=min(eff_timeout, 10.0)) as conn:
                conn.execute("PRAGMA foreign_keys = OFF;")
                conn.execute(f"PRAGMA busy_timeout = {int(eff_timeout * 1000)};")

                def _progress_handler():
                    nonlocal timeout_flag
                    if (time.time() - start_time) > eff_timeout:
                        timeout_flag = True
                        return 1  # abort current statement
                    return 0

                # Call roughly every N SQLite VM opcodes
                conn.set_progress_handler(_progress_handler, 1000)
                try:
                    cur = conn.cursor()
                    cur.execute(sql)
                    rows = cur.fetchall()
                    cur.close()
                finally:
                    # Clear handler for safety
                    conn.set_progress_handler(None, 0)

            # Sort safely, treating None as negative infinity
            rows_sorted = sorted(rows, key=lambda row: [
                x if x is not None else float('-inf') for x in row
            ])
            return True, rows_sorted

        except Exception as exc:
            # Distinguish timeout from other SQL errors
            msg = str(exc)
            if 'interrupted' in msg.lower() or 'cancel' in msg.lower() or 'abort' in msg.lower():
                return False, f"TIMEOUT({int(eff_timeout)}s)"
            return False, msg



    def reset(self, seed: int | None = None, **kwargs) -> Any:
        with all_seed(seed):
            self.sample = random.choice(self.dataset)

        self.question = self.sample["question"]
        self.gold_sql = self.sample["SQL"]
        self.db_id = self.sample["db_id"]
        self.schema = self.sample["schema"]
        self.step_num = 0
        # New episode: mark DB as not-checked for readiness
        self._db_ready = False
        self._last_db_id = None

        self.render_cache = f"[DB schema:\n{self.schema}] {self.question}"
        return self.render_cache

    def step(self, action: str):
        self.step_num += 1
        match = self.CODE_FENCE_RE.search(action or "")

        if not match:
            observation = "❌ No ```sql``` block detected."
            reward = -0.5
            done = True
            info = {
                "action_is_valid(code_block)": False,
                "success": False
            }
            self.render_cache = observation
            return observation, reward, done, info

        submitted_sql = self._normalize_sql(match.group(1))

        # Execute submission first with timeout; only run gold if submission succeeds
        sub_ok, sub_res_or_err = self._execute_sql(submitted_sql, timeout_s=self._step_timeout_s)
        if not sub_ok:
            # Timed out or errored; format observation and continue multi-turn
            timed_out = isinstance(sub_res_or_err, str) and sub_res_or_err.startswith("TIMEOUT(")
            if timed_out:
                observation = (
                    f"⏱️ SQL execution exceeded {int(self._step_timeout_s)}s. "
                    "Please simplify the query and try again with a succinct ```sql``` block."
                )
                reward = 0.0
                done = self.step_num >= self.config.get('max_steps', 5)
                info = {
                    "action_is_valid(code_block)": True,
                    "success": False,
                    "timeout": True,
                }
                self.render_cache = observation
                return observation, reward, done, info

        # If submission succeeded, execute gold for comparison (also guarded by timeout)
        gold_ok, gold_res_or_err = self._execute_sql(self.gold_sql, timeout_s=self._step_timeout_s)

        result_match = False
        sql_error_msg = ""

        if gold_ok and sub_ok:
            result_match = gold_res_or_err == sub_res_or_err
        else:
            # Prefer to surface the *submission* error, fall back to gold error.
            sql_error_msg = sub_res_or_err if not sub_ok else gold_res_or_err

        reward = (1.0 / (2 ** (self.step_num - 1))) if result_match else 0.0
        done = result_match or self.step_num >= self.config.get('max_steps', 5)

        if result_match:
            observation = "~~~Result matches~~~!"
        else:
            observation = (
                f"!!!Result mismatch. {('SQL error: ' + sql_error_msg) if sql_error_msg else ''}"
            ).strip()

        info = {
            "action_is_valid(code_block)": True,
            "success": result_match
        }

        self.render_cache = observation
        return observation, reward, done, info

    def render(self) -> str:
        return self.render_cache
