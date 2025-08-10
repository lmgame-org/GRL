import argparse
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
        # runtime connection cached per active db_id
        self._conn: sqlite3.Connection | None = None
        self._conn_db_id: str | None = None
        self._gold_ok: bool | None = None
        self._gold_rows: List[Tuple[Any, ...]] | str | None = None

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

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Cheap whitespace / case normalisation before execution."""
        sql = sql.strip().rstrip(";")
        sql = re.sub(r"\s+", " ", sql)
        return sql.strip()

    def _db_dir(self) -> str:
        return os.path.join(self.config['db_root'], self.db_id)

    def _db_file(self) -> str:
        return os.path.join(self._db_dir(), f"{self.db_id}.sqlite")

    def _ready_marker(self) -> str:
        return os.path.join(self._db_dir(), ".initialized")

    def _lock_file(self) -> str:
        return os.path.join(self._db_dir(), ".init_lock")

    def _ensure_db_initialized(self) -> None:
        """
        Ensure the SQLite file for current db_id exists and has schema.
        Uses a tiny on-disk lock file and a ready marker to avoid concurrent init.
        """
        db_dir = self._db_dir()
        db_file = self._db_file()
        ready = self._ready_marker()
        lockf = self._lock_file()

        os.makedirs(db_dir, exist_ok=True)
        if os.path.exists(db_file) and os.path.exists(ready):
            return

        # Try to become initializer
        started_init = False
        try:
            fd = os.open(lockf, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            started_init = True
        except FileExistsError:
            started_init = False

        if started_init:
            try:
                # Create DB and apply schema once
                with sqlite3.connect(db_file, timeout=float(self.config.get('sqlite_init_timeout', 60))) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    schema_sql = self.sample.get("schema") or self.schema
                    if not schema_sql:
                        raise RuntimeError("Missing schema; cannot initialize database.")
                    conn.executescript(schema_sql)
                    conn.commit()
                Path(ready).write_text("ok", encoding="utf-8")
            finally:
                try:
                    os.remove(lockf)
                except FileNotFoundError:
                    pass
        else:
            # Wait for initializer to finish
            deadline = time.time() + float(self.config.get('sqlite_init_timeout', 60))
            while time.time() < deadline:
                if os.path.exists(ready) and os.path.exists(db_file):
                    break
                time.sleep(0.05)
            # proceed regardless; if still not ready, connection open may fail which we surface

    def _open_ro_connection(self) -> sqlite3.Connection:
        """Open a read-only connection to the current db_id file."""
        db_file = self._db_file()
        uri = f"file:{db_file}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=float(self.config.get('sqlite_timeout', 5.0)))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn


    def _execute_sql(self, sql: str) -> Tuple[bool, Union[List[Tuple[Any, ...]], str]]:
        """Execute one SQL statement using cached read-only connection; returns sorted rows."""
        try:
            assert self._conn is not None, "Connection not initialized"
            cur = self._conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cur.close()
            rows_sorted = sorted(rows, key=lambda row: [x if x is not None else float('-inf') for x in row])
            return True, rows_sorted
        except Exception as exc:
            return False, str(exc)



    def reset(self, seed: int | None = None, **kwargs) -> Any:
        with all_seed(seed):
            self.sample = random.choice(self.dataset)

        self.question = self.sample["question"]
        self.gold_sql = self.sample["SQL"]
        self.db_id = self.sample["db_id"]
        self.schema = self.sample["schema"]
        self.step_num = 0

        self.render_cache = f"[DB schema:\n{self.schema}] {self.question}"
        # Ensure DB exists/initialized, then open cached read-only connection and cache gold result
        try:
            self._ensure_db_initialized()
            if self._conn is None or self._conn_db_id != self.db_id:
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                self._conn = self._open_ro_connection()
                self._conn_db_id = self.db_id
            # Cache gold result once per episode
            self._gold_ok, self._gold_rows = self._execute_sql(self.gold_sql)
        except Exception as e:
            # Defer failure to step
            self._gold_ok, self._gold_rows = False, f"init error: {e}"
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

        # Execute submitted SQL; gold is cached from reset
        gold_ok, gold_res_or_err = self._gold_ok, self._gold_rows
        sub_ok, sub_res_or_err = self._execute_sql(submitted_sql)

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
