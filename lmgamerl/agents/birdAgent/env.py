import argparse
import json
import os
import random
import re
import sqlite3
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
            # assume this file lives at <repo>/agents/birdAgent/env.py
            repo_root = Path(__file__).resolve().parents[2]  # adjust if your layout differs
            abs_path = (repo_root / raw_path).resolve()
            self.config["dataset_path"] = str(abs_path)
        # ─────────────────────────────────────────────────────────────────

        # ── ensure db_root is absolute ───────────────────────────────────
        raw_db = self.config.get("db_root",
                                 "datasets/bird_train/train/train_databases")
        if raw_db and not os.path.isabs(raw_db):
            repo_root = Path(__file__).resolve().parents[2]
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

    def _db_file(self) -> str:
        return os.path.join(
            self.config['db_root'],
            self.db_id,
            f"{self.db_id}.sqlite"
        )


    def _execute_sql(self, sql: str) -> Tuple[bool, Union[List[Tuple[Any, ...]], str]]:
        """Execute *one* SQL statement and fetch all results."""
        db_file = self._db_file()

        try:
            init_new_db = not os.path.exists(db_file)

            with sqlite3.connect(db_file) as conn:
                conn.execute("PRAGMA foreign_keys = OFF;")

                if init_new_db:
                    schema_sql = self.sample.get("schema")
                    if schema_sql:
                        conn.executescript(schema_sql)
                    else:
                        return False, "Missing schema; cannot initialize database."

                cur = conn.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                cur.close()

            # Sort safely, treating None as negative infinity
            rows_sorted = sorted(rows, key=lambda row: [
                x if x is not None else float('-inf') for x in row
            ])
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

        # Execute gold and submitted SQL
        gold_ok, gold_res_or_err = self._execute_sql(self.gold_sql)
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
