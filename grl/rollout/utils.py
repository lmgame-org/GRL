# ─────────────────── UTILS ───────────────────

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass
class RolloutBatch:
    """
    Universal rollout batch container (numpy-only) for multi-turn PPO.

    Fields:
      - input_ids: np.ndarray shape [N, L]
      - loss_mask: np.ndarray shape [N, L-1] (aligned to next-token prediction)
      - reward_scores: np.ndarray shape [N, L-1] (sparse rewards aligned to targets)
      - agent_raw_data: Dict with keys:
          - env_ids: np.ndarray dtype=object shape [N]
          - group_ids: np.ndarray dtype=object shape [N]
          - messages_list: np.ndarray dtype=object shape [N]
      - meta_info: Dict[str, Any]
    """

    input_ids: np.ndarray
    loss_mask: np.ndarray
    reward_scores: np.ndarray
    agent_raw_data: Dict[str, np.ndarray]
    meta_info: Dict[str, Any]
