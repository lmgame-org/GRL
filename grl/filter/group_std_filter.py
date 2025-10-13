# filters/group_std_filter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Tuple, List
import numpy as np
from .types import RolloutBatch  # or adjust import to where RolloutBatch lives

FilterType = Literal["std", "std_rev"]

@dataclass(frozen=True)
class GroupStdFilterConfig:
    rollout_filter_ratio: float          # e.g., 0.5
    rollout_filter_type: FilterType      # "std" or "std_rev"
    num_groups: int                      # G
    group_size: int                      # S

class GroupSelectorByStd:
    """Select groups by in-group std (or reversed)."""

    @staticmethod
    def group_stats(per_sample_scores: np.ndarray,
                    num_groups: int, group_size: int) -> Dict[str, np.ndarray]:
        # per_sample_scores: [N]
        group_scores = per_sample_scores.reshape(num_groups, group_size)  # [G,S]
        in_group_std  = group_scores.std(axis=-1)     # [G]
        in_group_max  = group_scores.max(axis=-1)     # [G]
        in_group_mean = group_scores.mean(axis=-1)    # [G]
        return {
            "group_scores": group_scores,
            "in_group_std": in_group_std,
            "in_group_max": in_group_max,
            "in_group_mean": in_group_mean,
        }

    @staticmethod
    def select_groups(stats: Dict[str, np.ndarray],
                      k: int, mode: FilterType) -> np.ndarray:
        vals = stats["in_group_std"]
        if mode == "std":      # keep largest std
            top_idx = np.argpartition(-vals, kth=k-1)[:k]
            # stable sort for readability (optional)
            top_idx = top_idx[np.argsort(-vals[top_idx])]
        elif mode == "std_rev":  # keep smallest std
            top_idx = np.argpartition(vals, kth=k-1)[:k]
            top_idx = top_idx[np.argsort(vals[top_idx])]
        else:
            raise ValueError(f"Unknown filter type: {mode}")
        return top_idx  # shape [k]

class GroupStdRolloutFilter:
    """
    Filters a RolloutBatch by keeping entire groups
    whose in-group std satisfies the chosen criterion.
    Behavior matches the original implementation.
    """

    def __init__(self, cfg: GroupStdFilterConfig):
        self.cfg = cfg

    def _build_keep_indices(self, top_groups: np.ndarray) -> np.ndarray:
        # Expand groups to member indices
        Gs = self.cfg.group_size
        base = top_groups * Gs  # [k]
        offsets = np.arange(Gs) # [S]
        keep_idx = (base[:, None] + offsets[None, :]).reshape(-1)  # [k*S]
        keep_idx.sort()
        return keep_idx

    def _metrics(self, stats: Dict[str, np.ndarray],
                 top_groups: np.ndarray) -> Dict[str, float]:
        ig_std  = stats["in_group_std"]
        ig_max  = stats["in_group_max"]
        ig_mean = stats["in_group_mean"]
        return {
            "rollout/in_group_std":  float(ig_std.mean()),
            "rollout/in_group_max":  float(ig_max.mean()),
            "rollout/in_group_mean": float(ig_mean.mean()),
            "rollout/chosen_in_group_std":  float(ig_std[top_groups].mean()),
            "rollout/chosen_in_group_max":  float(ig_max[top_groups].mean()),
            "rollout/chosen_in_group_mean": float(ig_mean[top_groups].mean()),
        }

    def apply(self, batch: RolloutBatch) -> Tuple[RolloutBatch, Dict[str, float]]:
        # Fast path: keep all
        ratio = self.cfg.rollout_filter_ratio
        G, S = self.cfg.num_groups, self.cfg.group_size
        if ratio >= 1.0:
            # compute metrics anyway for logging
            per_sample = batch.reward_scores.sum(axis=-1)
            stats = GroupSelectorByStd.group_stats(per_sample, G, S)
            metrics = self._metrics(stats, np.arange(G))
            return batch, metrics

        # 1) scores per sample, then per group
        per_sample = batch.reward_scores.sum(axis=-1)     # [N]
        stats = GroupSelectorByStd.group_stats(per_sample, G, S)

        # 2) choose groups
        k = max(1, int(ratio * G))
        top_groups = GroupSelectorByStd.select_groups(stats, k, self.cfg.rollout_filter_type)

        # 3) expand to flat indices
        keep_idx = self._build_keep_indices(top_groups)

        # 4) filter tensor fields
        input_ids     = batch.input_ids[keep_idx]
        loss_mask     = batch.loss_mask[keep_idx]
        reward_scores = batch.reward_scores[keep_idx]

        # 5) filter agent_raw_data consistently
        agent_raw_out = {}
        N = per_sample.shape[0]
        for key, value in batch.agent_raw_data.items():
            if isinstance(value, np.ndarray) and value.shape[:1] == (N,):
                agent_raw_out[key] = value[keep_idx]
            elif isinstance(value, list) and len(value) == int(N):
                agent_raw_out[key] = [value[i] for i in keep_idx.tolist()]
            else:
                agent_raw_out[key] = value  # untouched (mismatched leading dim)

        # 6) metrics
        metrics = self._metrics(stats, top_groups)

        # 7) new batch
        filtered = RolloutBatch(
            input_ids=input_ids,
            loss_mask=loss_mask,
            reward_scores=reward_scores,
            agent_raw_data=agent_raw_out,
            meta_info=batch.meta_info,
        )
        return filtered, metrics

# Convenience functional API
def filter_rollout_batch(batch: RolloutBatch,
                         num_groups: int, group_size: int,
                         ratio: float, mode: FilterType):
    cfg = GroupStdFilterConfig(
        rollout_filter_ratio=ratio,
        rollout_filter_type=mode,
        num_groups=num_groups,
        group_size=group_size,
    )
    return GroupStdRolloutFilter(cfg).apply(batch)
