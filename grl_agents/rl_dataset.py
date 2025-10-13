from __future__ import annotations

from typing import List, Dict, Any, Sequence, Optional

import asyncio
from grl_agents.agent_group_builder import AgentGroupBuilder
from grl_agents import get_agent_cls

"""
RLDataset

Public API:

- __init__(base_configs: List[Dict], seeds: List[int], group_nums: Optional[List[int]] = None, group_sizes: Optional[List[int]] = None, agent_names: Optional[List[str]] = None)
  Configure one or many groups per (seed, config). For each i: number of groups is
  `group_nums[i]` (default 1). Group seeds are `seeds[i] + local_group_id`.
  Each group's size is from `group_sizes[i]` (default 1).

- get_batch()
  Return a flat list of `AgentGroupBuilder` instances for all groups across all i.

- async collect_group_trajectories()
  Concurrently collect final rollout states for all groups. Returns a list where
  each element is the per-agent rollout list for that group's builder.
"""


class RLDataset:
  """Dataset of groups with optional replication per index.

  For each i, build `group_nums[i]` groups (default 1). Group j uses seed `seeds[i] + j`.
  Each group's size is `group_sizes[i]` (default 1). Agent class comes from `agent_names[i]`
  or `config['agent_type']` (default 'sokobanCodingAgent').
  """

  def __init__(
      self,
      base_configs: Sequence[Dict[str, Any]],
      seeds: Sequence[int],
      group_nums: Optional[Sequence[int]] = None,
      group_sizes: Optional[Sequence[int]] = None,
      agent_names: Optional[Sequence[str]] = None,
  ):
    assert len(base_configs) == len(seeds), "base_configs and seeds must align"
    if group_nums is not None:
      assert len(group_nums) == len(base_configs), "group_nums must align with base_configs"
    if group_sizes is not None:
      assert len(group_sizes) == len(base_configs), "group_sizes must align with base_configs"
    if agent_names is not None:
      assert len(agent_names) == len(base_configs), "agent_names must align with base_configs"

    self.base_configs = list(base_configs)
    self.seeds = [int(s) for s in seeds]
    self.group_nums = [int(g) for g in group_nums] if group_nums is not None else None
    self.group_sizes = [int(g) for g in group_sizes] if group_sizes is not None else None
    self.agent_names = list(agent_names) if agent_names is not None else None

    # Prebuild one builder per index (one group per seed)
    self._builders: List[AgentGroupBuilder] = []

    global_group_counter = 0
    for i, cfg in enumerate(self.base_configs):
      base_seed = self.seeds[i]
      num_groups = int(self.group_nums[i]) if self.group_nums is not None else 1
      group_sz = int(self.group_sizes[i]) if self.group_sizes is not None else 1
      # Determine agent name then resolve class from registry
      cfg_agent_type = (
          str(cfg.get("agent_type"))
          if isinstance(cfg, dict) and cfg.get("agent_type") is not None
          else None
      )
      agent_name = (
          self.agent_names[i]
          if self.agent_names is not None
          else (cfg_agent_type or "sokobanCodingAgent")
      )
      agent_cls = get_agent_cls(agent_name)

      for local_group_id in range(num_groups):
        self._builders.append(
            AgentGroupBuilder(
                seed=base_seed + local_group_id,
                config=cfg,
                group_num=group_sz,
                group_id=global_group_counter,
                agent_id_offset=0,
                agent_cls=agent_cls,
                agent_name=agent_name,
            )
        )
        global_group_counter += 1

  def get_batch(self, index: Optional[int] = None, agent_name: Optional[str] = None, group_num: Optional[int] = None) -> List[AgentGroupBuilder]:
    """Return the prebuilt builders.

    Backward-compatible signature: extra parameters are ignored.
    """
    return list(self._builders)

  async def collect_group_trajectories(self) -> List[List[Dict[str, Any]]]:
    builders = self.get_batch()
    return await asyncio.gather(*[b.generate_full_trajectories() for b in builders])
