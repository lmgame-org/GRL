from __future__ import annotations

from typing import List, Dict, Any

from grl_agents.agent_group_builder import AgentGroupBuilder


class RLDataset:
  """Dataset of AgentGroupBuilders."""

  def __init__(
      self,
      base_config: Dict[str, Any],
      groups_per_batch: int = 1,
      seeds_per_group: int = 1,
      seed_start: int = 0,
  ):
    self.base_config = base_config
    self.groups_per_batch = groups_per_batch
    self.seeds_per_group = seeds_per_group
    self.seed_start = seed_start

  def get_batch(self, index: int) -> List[AgentGroupBuilder]:
    builders: List[AgentGroupBuilder] = []
    for g in range(self.groups_per_batch):
      seeds = [
          self.seed_start + index * self.seeds_per_group + i
          for i in range(self.seeds_per_group)
      ]
      builders.append(AgentGroupBuilder(config=self.base_config, seeds=seeds))
    return builders
