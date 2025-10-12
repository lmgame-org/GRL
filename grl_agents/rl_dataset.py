from __future__ import annotations

from typing import List, Dict, Any

import asyncio
from grl_agents.agent_group_builder import AgentGroupBuilder

"""
RLDataset

Public API:

- __init__(base_config, groups_per_batch=1, seeds_per_group=1, seed_start=0)
  Configure deterministic seed generation and per-group agent counts.

- get_batch(index)
  Return a list of AgentGroupBuilder instances for the given index.

- async collect_group_trajectories(index)
  Build agent groups for the given index and concurrently collect final rollout
  states from all agents in all groups. Returns a list per group; each element
  is a list of trajectory dicts (one per agent in the group).
"""


class RLDataset:
  """Dataset helper that assembles `AgentGroupBuilder` groups and collects trajectories."""

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
    for _ in range(self.groups_per_batch):
      seeds = [
          self.seed_start + index * self.seeds_per_group + i
          for i in range(self.seeds_per_group)
      ]
      builders.append(AgentGroupBuilder(config=self.base_config, seeds=seeds))
    return builders

  async def collect_group_trajectories(self, index: int) -> List[List[Dict[str, Any]]]:
    builders = self.get_batch(index)

    async def collect_one(builder: AgentGroupBuilder) -> List[Dict[str, Any]]:
      agents = await builder.make_agents()
      return await builder.generate_full_trajectories(agents=agents)

    return await asyncio.gather(*[collect_one(b) for b in builders])
