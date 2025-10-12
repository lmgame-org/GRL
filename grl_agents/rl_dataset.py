from __future__ import annotations

from typing import List, Dict, Any, Sequence

import asyncio
from grl_agents.agent_group_builder import AgentGroupBuilder

"""
RLDataset

Public API:

- __init__(base_configs: List[Dict], seeds: List[int])
  Configure a dataset as a list of groups, each with its own base_config and seed.

- get_batch(index)
  Return a list of AgentGroupBuilder instances for the given index (single group index).

- async collect_group_trajectories(index, agent_name="sokobanCodingAgent", group_num=1)
  Build the agent group for the given index and concurrently collect final rollout
  states from all agents in the group. Returns a single-element list containing
  the list of per-agent trajectory dicts for that group.
"""


class RLDataset:
  """Dataset of groups, each defined by a base_config and a seed."""

  def __init__(self, base_configs: Sequence[Dict[str, Any]], seeds: Sequence[int]):
    assert len(base_configs) == len(seeds), "base_configs and seeds must align"
    self.base_configs = list(base_configs)
    self.seeds = [int(s) for s in seeds]

  def get_batch(self, index: int, agent_name: str = "sokobanCodingAgent", group_num: int = 1) -> List[AgentGroupBuilder]:
    seed = self.seeds[index]
    cfg = self.base_configs[index]
    return [AgentGroupBuilder(seed=seed, config=cfg, group_num=group_num, agent_name=agent_name)]

  async def collect_group_trajectories(self, index: int, agent_name: str = "sokobanCodingAgent", group_num: int = 1) -> List[List[Dict[str, Any]]]:
    builders = self.get_batch(index, agent_name=agent_name, group_num=group_num)

    async def collect_one(builder: AgentGroupBuilder) -> List[Dict[str, Any]]:
      agents = await builder.make_agents()
      return await builder.generate_full_trajectories(agents=agents)

    return await asyncio.gather(*[collect_one(b) for b in builders])
