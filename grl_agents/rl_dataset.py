from __future__ import annotations

from typing import List, Dict, Any, Sequence, Optional

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
  """Dataset of groups, each defined by a base_config and a seed.

  Optionally supports one-shot builder construction across multiple agent types
  when group_nums and group_sizes are provided.
  """

  def __init__(
      self,
      base_configs: Sequence[Dict[str, Any]],
      seeds: Sequence[int],
      group_nums: Optional[Sequence[int]] = None,
      group_sizes: Optional[Sequence[int]] = None,
  ):
    assert len(base_configs) == len(seeds), "base_configs and seeds must align"
    if group_nums is not None:
      assert len(group_nums) == len(base_configs), "group_nums must align with base_configs"
    if group_sizes is not None:
      assert len(group_sizes) == len(base_configs), "group_sizes must align with base_configs"
    self.base_configs = list(base_configs)
    self.seeds = [int(s) for s in seeds]
    self.group_nums = [int(g) for g in group_nums] if group_nums is not None else None
    self.group_sizes = [int(g) for g in group_sizes] if group_sizes is not None else None

  def get_batch(
      self,
      index: Optional[int] = None,
      agent_name: str = "sokobanCodingAgent",
      group_num: int = 1,
      group_size: int = 1,
  ) -> List[AgentGroupBuilder]:
    """
    Build and return `AgentGroupBuilder` instances.

    - Default (index=None): Build all groups across all indices using the
      per-type `group_nums` and `group_sizes` configured at construction.
    - Per-index (index=int): Build `group_num` groups for the specific index,
      each with `group_size` agents. Seeds per group are deterministic:
      `seeds[index] + local_group_id`.
    """
    builders: List[AgentGroupBuilder] = []
    if index is None:
      assert self.group_nums is not None and self.group_sizes is not None, "group_nums/group_sizes required when index=None"
      for i, cfg in enumerate(self.base_configs):
        base_seed = self.seeds[i]
        num_groups = int(self.group_nums[i])
        group_sz = int(self.group_sizes[i])
        for local_group_id in range(num_groups):
          builders.append(
              AgentGroupBuilder(
                  seed=base_seed + local_group_id,
                  config=cfg,
                  group_num=group_sz,
                  agent_name=agent_name,
              )
          )
      return builders

    # index provided → per-index builders
    seed_base = self.seeds[index]
    cfg = self.base_configs[index]
    for local_group_id in range(int(group_num)):
      builders.append(
          AgentGroupBuilder(
              seed=seed_base + local_group_id,
              config=cfg,
              group_num=int(group_size),
              agent_name=agent_name,
          )
      )
    return builders

  async def collect_group_trajectories(self, index: Optional[int] = None, agent_name: str = "sokobanCodingAgent", group_num: int = 1, group_size: int = 1) -> List[List[Dict[str, Any]]]:
    builders = self.get_batch(index=index, agent_name=agent_name, group_num=group_num, group_size=group_size)

    async def collect_one(builder: AgentGroupBuilder) -> List[Dict[str, Any]]:
      agents = await builder.make_agents()
      return await builder.generate_full_trajectories(agents=agents)

    return await asyncio.gather(*[collect_one(b) for b in builders])
