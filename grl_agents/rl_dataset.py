from __future__ import annotations

from typing import List, Dict, Any

from grl_agents.agent_group_builder import AgentGroupBuilder
import asyncio


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

  async def generate_group_trajectories(
      self,
      index: int,
      reset: bool = True,
      max_workers: int = 4,
  ) -> List[List[Dict[str, Any]]]:
    """
    Build agent groups for the given index and concurrently collect trajectories
    from all agents in all groups.

    Returns a list with one element per group; each element is a list of
    trajectory dicts (one per agent in the group).
    """
    builders = self.get_batch(index)
    async def collect_one(builder: AgentGroupBuilder):
      # Create agents and collect trajectories concurrently per group
      agents = await builder.make_agents(parallel=True, max_workers=max_workers)
      return await builder.generate_trajectories(
          agents=agents, reset=reset, max_workers=max_workers
      )

    results = await asyncio.gather(*[collect_one(b) for b in builders])
    return results

  def generate_group_trajectories_sync(
      self, index: int, reset: bool = True, max_workers: int = 4
  ) -> List[List[Dict[str, Any]]]:
    """Synchronous wrapper for generate_group_trajectories()."""
    return asyncio.run(
        self.generate_group_trajectories(index=index, reset=reset, max_workers=max_workers)
    )
