from __future__ import annotations

from typing import List, Sequence, Dict, Any, Type
import asyncio

from grl_agents.puzzle_agents.sokoban_coding_agent.sokoban_coding_agent import SokobanCodingAgent

"""
AgentGroupBuilder

Public API (kept intentionally small and async):

- __init__(seeds, config, agent_cls, agent_name)
  Construct a builder for a single group. Each provided seed maps 1:1 to an agent.
  The `agent_cls` is instantiated per seed with params: config, group_id=0, agent_id, seed, tag.

- async make_agents()
  Create agents for all seeds.

- async generate_full_trajectories(agents=None)
  Collect final rollout states from all agents in the group.
  If `agents` is None, it calls `make_agents`.
"""

class AgentGroupBuilder:
  """
  Builds a group of agents.

  Training loops use this to instantiate fresh agent+env instances per episode.
  """

  def __init__(
      self,
      seeds: Sequence[int],
      config: Dict[str, Any],
      agent_cls: Type[SokobanCodingAgent] = SokobanCodingAgent,
      agent_name: str = "sokobanCodingAgent",
  ) -> None:
    self.seeds = list(seeds)
    self.config = config
    self.agent_cls = agent_cls
    self.agent_name = agent_name

  async def make_agents(self) -> Sequence[SokobanCodingAgent]:
    """
    Build a group of fresh agents.
    """
    prepared: List[int] = [int(s) for s in self.seeds]

    agents: List[SokobanCodingAgent] = []
    for idx, seed in enumerate(prepared):
      agent = self.agent_cls(
          config=self.config,
          group_id=0,
          agent_id=idx,
          seed=seed,
          tag=f"{self.agent_name}-{seed}",
      )
      agents.append(agent)
    return agents

  async def generate_full_trajectories(
      self,
      agents: Sequence[SokobanCodingAgent] | None = None,
  ) -> List[Dict[str, Any]]:
    """
    Concurrently collect final rollout states from all agents in this group.

    If `agents` is None, fresh agents are created via make_agents().

    Returns a list of rollout state dicts as produced by agent.get_final_rollout_states().
    """
    if agents is None:
      agents = await self.make_agents()

    results: List[Dict[str, Any]] = await asyncio.gather(
        *[agent.aget_final_rollout_states() for agent in agents]
    )
    return results
