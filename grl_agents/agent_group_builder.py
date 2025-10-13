from __future__ import annotations

from typing import List, Dict, Any, Type, Sequence
import asyncio

from grl_agents.puzzle_agents.sokoban_coding_agent.sokoban_coding_agent import SokobanCodingAgent

"""
AgentGroupBuilder

Public API (kept intentionally small and async):

- __init__(seed, config, group_num, agent_cls, agent_name)
  Construct a builder for a single group. It will create `group_num` agents
  with deterministic seeds: seed, seed+1, ..., seed+group_num-1.
  Each agent is instantiated with: config, group_id=0, agent_id, seed, tag.

- async make_agents()
  Create agents for the group using the sequential seeds.

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
      seed: int,
      config: Dict[str, Any],
      group_num: int,
      group_id: int = 0,
      agent_id_offset: int = 0,
      agent_cls: Type[SokobanCodingAgent] = SokobanCodingAgent,
      agent_name: str = "sokobanCodingAgent",
  ) -> None:
    self.seed = int(seed)
    self.group_num = int(group_num)
    self.config = config
    self.group_id = int(group_id)
    self.agent_id_offset = int(agent_id_offset)
    self.agent_cls = agent_cls
    self.agent_name = agent_name

  async def make_agents(self) -> Sequence[SokobanCodingAgent]:
    """
    Build a group of fresh agents.
    """
    # All agents in this group share the same environment seed
    agents: List[SokobanCodingAgent] = []
    for idx in range(self.group_num):
      agent = self.agent_cls(
          config=self.config,
          group_id=self.group_id,
          agent_id=self.agent_id_offset + idx,
          seed=self.seed,
          tag=self.agent_name,
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
