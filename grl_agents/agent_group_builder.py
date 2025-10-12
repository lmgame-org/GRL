from __future__ import annotations

from typing import List, Sequence, Dict, Any, Type
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio

from grl_agents.puzzle_agents.sokoban_coding_agent.sokoban_coding_agent import SokobanCodingAgent


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
      agent_name: str = "sokobanAgent",
  ) -> None:
    self.seeds = list(seeds)
    self.config = config
    self.agent_cls = agent_cls
    self.agent_name = agent_name

  async def make_agents(
      self, parallel: bool = True, max_workers: int = 4
  ) -> Sequence[SokobanCodingAgent]:
    """
    Build a group of fresh agents.

    If parallel=True, seeds are preprocessed via ProcessPoolExecutor to leverage
    multiple processes (useful if seed preparation involves heavier work in future).
    Agents themselves are constructed in-process to avoid cross-process object transfer.
    """
    if not parallel:
      return [
          self.agent_cls(
              config=self.config,
              group_id=0,
              agent_id=idx,
              seed=seed,
              tag=f"{self.agent_name}-{seed}",
          )
          for idx, seed in enumerate(self.seeds)
      ]

    def _prepare_seed(s: int) -> int:
      return int(s)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
      prepared: List[int] = list(pool.map(_prepare_seed, self.seeds))

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

  async def generate_trajectories(
      self,
      agents: Sequence[SokobanCodingAgent] | None = None,
      reset: bool = False,
      max_workers: int = 4,
  ) -> List[Dict[str, Any]]:
    """
    Concurrently collect final rollout states from all agents in this group.

    If `agents` is None, fresh agents are created via make_agents().
    If `reset` is True, each agent is reset with the group's precomputed seeds
    before collecting trajectories.

    Returns a list of rollout state dicts as produced by agent.get_final_rollout_states().
    """
    if agents is None:
      agents = await self.make_agents(parallel=True, max_workers=max_workers)

    loop = asyncio.get_running_loop()

    # Use a dedicated thread pool for both reset and collection
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      if reset:
        # Map each agent to its corresponding seed if available
        def _reset_with_seed(agent: SokobanCodingAgent, sd: int | None):
          agent.reset(seed=sd)
          return True

        reset_tasks = []
        for idx, agent in enumerate(agents):
          seed = self.seeds[idx] if idx < len(self.seeds) else None
          reset_tasks.append(
              loop.run_in_executor(executor, _reset_with_seed, agent, seed)
          )
        await asyncio.gather(*reset_tasks)

      def _collect(agent: SokobanCodingAgent) -> Dict[str, Any]:
        return agent.get_final_rollout_states()

      tasks = [loop.run_in_executor(executor, _collect, agent) for agent in agents]
      results: List[Dict[str, Any]] = await asyncio.gather(*tasks)
      return results

  def generate_trajectories_sync(
      self,
      agents: Sequence[SokobanCodingAgent] | None = None,
      reset: bool = False,
      max_workers: int = 4,
  ) -> List[Dict[str, Any]]:
    """
    Synchronous convenience wrapper around generate_trajectories().
    """
    return asyncio.run(
        self.generate_trajectories(agents=agents, reset=reset, max_workers=max_workers)
    )
