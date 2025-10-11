from __future__ import annotations

from typing import List, Sequence, Dict, Any, Type
from concurrent.futures import ProcessPoolExecutor

from grl_agents.puzzle_agents.sokoban_coding_agent.sokoban_coding_agent import SokobanCodingAgent


class AgentGroupBuilder:
  """
  Builds a group of agents.

  Training loops use this to instantiate fresh agent+env instances per episode.
  """

  def __init__(self, seeds: Sequence[int], config: Dict[str, Any], agent_cls: Type[SokobanCodingAgent] = SokobanCodingAgent, agent_name: str = "sokobanAgent") -> None:
    self.seeds = list(seeds)
    self.config = config
    self.agent_cls = agent_cls
    self.agent_name = agent_name

  async def make_agents(self, parallel: bool = True, max_workers: int = 4) -> Sequence[SokobanCodingAgent]:
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
          tag=f"{self.agent_name}-{seed}"
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
        tag=f"{self.agent_name}-{seed}"
      )
      agents.append(agent)
    return agents


