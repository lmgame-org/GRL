class BaseEnv:
    """
    Minimal reference implementation for an environment class.

    Every concrete environment should inherit from this and implement
    `reset`, `step`, `render`, and `close`.
    """

    def __init__(self, config=None, **kwargs):
        self.config = config or {}
        # Nothing else here—sub-classes add their own state.

    # ──────────────────────────────────────────────────────────
    # Required API (no-op stubs)
    # ──────────────────────────────────────────────────────────
    def reset(self, seed=None, **kwargs):
        """Reset the environment to an initial state."""
        pass

    def step(self, action):
        """
        Advance the environment by one timestep using `action`.
        Returns: observation, reward, done, info
        """
        pass

    def render(self, mode="text"):
        """Return a human-readable representation of the current state."""
        pass

    def close(self):
        """Clean up resources (files, sockets, etc.)."""
        pass
