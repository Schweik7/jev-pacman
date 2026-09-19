from .baseline import GreedyAgent, RandomAgent
from .jev import JevAgent


def make_agent(name, seed=None, mock_latency_ms=(70, 500), include_map=True):
    if name == "random":
        return RandomAgent(seed)
    if name == "greedy":
        return GreedyAgent()
    if name == "mockjev":
        return JevAgent.mock(latency_ms=mock_latency_ms, seed=seed, include_map=include_map)
    if name == "jev":
        return JevAgent.from_typesafe(include_map=include_map)
    if name == "jev-openrouter":
        return JevAgent.from_openrouter(include_map=include_map)
    raise ValueError(f"未知智能体: {name}")


__all__ = ["GreedyAgent", "RandomAgent", "JevAgent", "make_agent"]
