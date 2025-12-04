from typing import Dict
from ..agents.base_agent import BaseAgent
from ..agents.ts_fs_agent import TSFSAgent

# You can register more agents here
AGENTS: Dict[str, BaseAgent] = {
    TSFSAgent.id: TSFSAgent(),
}


def list_agents():
    return [
        {"id": agent.id, "name": agent.name, "description": agent.description}
        for agent in AGENTS.values()
    ]


def get_agent(agent_id: str) -> BaseAgent:
    if agent_id not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_id}")
    return AGENTS[agent_id]
