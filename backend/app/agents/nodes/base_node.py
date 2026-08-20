"""
Base node interface (Strategy pattern) for the LangGraph agent workflow.

Each graph node (router, retrieve, generate, critique) implements this
interface. LangGraph itself just needs a callable per node, but wrapping
each one as a class with a `run(state) -> state` method makes each node
a genuine interchangeable strategy — e.g. swapping GenerateNode for an
alternate generation strategy later means writing one new class and
changing one line in graph.py, not touching the graph wiring itself.
"""

from abc import ABC, abstractmethod

from app.agents.state import AgentState


class AgentNode(ABC):
    """Strategy interface for a single LangGraph node."""

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        """Read from state, do this node's work, and return the updated state."""
        raise NotImplementedError

    def __call__(self, state: AgentState) -> AgentState:
        """Makes instances directly usable as LangGraph node callables."""
        return self.run(state)