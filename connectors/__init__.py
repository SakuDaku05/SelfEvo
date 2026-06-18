"""
connectors — plug-and-play agent adapters for the MASC/SePO framework.
"""
from .base_agent import BaseAgent
from .agent_connector import AgentConnector

__all__ = ["BaseAgent", "AgentConnector"]
