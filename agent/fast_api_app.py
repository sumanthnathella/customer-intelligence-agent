"""
agent/fast_api_app.py — FastAPI app exposing the ADK agent for agents-cli playground.

Configured via agents-cli-manifest.yaml with `agent_directory: agent`.
"""
from __future__ import annotations

from google.adk.cli.fast_api import get_fast_api_app

from agent.agent import build_agent

# Build the agent instance
agent = build_agent()

# The ADK CLI automatically discovers this app variable
app = get_fast_api_app(agents=[agent])
