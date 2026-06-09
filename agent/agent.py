"""
agent/agent.py — Single ReAct LlmAgent using Google ADK.

Thin reasoning layer: reads gbrain via tools, never computes metrics.
Model: OpenRouter nemotron-3-ultra:free via LiteLlm, with Ollama fallback.
"""
from __future__ import annotations

import logging
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from agent.tools import (
    tool_get_curated_set,
    tool_get_drivers,
    tool_get_exemplars,
    tool_get_l5_profile,
    tool_get_ops_hotspots,
    tool_get_systemic_drivers,
    tool_get_taxonomy,
    tool_get_top_l5,
    tool_get_zscore_spikes,
    tool_read_memory,
    tool_search_transcripts,
    tool_verify_claim,
    tool_write_insight,
)
from shared.config import AGENT_MODEL, AGENT_MODEL_FALLBACK, OPENROUTER_API_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instruction prompt
# ---------------------------------------------------------------------------

_INSTRUCTION = """You are a Customer Intelligence Analyst AI.
Your job is to read from the graph brain (gbrain) and produce a high-quality,
actionable weekly report on the most egregious customer pain points.

RULES:
1. You NEVER compute numbers. You only READ pre-computed metrics via tools.
2. Every claim must cite a tool call result.
3. Severity is computed deterministically; you only narrate it.
4. Deep dives must tie pain points to operational drivers (specific dimension values).
5. The report format is Markdown with sections per REPORT_SPEC.md.

WORKFLOW (one ReAct loop):
1. tool_get_curated_set → the importance-tagged backbone of the report
   (very_high / high / fair / low, with movement since last run). Also call
   tool_get_taxonomy + tool_get_zscore_spikes for context.
2. tool_get_systemic_drivers(only_bridges=True) → the "Systemic Drivers"
   section: single operational levers that drive MANY pain points at once.
   These are the highest-leverage fixes — lead with them.
3. For each very_high / escalated L5: tool_get_l5_profile(l5_id) for the
   deep dive (compositional groups + sub-themes in words with quotes), plus
   tool_get_drivers(l5_id) + tool_get_exemplars(l5_id) + tool_read_memory(l5_id).
   Narrate the sub-themes as the actual problems and name the dominant groups
   (vendor / product_category / fulfillment / region) so the reader can act.
4. Before asserting any driver, call tool_verify_claim(l5_id, dimension, value)
   and cite the returned lift / support / p_value. Use tool_search_transcripts
   to pull concrete supporting quotes.
5. tool_get_ops_hotspots → "Operations Hotspots" (ops-first view).
6. Synthesise root cause + recommended action; tool_write_insight per top L5.
7. Render the final report as Markdown text.

Be concise, factual, and specific. Use the exact lift values and p-values from
tools. Only state a driver after it returns verdict="verified". Never invent
dimension names or values.
"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

APP_NAME = "customer_intelligence_agent"


def _resolve_model(model: str) -> Any:
    """Return a model spec ADK understands.

    Native Gemini models are passed as plain strings; everything else
    (OpenRouter, Ollama, etc.) is routed through LiteLLM via the ADK
    ``LiteLlm`` wrapper so the agent can actually call the provider.
    """
    if model.startswith("gemini"):
        return model
    return LiteLlm(model=model)


def build_agent(model: str = AGENT_MODEL) -> LlmAgent:
    """Build and return the ReAct LlmAgent for the given model slug."""
    tools: list[Any] = [
        tool_get_taxonomy,
        tool_get_curated_set,
        tool_get_top_l5,
        tool_get_zscore_spikes,
        tool_get_systemic_drivers,
        tool_get_drivers,
        tool_get_l5_profile,
        tool_get_ops_hotspots,
        tool_get_exemplars,
        tool_search_transcripts,
        tool_verify_claim,
        tool_read_memory,
        tool_write_insight,
    ]

    return LlmAgent(
        model=_resolve_model(model),
        instruction=_INSTRUCTION,
        tools=tools,
        description="Customer Intelligence Agent that reads gbrain and writes reports.",
        name=APP_NAME,
    )


async def _arun(agent: LlmAgent, query: str) -> str:
    """Drive one ReAct conversation to completion via an ADK in-memory runner."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    user_id, session_id = "analyst", "report-session"
    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    message = types.Content(role="user", parts=[types.Part(text=query)])

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or ""
    return final_text


def run_headless(
    query: str = "Generate the weekly customer intelligence report.",
    model: str = AGENT_MODEL,
    fallback: str = AGENT_MODEL_FALLBACK,
) -> str:
    """Run the agent to completion without the interactive playground.

    Tries the primary model first (OpenRouter Nemotron). On any failure
    (missing key, rate limit, network) it transparently falls back to the
    local Ollama model so the flow still completes offline.
    """
    import asyncio

    if model.startswith("openrouter") and not OPENROUTER_API_KEY:
        logger.warning(
            "OPENROUTER_API_KEY not set; using local fallback model %s.", fallback
        )
        model = fallback

    logger.info("Running agent headless on %s", model)
    try:
        return asyncio.run(_arun(build_agent(model), query))
    except Exception as exc:  # noqa: BLE001 - resilience is the point here
        if model != fallback:
            logger.warning("Primary model %s failed (%s); falling back to %s.", model, exc, fallback)
            return asyncio.run(_arun(build_agent(fallback), query))
        raise
