import os
import logging
from typing import Any
from starlette.applications import Starlette
import sys
import os

# Ensure local imports work by adding src to sys.path
from src.google.adk.runners import Runner
from src.google.adk.sessions import in_memory_session_service
from src.google.adk.agents import llm_agent, parallel_agent
from src.google.adk.tools import long_running_tool
from src.google.adk.apps import app as app_lib
from src.google.adk.apps.app import ResumabilityConfig
from src.google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from src.google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from src.google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from src.google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor, A2aAgentExecutorConfig
from src.google.adk.a2a.converters.request_converter import convert_a2a_request_to_agent_run_request
from src.google.adk.agents.run_config import RunConfig, StreamingMode

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities


InMemorySessionService = in_memory_session_service.InMemorySessionService


async def ask_for_city() -> dict[str, Any]:
  """Ask for a city."""
  return {"city": None}


async def fake_fun_fact_agent(city: str) -> dict[str, Any]:
  """Generate a fake fun fact."""
  return {
      "fun_fact": (
          "Did you know that Turin is the capital of Italy? Actually this is"
          " not true haha I fooled you looser idiot. But in 1861 Turin was the"
          " capital of Italy, when the country was unified."
      )
  }


def create_app() -> Starlette:
  # Create a simple ADK agent
  simple_agent_1 = llm_agent.LlmAgent(
      name="fun_fact_agent_1",
      model="gemini-2.5-flash",
      description="Agent specialized in generating fun facts.",
      instruction="""
                You are a fun fact generator. You will receive the request of the user and you will generate a fun fact about it.
                If no city is provided, you will ask for a city to the user, using the long running tool ask_for_city.
               """,
      tools=[long_running_tool.LongRunningFunctionTool(ask_for_city)],
  )

  simple_agent_2 = llm_agent.LlmAgent(
      name="fun_fact_agent_2",
      model="gemini-2.5-flash",
      description="Agent specialized in generating fun facts.",
      instruction="""
                You are a fun fact generator. Generate a fun fact about geography regardless of the input.
               """,
  )

  parallel = parallel_agent.ParallelAgent(
      name="parallel_agent",
      description="Agent specialized running multiple agents in parallel",
      sub_agents=[simple_agent_1, simple_agent_2],
  )

  # Manually setup A2A server
  async def create_runner():
    return Runner(
        app=app_lib.App(
            name="remote_agent",
            root_agent=parallel,
            resumability_config=ResumabilityConfig(is_resumable=True),
        ),
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=InMemoryCredentialService(),
    )

  def streaming_request_converter(request, part_converter):
    agent_run_request = convert_a2a_request_to_agent_run_request(
        request, part_converter
    )
    """
    if agent_run_request.run_config:
      agent_run_request.run_config.streaming_mode = StreamingMode.SSE
    else:
      agent_run_request.run_config = RunConfig(streaming_mode=StreamingMode.SSE)
    """
    return agent_run_request

  executor = A2aAgentExecutor(
      runner=create_runner,
      config=A2aAgentExecutorConfig(
          request_converter=streaming_request_converter
      ),
  )
  task_store = InMemoryTaskStore()
  request_handler = DefaultRequestHandler(executor, task_store)

  agent_card = AgentCard(
      name="remote_agent",
      url="http://test-agent",
      description="A fun fact generator agent",
      capabilities=AgentCapabilities(streaming=True),
      version="0.0.1",
      default_input_modes=["text/plain"],
      default_output_modes=["text/plain"],
      skills=[],
  )

  a2a_app_helper = A2AStarletteApplication(
      agent_card=agent_card, http_handler=request_handler
  )
  a2a_app = Starlette()
  a2a_app_helper.add_routes_to_app(a2a_app)
  return a2a_app
