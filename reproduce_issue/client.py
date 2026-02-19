import logging
import asyncio
import httpx
import sys
import os

from src.google.adk import Runner
from src.google.adk.sessions import in_memory_session_service
from src.google.adk.agents import run_config, remote_a2a_agent
from src.google.adk.agents.run_config import RunConfig
from src.google.adk.apps import app as app_lib
from src.google.adk.agents import llm_agent, sequential_agent
from src.google.adk.apps.app import ResumabilityConfig
from src.google.adk.cli import cli
from google.genai import types 
from a2a.client import ClientConfig, ClientFactory
from a2a.types import TransportProtocol
from reproduce_issue.server import create_app

# Configure ADK logging
logger = logging.getLogger("google_adk")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

InMemorySessionService = in_memory_session_service.InMemorySessionService
StreamingMode = run_config.StreamingMode

async def main(argv):

  query = "Tell me a fun fact about Turin"
  session_service = InMemorySessionService()
  my_user_id = "adk_adventurer_001"
  session = await session_service.create_session(
      app_name="fun_fact_agent", user_id=my_user_id
  )

  # Manually setup A2A server
  a2a_app = create_app()

  # Create an HTTP client that communicates directly with the A2A app
  # We use the app as the transport to avoid needing a separate server process
  client = httpx.AsyncClient(
      transport=httpx.ASGITransport(app=a2a_app), base_url="http://test-agent"
  )

  # Create the remote agent using the client
  remote_agent = remote_a2a_agent.RemoteA2aAgent(
      name="remote_agent",
      agent_card="http://test-agent/.well-known/agent.json",
      httpx_client=client,
  )
  runner = Runner(
      app=app_lib.App(
          name="fun_fact_agent",
          root_agent=remote_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      ),
      session_service=session_service,
      app_name="fun_fact_agent",
  )

  async for event in runner.run_async(
      user_id=my_user_id,
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text=query)], role="user"),
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  ):
    print(
        f"EVENT: {event.content}, {event.long_running_tool_ids}, \n  Partial:"
        f" {event.partial}"
    )

async def adk_web(argv):
  del argv  # Unused.

  """
  a2a_app = create_app()
  # Create an HTTP client that communicates directly with the A2A app
  # We use the app as the transport to avoid needing a separate server process
  client = httpx.AsyncClient(
      transport=httpx.ASGITransport(app=a2a_app), base_url="http://test-agent"
  )
  """

  # Create the remote agent using the client
  remote_agent = remote_a2a_agent.RemoteA2aAgent(
      name="remote_agent",
      agent_card="http://127.0.0.1:8000/.well-known/agent.json",
      a2a_client_factory=ClientFactory(
          ClientConfig(
            streaming=True,
          )
      ),
  )
  summarize_agent = llm_agent.LlmAgent(
      name="summarize_agent",
      model="gemini-2.5-flash",
      description="Agent specialized in changing the input.",
      instruction="""
                You will receive a text in input and you will add french words to it..
               """,
  )
  sequential = sequential_agent.SequentialAgent(
      name="sequential_agent",
      description="Agent specialized running multiple agents in sequence",
      sub_agents=[remote_agent, summarize_agent],
  )
  # Initialize a custom loader to mount the agent explicitly
  from src.google.adk.cli.utils.base_agent_loader import BaseAgentLoader
  class SingleAgentLoader(BaseAgentLoader):
      def __init__(self, agent):
          self.agent = agent
      def load_agent(self, agent_name: str):
          return app_lib.App(
              name=self.agent.name,
              root_agent=self.agent,
              resumability_config=ResumabilityConfig(is_resumable=True),
          )
      def list_agents(self):
          return [self.agent.name]
      def list_agents_detailed(self):
          return [{
              "name": self.agent.name,
              "root_agent_name": self.agent.name,
              "description": "",
              "language": "python",
              "is_computer_use": False
          }]

  # Pull all the local dependencies
  from src.google.adk.cli.adk_web_server import AdkWebServer
  from src.google.adk.sessions.in_memory_session_service import InMemorySessionService as InMemorySessionSvc
  from src.google.adk.memory.in_memory_memory_service import InMemoryMemoryService
  from src.google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
  from src.google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
  from src.google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
  from src.google.adk.evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
  from pathlib import Path
  import uvicorn
  import src.google.adk.cli.fast_api as fast_api_lib

  # Start the ADK web server mapping over the built test-agent directly in-process
  web_server = AdkWebServer(
      agent_loader=SingleAgentLoader(remote_agent),
      session_service=InMemorySessionSvc(),
      memory_service=InMemoryMemoryService(),
      artifact_service=InMemoryArtifactService(),
      credential_service=InMemoryCredentialService(),
      eval_sets_manager=InMemoryEvalSetsManager(),
      eval_set_results_manager=LocalEvalSetResultsManager("."),
      agents_dir=".",
  )

  # Fetch the generic ADK Dev UI path and prepare to attach it
  web_assets_dir = Path(fast_api_lib.__file__).parent.resolve() / "browser"
  app = web_server.get_fast_api_app(web_assets_dir=web_assets_dir)
  
  # Configure and start FastAPI with Uvicorn
  config = uvicorn.Config(app, host="127.0.0.1", port=8001)
  server = uvicorn.Server(config)
  await server.serve()


if __name__ == "__main__":
    asyncio.run(adk_web([]))
    # asyncio.run(main([]))
