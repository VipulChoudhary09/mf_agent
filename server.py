"""Compatibility entrypoint for the former FastAPI/Daily server.

Use ``python server.py dev`` for development or ``python server.py start`` for
production. LiveKit now owns room creation, dispatch, media transport, and the
agent-server health endpoint.
"""

from livekit.agents import cli

from bot import server


if __name__ == "__main__":
    cli.run_app(server)
