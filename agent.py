"""Conventional LiveKit CLI entrypoint.

``lk agent dev`` discovers this file automatically. The implementation remains
in ``bot.py`` to preserve the repository's historical import path.
"""

from livekit.agents import cli

from bot import server


if __name__ == "__main__":
    cli.run_app(server)
