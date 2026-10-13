"""Start the FastAPI service with a Windows-compatible asyncio loop."""

from __future__ import annotations

import asyncio
import sys

import uvicorn

from app.config import settings


def event_loop_factory() -> asyncio.AbstractEventLoop:
    """Use SelectorEventLoop because psycopg async rejects ProactorEventLoop."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


def main() -> None:
    config = uvicorn.Config(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        loop="scripts.run_api:event_loop_factory",
    )
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
