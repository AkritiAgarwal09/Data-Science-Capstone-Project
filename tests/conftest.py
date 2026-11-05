"""Keep tests off the network: only RUN_LIVE_LLM=1 lets a test see the Vertex settings."""

import os

import pytest

VERTEX_SETTINGS = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "VERTEX_MODEL")


@pytest.fixture(autouse=True)
def no_vertex_unless_live(monkeypatch):
    """A teammate with Vertex set up would otherwise make real calls from any test that has no
    fake client, and the same test would pass or fail depending on whose machine runs it."""
    if not os.environ.get("RUN_LIVE_LLM"):
        for name in VERTEX_SETTINGS:
            monkeypatch.delenv(name, raising=False)
