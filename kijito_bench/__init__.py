"""Benchmark harnesses that measure Kijito (kijito.ai) through its public MCP endpoint.

Modules: client (the MCP client), llm (reader/judge providers), runner (checkpointing, caching,
statistics), baselines (the BM25 notes control), and one module per benchmark: longmemeval, stale,
stale_agent, groupmembench. Run one with `python -m kijito_bench.<benchmark> --help`.
"""
__version__ = "0.1.0"
