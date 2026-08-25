"""Process-wide ranker for the P1-026 dynamic conditions.

The P1-026 contract fixes ``hybrid_three_way`` as the scorer for every ranking
condition -- BM25, dense retrieval, and a cross-encoder rerank fused by
Reciprocal Rank Fusion. That is the best-performing method from the closed
P1-025 line, and it is used here unchanged: this module builds a callable
around it, and owns no ranking logic of its own.

Cost is why this is a module-level singleton rather than a per-request object.
A ranking condition calls the scorer once per completion cycle, so a full
P1-026 grid makes on the order of ten thousand ranking calls. Loading model
weights or re-embedding the 126 tool documents at that rate would dominate the
run. Both are done once, on first use, and reused for the process lifetime.

Model choices are frozen here and deliberately not configurable, so a P1-026 run
cannot silently rank with a different model than the studies it extends. What
they are frozen *to* changed on 2026-08-24, and the reason is recorded because
the constants no longer match the P1-025 line's originals.

**MiniLM was replaced by Qwen (2026-08-24).** The supervisor rejected MiniLM on
2026-08-18 (traceability ``T-116``); because ``hybrid_three_way`` uses a MiniLM
twice -- dense *and* cross-encoder -- both stages were replaced, since a partial
swap would leave the objection standing. Every P1-026 replay study since
(``RA`` Qwen arm, ``RA2``, ``RA3``, ``RA4``) ran on Qwen, so leaving this module
on MiniLM meant the executable path ranked with a different instrument than the
replay line that justifies running it at all. Note this is not a contract
amendment: the P1-026 contract freezes no ranker and never named one.

**This also changes where the models run.** MiniLM was loaded locally through
``sentence-transformers``; the Qwen pair is served over HTTP by ScaDS, through
``active_registry_core.scads_ranking_backends`` -- the same backend the replay
runners already use, which closes a live divergence between the two paths rather
than opening one.

Two consequences follow and are handled below. The singleton's cost argument
gets *stronger*, not weaker, because a ranking call is now a network round trip;
and the rerank endpoint is **not bit-deterministic** -- ``RA4`` measured
absolute availability drifting up to ``0.0141`` between runs -- so both caches
are wired by default, which removes that drift rather than bounding it.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Frozen. See the module docstring for why these changed on 2026-08-24.
DENSE_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
CROSS_ENCODER_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"

# Caching is on by default. Without it the rerank endpoint's non-determinism
# reappears in every P1-026 number, and a full grid repeats on the order of ten
# thousand identical scoring calls over an unchanging 126-tool inventory.
_CACHE_DIR = Path(
    os.getenv("P1_026_RANKER_CACHE_DIR")
    or Path(__file__).resolve().parents[2] / "p1_026_ranker_cache"
)

_lock = threading.Lock()
_ranker = None


class P1026RankerUnavailableError(RuntimeError):
    """Raised when the ranking models cannot be loaded in this environment."""


def _load_ranker():
    from active_registry_core import (
        McpToolCapability,
        McpToolInventory,
        hybrid_three_way_top_b,
    )

    try:
        from active_registry_core.scads_ranking_backends import (
            build_scads_cross_encoder,
            build_scads_embedder,
        )
    except ImportError as error:  # pragma: no cover - environment dependent
        raise P1026RankerUnavailableError(
            "P1-026 ranking conditions need active_registry_core's ScaDS "
            "backends; only p1_026_full runs without a ranker"
        ) from error

    from .config import config

    # The key is required, and its absence is raised here rather than surfacing
    # later as an opaque provider error. A ranking condition without a reachable
    # endpoint cannot rank at all, so failing at load is the honest outcome.
    api_key = getattr(config, "LLM_API_KEY", "") or ""
    if not api_key:
        raise P1026RankerUnavailableError(
            "P1-026 ranking conditions need LLM_API_KEY set for the ScaDS "
            "endpoint; only p1_026_full runs without a ranker"
        )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Building P1-026 ranking backends (once per process): dense=%s cross=%s "
        "cache=%s",
        DENSE_MODEL_NAME, CROSS_ENCODER_MODEL_NAME, _CACHE_DIR,
    )

    embed = build_scads_embedder(
        api_key=api_key,
        model=DENSE_MODEL_NAME,
        cache_path=_CACHE_DIR / "p1_026_embeddings.jsonl",
    )
    cross_encode = build_scads_cross_encoder(
        api_key=api_key,
        model=CROSS_ENCODER_MODEL_NAME,
        cache_path=_CACHE_DIR / "p1_026_rerank.jsonl",
    )

    def rank(tools, query: str) -> tuple[str, ...]:
        """Full ranking of the discovered tools for one query.

        A synthetic single-server inventory is built because the scorer's
        contract is defined over ``McpToolInventory``. Server identity is not
        used by ``hybrid_three_way_top_b``, and the adapter must not invent
        server semantics, so a constant server ID is used and tool names are
        returned unchanged.
        """

        inventory = McpToolInventory(
            capabilities=tuple(
                McpToolCapability(
                    server_id="dynamic",
                    tool_name=tool.name,
                    description=tool.description,
                    input_schema=dict(tool.input_schema or {"type": "object"}),
                )
                for tool in tools
            )
        )
        by_id = inventory.by_id
        ordered_ids = hybrid_three_way_top_b(
            inventory,
            prompt=query,
            budget=len(inventory.capabilities),
            embed=embed,
            cross_encode=cross_encode,
        )
        return tuple(by_id[capability_id].tool_name for capability_id in ordered_ids)

    return rank


def build_p1_026_ranker():
    """Return the process-wide ranker, loading models on first use."""

    global _ranker
    if _ranker is None:
        with _lock:
            if _ranker is None:
                _ranker = _load_ranker()
    return _ranker


def reset_p1_026_ranker_for_tests() -> None:
    """Drop the cached ranker so a test can inject its own."""

    global _ranker
    with _lock:
        _ranker = None


def set_p1_026_ranker_for_tests(ranker) -> None:
    """Install a fake ranker, so selector wiring is testable without weights."""

    global _ranker
    with _lock:
        _ranker = ranker
