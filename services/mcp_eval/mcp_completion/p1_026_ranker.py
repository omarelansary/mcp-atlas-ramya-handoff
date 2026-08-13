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

Model choices are the frozen ones from the P1-025 contract and are not
configurable here, so a P1-026 run cannot silently rank with a different model
than the line it extends.
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

logger = logging.getLogger(__name__)

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as error:  # pragma: no cover - environment dependent
        raise P1026RankerUnavailableError(
            "P1-026 ranking conditions need sentence-transformers installed in "
            "this service environment; only p1_026_full runs without it"
        ) from error

    logger.info("Loading P1-026 ranking models (once per process)")
    dense = SentenceTransformer(DENSE_MODEL_NAME)
    cross = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

    def embed(texts: Sequence[str]):
        return dense.encode(list(texts), convert_to_numpy=True)

    def cross_encode(pairs):
        return cross.predict(list(pairs))

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
