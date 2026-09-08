"""Main FastAPI application for MCP eval."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request, Response

from .agent_eval import (
    dynamic_safe_trace,
    handle_run_mcp_eval,
    run_dynamic_mcp_eval_request,
)
from .dynamic_eval import DynamicMcpEvalError, HiddenToolRequestError
from .schema import RunAgentAPIRequestBody, RunDynamicAgentAPIRequestBody
from .errors import MCPClientToolExecutionError, MCPClientToolTimeoutError
from .config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Eval",
    description="Standalone MCP evaluation environment",
    version="0.1.0",
)


def _dynamic_failure_code(error: Exception) -> str:
    """Return a stable, prompt-free code for dynamic-route failures."""

    if isinstance(error, HiddenToolRequestError):
        return "hidden_tool_request"
    if isinstance(error, MCPClientToolTimeoutError):
        return "mcp_tool_call_timeout"
    if isinstance(error, MCPClientToolExecutionError):
        return "mcp_tool_execution_error"
    if isinstance(error, DynamicMcpEvalError):
        if str(error) == "model did not finish within max_turns":
            return "max_turns_exhausted"
        return "dynamic_contract_error"
    return "dynamic_host_error"


# Modules whose exceptions identify the stage a failure occurred in. The mapping
# is by module rather than by class because the provider SDKs raise a wide and
# changing set of classes, but always from a stable package.
_STAGE_BY_MODULE_PREFIX = (
    ("openai", "completion"),
    ("litellm", "completion"),
    ("httpx", "completion"),
    ("httpcore", "completion"),
    ("anthropic", "completion"),
    ("sentence_transformers", "selection"),
    ("transformers", "selection"),
    ("torch", "selection"),
)

_MESSAGE_LIMIT = 500


def _dynamic_failure_stage(error: Exception) -> str:
    """Best-effort stage attribution: selection | exposure | completion | tools.

    **Inferred, not instrumented.** The route does not currently mark which
    phase it is in, so this reads the exception's originating module. That is
    reliable for the provider SDKs -- which is the case that matters, because
    every unattributed failure so far has been a provider call -- and returns
    ``unknown`` rather than guessing when it is not.

    A true stage marker would require the dynamic route to record its phase as
    it advances. That is the stronger fix and is deliberately not done here.
    """

    if isinstance(error, HiddenToolRequestError):
        return "exposure"
    if isinstance(error, (MCPClientToolTimeoutError, MCPClientToolExecutionError)):
        return "tools"
    if isinstance(error, DynamicMcpEvalError):
        return "completion"

    module = type(error).__module__ or ""
    for prefix, stage in _STAGE_BY_MODULE_PREFIX:
        if module == prefix or module.startswith(prefix + "."):
            return stage
    return "unknown"


def _dynamic_failure_detail(error: Exception) -> Dict[str, Any]:
    """Return the failure payload: the code, plus what it takes to diagnose it.

    **This exists because a bucket code alone destroyed an experiment.** 150
    P1-026 pilot runs failed and recorded the single word ``dynamic_host_error``
    with an empty payload. It was read as a verdict on the task model. It was
    not -- the model had never been reached -- and the true cause is now
    permanently unrecoverable, because nothing persisted the exception. A later
    reproduction found the message was ``team not allowed to access model``,
    which is itself misleading: it names a permission tier that does not exist
    on that gateway, and the model had simply been withdrawn.

    So the code is kept exactly as it was -- callers and the contract depend on
    it -- and the diagnosis is added beside it.

    **The diagnosis is added for the untyped bucket only, and this is
    deliberate.** Every other code already names its own cause: a
    ``hidden_tool_request`` needs no explanation. More importantly, the typed
    errors' messages can carry **model-derived content** -- a
    ``HiddenToolRequestError`` names the tool the model asked for, which is
    model output. The existing tests guarantee those responses stay free of
    error text, and that guarantee is preserved here unchanged. Only
    ``dynamic_host_error`` -- the catch-all that destroyed attribution -- gains
    fields.

    **On the raw-material boundary.** ``message`` is a provider-originated error
    string, not a model response, and is truncated. Provider errors do not
    normally echo request content, but they are not guaranteed never to. The
    containment is that raw run records are gitignored and never committed; this
    field must not be promoted into a public record without being read first.
    """

    failure_code = _dynamic_failure_code(error)
    if failure_code != "dynamic_host_error":
        return {"failure_code": failure_code}

    detail: Dict[str, Any] = {
        "failure_code": failure_code,
        "stage": _dynamic_failure_stage(error),
        "exception_type": type(error).__name__,
        "exception_module": type(error).__module__,
    }

    # openai/litellm carry the upstream HTTP status; most other exceptions do not.
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        detail["upstream_status"] = status

    message = str(error)
    if message:
        detail["message"] = (
            message
            if len(message) <= _MESSAGE_LIMIT
            else message[:_MESSAGE_LIMIT] + "... [truncated]"
        )
    return detail


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests with their actual response status codes."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"{request.client.host}:{request.client.port} - "
        f'"{request.method} {request.url.path} HTTP/1.1" {response.status_code} '
        f"- {process_time:.3f}s"
    )

    return response


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "MCP Eval is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/v2/mcp_eval/run_agent")
async def run_agent(
    body: RunAgentAPIRequestBody,
    authorization: Optional[str] = Header(None),
):
    """
    MCP evaluation endpoint. The main entrypoint. For simplicity, no authentication or rate limiting is used.
    """
    logger.info(f"v2 API /run_agent called with model: {body.model}")

    try:
        # Process agent outputs and return results
        results = []
        async for agent_output in handle_run_mcp_eval(body):
            result = {
                "type": agent_output.type,
                "data": agent_output.data,
            }
            results.append(result)

        return results

    except MCPClientToolExecutionError as error:
        logger.error(f"MCP client tool execution error: {error}")
        raise HTTPException(status_code=500, detail={"error": str(error)})

    except Exception as error:
        logger.error(f"Error during MCP eval execution: {error}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Unknown error during mcp_eval: {str(error)}",
            },
        )


@app.post("/v2/mcp_eval/run_agent_dynamic")
async def run_agent_dynamic(
    body: RunDynamicAgentAPIRequestBody,
    authorization: Optional[str] = Header(None),
):
    """P1-016 dynamic-loop endpoint; the fixed-list route remains unchanged."""
    logger.info(f"v2 API /run_agent_dynamic called with model: {body.model}")

    try:
        dynamic_result = await run_dynamic_mcp_eval_request(body)
        return {
            "outputs": list(dynamic_result.outputs),
            "dynamic_trace": dynamic_safe_trace(dynamic_result),
            # Run metadata, deliberately outside "outputs" so that stream stays
            # a pure message list for consumers that index into it.
            "usage": dynamic_result.usage,
        }
    except Exception as error:
        detail = _dynamic_failure_detail(error)
        logger.error(
            "Dynamic MCP eval failed with %s at stage %s (%s.%s): %s",
            detail["failure_code"],
            detail.get("stage", "n/a"),
            detail.get("exception_module", type(error).__module__),
            detail.get("exception_type", type(error).__name__),
            error,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=detail) from error


def main():
    # Validate required configuration at startup
    config.validate_required_config()

    logger.info(f"Starting MCP Eval server on {config.HOST}:{config.PORT}")

    uvicorn.run(
        "mcp_completion.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,  # Set to True for development
        log_level=config.LOG_LEVEL.lower(),
        access_log=False,  # Disable default access logs (we have custom middleware)
    )


if __name__ == "__main__":
    main()
