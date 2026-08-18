"""MCP evaluation functionality."""

import json
import logging
from dataclasses import asdict
from typing import AsyncGenerator, Dict, List, Union, Any, Optional, Sequence

from .mcp_client import MCPClient, SandboxMCPClient
from .llm import create_completion, _transform_tool_calls
from .schema import (
    RunAgentAPIRequestBody,
    RunDynamicAgentAPIRequestBody,
    Message,
    AssistantMessage,
    ToolCallOutputMessage,
    TextContent,
    ImageContent,
    ResourceContent,
    Content,
    CallToolResponse,
    SystemMessage,
    UserMessage,
)
from .errors import MCPClientToolExecutionError
from .config import config
from .dynamic_eval import run_dynamic_mcp_eval
from .dynamic_selectors import build_registered_dynamic_selector
from .p1_026_ranker import build_p1_026_ranker
from .p1_026_selectors import build_registered_p1_026_dynamic_selector
from active_registry_core import REGISTERED_P1_026_SELECTOR_IDS

logger = logging.getLogger(__name__)


class AgentOutput:
    """MCP eval output wrapper."""

    def __init__(self, output_type: str, data: Any):
        self.type = output_type
        self.data = data


class _ConfiguredDynamicSelector:
    """Explicit full/selected active-set source for the first endpoint smoke."""

    def __init__(self, active_tool_names: Optional[Sequence[str]]):
        self._active_tool_names = (
            None if active_tool_names is None else tuple(active_tool_names)
        )
        self.selector_id = (
            "p1_016_dynamic_full"
            if self._active_tool_names is None
            else "p1_016_explicit_active_set"
        )

    def select(self, *, raw_tools, visible_messages, cycle_index):
        if self._active_tool_names is None:
            return [tool["name"] for tool in raw_tools]
        return list(self._active_tool_names)


async def run_mcp_eval(
    mcp_client: MCPClient,
    model: str,
    messages: List[Message],
    max_turns: int,
    extra_body: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[AgentOutput, None]:
    """
    Simple MCP evaluation loop that keeps calling tools until the model decides there are no more tools to call.
    """
    tools = await mcp_client.list_tools()
    transformed_tools = _transform_tool_calls([tool.model_dump() for tool in tools])

    all_messages: List[Message] = list(messages)

    # Token accounting across the whole run. Emitted once at the end as a
    # separate output type; every existing consumer filters on type ==
    # "message", so this is additive and changes nothing for them.
    run_usage: Dict[str, Any] = {"cycles": 0, "reported_cycles": 0}

    for i in range(max_turns):
        assistant_message = None
        original_content = None

        try:
            # Use unified LiteLLM completion for all models
            result = await create_completion(
                model=model,
                messages=all_messages,
                tools=transformed_tools,
                extra_body=extra_body,
            )

            assistant_message = result.message
            original_content = result.original_content

            run_usage["cycles"] += 1
            if result.usage:
                run_usage["reported_cycles"] += 1
                for key, value in result.usage.items():
                    if isinstance(value, int):
                        run_usage[key] = run_usage.get(key, 0) + value

        except Exception as error:
            logger.error(f"Model create completion or parsing failed: {error}")
            # Re-raise as server error instead of graceful handling
            raise Exception(f"LLM completion failed: {error}")

        all_messages.append(assistant_message)

        yield AgentOutput("message", assistant_message.model_dump())

        tool_calls = assistant_message.tool_calls or []

        if tool_calls:
            for tool_call in tool_calls:
                try:
                    # Parse tool arguments
                    args = json.loads(tool_call.function["arguments"])

                    # Call the tool
                    response = await mcp_client.call_tool(
                        tool_call.function["name"],
                        args,
                    )

                    # Create tool call message
                    tool_call_message = ToolCallOutputMessage(
                        role="tool",
                        content=response.content,
                        tool_call_id=tool_call.id,
                    )

                    all_messages.append(tool_call_message)
                    yield AgentOutput("message", tool_call_message.model_dump())

                except Exception as error:
                    logger.error(
                        f"Tool call failed: {error}, tool: {tool_call.function['name']}"
                    )
                    # Re-raise tool execution errors as server errors
                    raise Exception(
                        f"Tool execution failed - tool: {tool_call.function['name']}, error: {error}"
                    )
        else:
            # No more tool calls, agent is done
            break

    # Transient timeouts the client absorbed. A run rescued by retries still
    # ran in a flaky environment, so the count travels with the result.
    run_usage["transient_tool_retries"] = getattr(
        mcp_client, "transient_retries", 0
    )
    yield AgentOutput("usage", run_usage)


async def handle_run_mcp_eval(
    body: RunAgentAPIRequestBody,
) -> AsyncGenerator[AgentOutput, None]:
    """
    Shared handler for running MCP eval that can be used by different routers.

    Args:
        body: Request body matching RunAgentAPIRequestBodySchema format

            Yields:
        AgentOutput: Generator that yields either successful messages or errors during MCP eval execution
    """
    mcp_client = None

    mcp_client = SandboxMCPClient(
        sandbox_url=config.MCP_SERVER_URL,
        enabled_tools=body.enabled_tools,
    )

    async for output in run_mcp_eval(
        mcp_client=mcp_client,
        model=body.model,
        messages=body.messages,
        max_turns=body.max_turns,
        extra_body=body.extra_body,
    ):
        yield output


async def run_dynamic_mcp_eval_request(
    body: RunDynamicAgentAPIRequestBody,
) -> Any:
    """Execute the separate P1-016 loop and retain only structural provenance."""

    mcp_client = SandboxMCPClient(
        sandbox_url=config.MCP_SERVER_URL,
        enabled_tools=None,
    )
    if body.selector_id is not None and body.tool_budget is not None:
        if body.selector_id in REGISTERED_P1_026_SELECTOR_IDS:
            # P1-026 conditions rank with hybrid_three_way, which needs models.
            # The ranker is built once per process and cached, so weights are
            # not reloaded per cycle or per run.
            #
            # p1_026_full exposes every tool and ranks nothing, so it must not
            # require the ranking models. Building them unconditionally made the
            # control condition fail in environments without
            # sentence-transformers -- contradicting the ranker's own error
            # message, which states that p1_026_full runs without it.
            needs_ranker = body.selector_id != "p1_026_full"
            selector = build_registered_p1_026_dynamic_selector(
                selector_id=body.selector_id,
                tool_budget=body.tool_budget,
                messages=body.messages,
                ranker=build_p1_026_ranker() if needs_ranker else None,
            )
        else:
            selector = build_registered_dynamic_selector(
                selector_id=body.selector_id,
                tool_budget=body.tool_budget,
            )
    else:
        selector = _ConfiguredDynamicSelector(body.active_tool_names)
    return await run_dynamic_mcp_eval(
        mcp_client=mcp_client,
        selector=selector,
        completion=create_completion,
        model=body.model,
        messages=body.messages,
        max_turns=body.max_turns,
        extra_body=body.extra_body,
    )


async def handle_run_dynamic_mcp_eval(
    body: RunDynamicAgentAPIRequestBody,
) -> AsyncGenerator[AgentOutput, None]:
    """Yield dynamic outputs in the same internal form as the fixed-list path."""
    result = await run_dynamic_mcp_eval_request(body)
    for output in result.outputs:
        yield AgentOutput(output["type"], output["data"])


def dynamic_safe_trace(result: Any) -> dict[str, Any]:
    """Return route-safe provenance without prompts, schemas, calls, or results."""
    return {
        "cycles": [asdict(cycle) for cycle in result.cycles],
        "model_final_text_present": result.final_text is not None,
    }
