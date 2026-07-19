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
    return await run_dynamic_mcp_eval(
        mcp_client=mcp_client,
        selector=_ConfiguredDynamicSelector(body.active_tool_names),
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
