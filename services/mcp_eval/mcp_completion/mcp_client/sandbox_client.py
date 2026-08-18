"""Sandbox MCP client implementation."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from .base_client import MCPClient
from ..errors import MCPClientToolExecutionError, MCPClientToolTimeoutError
from ..schema import ToolDefinition, CallToolResponse, TextContent
from ..config import config

logger = logging.getLogger(__name__)


class SandboxMCPClient(MCPClient):
    """MCP client that connects to pre-running sandbox environments."""

    def __init__(
        self,
        sandbox_url: str,
        enabled_tools: Optional[List[str]] = None,  # if None, all tools are enabled
    ):
        self.sandbox_url = sandbox_url
        self.enabled_tools = enabled_tools
        self.tool_call_timeout = config.TOOL_CALL_TIMEOUT
        self.list_tools_timeout = config.LIST_TOOLS_TIMEOUT
        # How many transient timeouts were absorbed. Reported rather than
        # hidden: retries that rescue a run are still evidence the environment
        # was flaky, and a run needing many of them is not a clean one.
        self.transient_retries = 0

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the sandbox."""
        try:
            tools_data = await self.list_raw_tools()
            tools = [ToolDefinition(**tool) for tool in tools_data]

            # Filter by enabled tools if specified.
            if self.enabled_tools:
                tools = [tool for tool in tools if tool.name in self.enabled_tools]

            return tools

        except Exception as error:
            logger.error(f"Failed to list tools from sandbox: {error}")
            raise

    async def list_raw_tools(self) -> List[Dict[str, Any]]:
        """Return the unprojected `/list-tools` response for dynamic adapters."""
        async with httpx.AsyncClient(timeout=self.list_tools_timeout) as client:
            response = await client.post(
                f"{self.sandbox_url}/list-tools",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            tools_data = response.json()
        if not isinstance(tools_data, list) or not all(
            isinstance(tool, dict) for tool in tools_data
        ):
            raise ValueError("sandbox /list-tools response must be a list of objects")
        return tools_data

    async def call_tool(self, tool_name: str, args: Any) -> CallToolResponse:
        """Call a tool in the sandbox, retrying transient timeouts.

        A timeout is not evidence about the model or the tool selection policy
        -- it means a remote service was slow for a moment -- yet without a
        retry it aborts the whole task and scores as a failure. Upstream
        MCP-Atlas added retry handling for transient tool errors in April 2026;
        this fork predates that, and a 5-row probe on 2026-08-17 lost a row to a
        60s `wikipedia_get_summary` timeout.

        The timeout escalates per attempt rather than repeating the same short
        window, since the failure mode being retried is slowness.

        Deliberately NOT retried: a non-200 response, which is returned to the
        model as an error result so it can react, and any other exception, which
        is not known to be transient.
        """
        attempts = max(1, config.TOOL_CALL_ATTEMPTS)
        body = {"tool_name": tool_name, "tool_args": args}
        last_timeout = self.tool_call_timeout

        for attempt in range(1, attempts + 1):
            last_timeout = self.tool_call_timeout * attempt
            try:
                async with httpx.AsyncClient(timeout=last_timeout) as client:
                    response = await client.post(
                        f"{self.sandbox_url}/call-tool",
                        json=body,
                        headers={"Content-Type": "application/json"},
                    )

                    if response.status_code != 200:
                        error_text = response.text
                        return CallToolResponse(
                            content=[TextContent(type="text", text=error_text)],
                            is_error=True,
                        )

                    response_data = response.json()
                    return CallToolResponse(
                        content=response_data,
                        is_error=False,
                    )

            except httpx.TimeoutException:
                self.transient_retries += 1
                if attempt < attempts:
                    logger.warning(
                        f"Tool {tool_name} timed out after {last_timeout}s "
                        f"(attempt {attempt}/{attempts}); retrying with "
                        f"{self.tool_call_timeout * (attempt + 1)}s"
                    )
                    continue
                logger.error(
                    f"Tool {tool_name} timed out after {last_timeout}s on the "
                    f"final attempt ({attempts}/{attempts})"
                )
                raise MCPClientToolTimeoutError(
                    f"Tool {tool_name} timed out after {last_timeout}s "
                    f"across {attempts} attempts"
                )
            except Exception as error:
                logger.error(f"Failed to call tool {tool_name} in sandbox: {error}")
                raise MCPClientToolExecutionError(
                    f"Failed to call tool {tool_name}: {error}"
                )

    @property
    def sandbox_info(self) -> Dict[str, Any]:
        """Get sandbox information."""
        return {
            "sandbox_url": self.sandbox_url,
        }
