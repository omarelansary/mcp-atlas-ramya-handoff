"""LLM completion functionality using LiteLLM."""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
import litellm
from pydantic import BaseModel

from .schema import Message, ToolCallSchema, AssistantMessage
from .config import config

logger = logging.getLogger(__name__)

# Configure LiteLLM - suppress verbose logging
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class LLMResponse(BaseModel):
    """Response from LLM completion."""

    message: AssistantMessage
    original_content: Optional[str] = None
    # Token accounting, carried so a caller can measure what a run actually
    # cost rather than projecting it. Optional: not every provider returns it.
    usage: Optional[Dict[str, Any]] = None


def configure_litellm():
    litellm.api_base = config.LLM_BASE_URL  # could also be just openai url
    litellm.api_key = config.LLM_API_KEY


# Configure LiteLLM once at module level
configure_litellm()


def strip_all_additional_properties(schema: any) -> any:
    """Recursively remove all `additionalProperties` keys from the schema."""
    if isinstance(schema, dict):
        # Remove 'additionalProperties' if it exists
        schema.pop("additionalProperties", None)

        # Recurse into all values
        for key, value in schema.items():
            strip_all_additional_properties(value)

    elif isinstance(schema, list):
        for item in schema:
            strip_all_additional_properties(item)

    return schema


async def create_completion(
    model: str,
    messages: List[Message],
    tools: List[ToolCallSchema],
    extra_body: Optional[Dict[str, Any]] = None,
) -> LLMResponse:
    """Create a completion using LiteLLM."""

    # Convert our schema to LiteLLM form at
    if "gemini" in model.lower():
        litellm_messages = [
            (
                msg.model_dump()
                if not isinstance(msg, AssistantMessage)
                else msg.original_message.model_dump()
            )
            for msg in messages
        ]
        litellm_tools = [
            strip_all_additional_properties(tool.model_dump()) for tool in tools
        ]
    else:
        litellm_messages = [msg.model_dump() for msg in messages]
        litellm_tools = [tool.model_dump() for tool in tools]

    # These specific models route through an internal proxy that expects the
    # "openai/" prefix in the model name. LiteLLM strips one "openai/" prefix
    # when a custom api_base is set, so we double-prepend it here so the proxy
    # receives the correct name (e.g. "openai/macaroni-alpha").
    _PROXY_PREFIX_MODELS = ("openai/macaroni-alpha", "openai/galapagos-alpha")
    if config.LLM_BASE_URL and model in _PROXY_PREFIX_MODELS:
        proxy_model = "openai/" + model
    else:
        proxy_model = model

    try:
        response = await litellm.acompletion(
            model=proxy_model,
            messages=litellm_messages,
            tools=litellm_tools,
            api_key=config.LLM_API_KEY,
            api_base=config.LLM_BASE_URL,
            timeout=config.DEFAULT_TIMEOUT,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        # Convert response back to our format
        # Handle tool_calls conversion from OpenAI format to our format
        tool_calls = None
        if response.choices[0].message.tool_calls:
            tool_calls = []
            for tool_call in response.choices[0].message.tool_calls:
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )

        assistant_message = AssistantMessage(
            role="assistant",
            content=response.choices[0].message.content,
            tool_calls=tool_calls,
            original_message=response.choices[0].message,
        )

        # Reasoning and cached-prefix counts are the two quantities a cost
        # projection cannot guess, so pull them out where the provider reports
        # them. Absent fields stay absent rather than defaulting to zero, which
        # would read as "measured and none" instead of "not reported".
        usage = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", None),
                "completion_tokens": getattr(raw_usage, "completion_tokens", None),
                "total_tokens": getattr(raw_usage, "total_tokens", None),
            }
            details = getattr(raw_usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", None)
            if reasoning is not None:
                usage["reasoning_tokens"] = reasoning
            details = getattr(raw_usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None)
            if cached is not None:
                usage["cached_prompt_tokens"] = cached

        return LLMResponse(message=assistant_message, usage=usage)

    except Exception as error:
        logger.error(f"LiteLLM completion failed: {error}")
        raise


def _transform_tool_calls(tools: List[Dict[str, Any]]) -> List[ToolCallSchema]:
    """Transform tool definitions to ToolCallSchema format."""
    return [
        ToolCallSchema(
            type="function",
            function={
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("input_schema", {}),
                "strict": False,
            },
        )
        for tool in tools
    ]
