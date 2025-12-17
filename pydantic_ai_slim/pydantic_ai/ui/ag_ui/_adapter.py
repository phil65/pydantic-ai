"""AG-UI adapter for handling requests."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from ... import ExternalToolset, ToolDefinition
from ...messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from ...toolsets import AbstractToolset

try:
    from ag_ui.core import (
        AssistantMessage,
        BaseEvent,
        BinaryInputContent,
        DeveloperMessage,
        Message,
        RunAgentInput,
        SystemMessage,
        TextInputContent,
        Tool as AGUITool,
        ToolMessage,
        UserMessage,
    )

    from .. import MessagesBuilder, UIAdapter, UIEventStream
    from ._event_stream import BUILTIN_TOOL_CALL_ID_PREFIX, AGUIEventStream
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

if TYPE_CHECKING:
    from ...messages import UserContent

__all__ = ['AGUIAdapter']


def _convert_agui_content(content: str | list[TextInputContent | BinaryInputContent]) -> str | list[UserContent]:
    """Convert AG-UI message content to pydantic-ai UserContent format.

    AG-UI UserMessage.content can be:
    - str: plain text (pass through)
    - list[TextInputContent | BinaryInputContent]: structured content

    This converts AG-UI content types to pydantic-ai equivalents:
    - TextInputContent -> str
    - BinaryInputContent with URL -> ImageUrl/AudioUrl/VideoUrl/DocumentUrl based on mime_type
    - BinaryInputContent with data -> BinaryContent

    Args:
        content: AG-UI message content

    Returns:
        Content in pydantic-ai format
    """
    if isinstance(content, str):
        return content

    result: list[UserContent] = []
    for item in content:
        if isinstance(item, TextInputContent):
            result.append(item.text)
        elif isinstance(item, BinaryInputContent):
            if item.url:
                # Convert URL-based content to appropriate pydantic-ai URL type based on mime_type
                if item.mime_type.startswith('image/'):
                    result.append(ImageUrl(url=item.url, media_type=item.mime_type))
                elif item.mime_type.startswith('audio/'):
                    result.append(AudioUrl(url=item.url, media_type=item.mime_type))
                elif item.mime_type.startswith('video/'):
                    result.append(VideoUrl(url=item.url, media_type=item.mime_type))
                else:
                    # Default to DocumentUrl for other types (PDF, text, etc.)
                    result.append(DocumentUrl(url=item.url, media_type=item.mime_type))
            elif item.data:
                # data is base64 encoded
                result.append(BinaryContent(data=base64.b64decode(item.data), media_type=item.mime_type))
        else:
            # Unknown type - try to convert to string
            result.append(str(item))

    return result


# Frontend toolset


class _AGUIFrontendToolset(ExternalToolset[AgentDepsT]):
    """Toolset for AG-UI frontend tools."""

    def __init__(self, tools: list[AGUITool]):
        """Initialize the toolset with AG-UI tools.

        Args:
            tools: List of AG-UI tool definitions.
        """
        super().__init__(
            [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.parameters,
                )
                for tool in tools
            ]
        )

    @property
    def label(self) -> str:
        """Return the label for this toolset."""
        return 'the AG-UI frontend tools'  # pragma: no cover


class AGUIAdapter(UIAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT, OutputDataT]):
    """UI adapter for the Agent-User Interaction (AG-UI) protocol."""

    @classmethod
    def build_run_input(cls, body: bytes) -> RunAgentInput:
        """Build an AG-UI run input object from the request body."""
        return RunAgentInput.model_validate_json(body)

    def build_event_stream(self) -> UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]:
        """Build an AG-UI event stream transformer."""
        return AGUIEventStream(self.run_input, accept=self.accept)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the AG-UI run input."""
        return self.load_messages(self.run_input.messages)

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Toolset representing frontend tools from the AG-UI run input."""
        if self.run_input.tools:
            return _AGUIFrontendToolset[AgentDepsT](self.run_input.tools)
        return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Frontend state from the AG-UI run input."""
        state = self.run_input.state
        if state is None:
            return None

        if isinstance(state, Mapping) and not state:
            return None

        return cast('dict[str, Any]', state)

    @classmethod
    def load_messages(cls, messages: Sequence[Message]) -> list[ModelMessage]:
        """Transform AG-UI messages into Pydantic AI messages."""
        builder = MessagesBuilder()
        tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.

        for msg in messages:
            if isinstance(msg, UserMessage | SystemMessage | DeveloperMessage) or (
                isinstance(msg, ToolMessage) and not msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
            ):
                if isinstance(msg, UserMessage):
                    content = _convert_agui_content(msg.content)
                    builder.add(UserPromptPart(content=content))
                elif isinstance(msg, SystemMessage | DeveloperMessage):
                    builder.add(SystemPromptPart(content=msg.content))
                else:
                    tool_call_id = msg.tool_call_id
                    tool_name = tool_calls.get(tool_call_id)
                    if tool_name is None:  # pragma: no cover
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')

                    builder.add(
                        ToolReturnPart(
                            tool_name=tool_name,
                            content=msg.content,
                            tool_call_id=tool_call_id,
                        )
                    )

            elif isinstance(msg, AssistantMessage) or (  # pragma: no branch
                isinstance(msg, ToolMessage) and msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
            ):
                if isinstance(msg, AssistantMessage):
                    if msg.content:
                        builder.add(TextPart(content=msg.content))

                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_call_id = tool_call.id
                            tool_name = tool_call.function.name
                            tool_calls[tool_call_id] = tool_name

                            if tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
                                _, provider_name, tool_call_id = tool_call_id.split('|', 2)
                                builder.add(
                                    BuiltinToolCallPart(
                                        tool_name=tool_name,
                                        args=tool_call.function.arguments,
                                        tool_call_id=tool_call_id,
                                        provider_name=provider_name,
                                    )
                                )
                            else:
                                builder.add(
                                    ToolCallPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        args=tool_call.function.arguments,
                                    )
                                )
                else:
                    tool_call_id = msg.tool_call_id
                    tool_name = tool_calls.get(tool_call_id)
                    if tool_name is None:  # pragma: no cover
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')
                    _, provider_name, tool_call_id = tool_call_id.split('|', 2)

                    builder.add(
                        BuiltinToolReturnPart(
                            tool_name=tool_name,
                            content=msg.content,
                            tool_call_id=tool_call_id,
                            provider_name=provider_name,
                        )
                    )

        return builder.messages
