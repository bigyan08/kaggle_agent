from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeLLM(GenericFakeChatModel):
    """Replays scripted AIMessages in order, so tests need no network or API key."""

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def fake_llm(*messages: AIMessage) -> FakeLLM:
    return FakeLLM(messages=iter(messages))
