# The data that flows through the graph.
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    # add_messages is a reducer: updates are appended (or replace a message with the same id).
    messages: Annotated[list[AnyMessage], add_messages]
