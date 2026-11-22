# The LangGraph agent: an LLM that calls tools, with a human approval step before sensitive ones.
#
#   START -> agent --(no tool calls)--> END
#              |
#              +--(tool calls)--> human_review --(approve/edit)--> tools -> agent
#                                      |
#                                      +--(reject)--> agent
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from app.config import AGENT_MODEL, SENSITIVE_TOOLS
from app.state import State
from app.tools import TOOLS

SYSTEM_PROMPT = """You are a data assistant that finds Kaggle datasets and trains baseline ML models.

Steps:
1. Call search_datasets and pick the most relevant candidate.
2. Call download_dataset with its exact ref. A human approves, edits or rejects this.
3. Call inspect_file on the most relevant tabular file.
4. Pick a target column that fits the user's goal, then call train_baselines.
   A human approves or edits the target. Use task_type "auto" unless the goal is clear.
5. Reply with a short summary: dataset, file, target, metrics, best model, and whether it beats the dummy. Then stop.

Rules:
- If a call is rejected, read the feedback and adjust, or ask the user what they want.
- Only help with Kaggle datasets and baseline modelling. Politely decline anything else.
- Use only refs, file names and column names returned by tools. Never invent them.
- If a tool returns an error, explain it and try one sensible alternative at most.
- Never repeat a tool call that already succeeded.
"""

def find_run_id(messages: list) -> str | None:
    """Read the run id out of the latest train_baselines result (deterministic, no LLM involved)."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "train_baselines" and "Run id: " in message.content:
            return message.content.split("Run id: ")[1].strip()
    return None


def build_graph(llm=None, checkpointer=None):
    """Build the graph. llm and checkpointer are arguments only so tests can pass fakes."""
    llm = llm or init_chat_model(AGENT_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent(state: State) -> dict:
        response = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    def route_after_agent(state: State) -> Literal["human_review", "__end__"]:
        return "human_review" if state["messages"][-1].tool_calls else END

    def human_review(state: State) -> Command[Literal["tools", "agent"]]:
        # On resume this node re-runs from the top, so nothing before interrupt() may have side effects.
        last = state["messages"][-1]
        sensitive = [call for call in last.tool_calls if call["name"] in SENSITIVE_TOOLS]
        if not sensitive:
            return Command(goto="tools")

        decision = interrupt({"tool_calls": sensitive})

        if decision["action"] == "approve":
            return Command(goto="tools")

        if decision["action"] == "edit":
            # Reusing the message id makes add_messages replace the original call.
            edited_calls = [{**call, "args": decision.get("args", {}).get(call["id"], call["args"])} for call in last.tool_calls]
            edited = AIMessage(content=last.content, tool_calls=edited_calls, id=last.id)
            return Command(goto="tools", update={"messages": [edited]})

        # Rejected: every tool call needs a reply, or the next LLM call fails.
        feedback = decision.get("feedback", "No reason given.")
        replies = [
            ToolMessage(content=f"Human rejected this. Feedback: {feedback}", tool_call_id=call["id"])
            for call in last.tool_calls
        ]
        return Command(goto="agent", update={"messages": replies})

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("human_review", human_review)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer or MemorySaver())
