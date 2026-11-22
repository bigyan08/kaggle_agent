# Tests the human-in-the-loop flow with a fake LLM and a fake Kaggle client.
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app import kaggle_client, tools
from app.agent import build_graph
from tests.fakes import fake_llm, tool_call

CONFIG = {"configurable": {"thread_id": "t1"}}


@pytest.fixture
def downloads(monkeypatch, tmp_path) -> list:
    """Replace Kaggle with a fake; returns the list of refs that were 'downloaded'."""
    refs = []
    monkeypatch.setattr(tools, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(kaggle_client, "search", lambda query: [])
    monkeypatch.setattr(kaggle_client, "find", lambda ref: SimpleNamespace(ref=ref, total_bytes=1000))
    monkeypatch.setattr(kaggle_client, "download", lambda ref, folder: refs.append(ref))
    return refs


def start(*llm_messages):
    graph = build_graph(llm=fake_llm(*llm_messages))
    result = graph.invoke({"messages": [("user", "find iris data")]}, CONFIG)
    return graph, result


def test_download_pauses_for_approval_then_runs(downloads):
    graph, result = start(tool_call("download_dataset", {"dataset_ref": "o/iris"}), AIMessage("done"))
    assert result["__interrupt__"][0].value["tool_calls"][0]["name"] == "download_dataset"
    assert downloads == []  # nothing ran yet

    result = graph.invoke(Command(resume={"action": "approve"}), CONFIG)
    assert downloads == ["o/iris"]
    assert result["messages"][-1].content == "done"


def test_edit_changes_the_arguments_that_run(downloads):
    graph, result = start(tool_call("download_dataset", {"dataset_ref": "o/iris"}), AIMessage("done"))
    call_id = result["__interrupt__"][0].value["tool_calls"][0]["id"]

    graph.invoke(Command(resume={"action": "edit", "args": {call_id: {"dataset_ref": "o/other"}}}), CONFIG)
    assert downloads == ["o/other"]


def test_reject_skips_the_tool_and_tells_the_llm_why(downloads):
    graph, _ = start(tool_call("download_dataset", {"dataset_ref": "o/iris"}), AIMessage("ok, what instead?"))

    result = graph.invoke(Command(resume={"action": "reject", "feedback": "too big"}), CONFIG)
    assert downloads == []
    replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "too big" in replies[0].content


def test_training_also_needs_approval(downloads):
    args = {"dataset_ref": "o/iris", "file_name": "iris.csv", "target_column": "species"}
    _, result = start(tool_call("train_baselines", args), AIMessage("done"))
    assert result["__interrupt__"][0].value["tool_calls"][0]["args"]["target_column"] == "species"


def test_search_does_not_need_approval(downloads):
    _, result = start(tool_call("search_datasets", {"query": "iris"}), AIMessage("done"))
    assert "__interrupt__" not in result
    assert result["messages"][-1].content == "done"
