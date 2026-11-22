# Runs the real agent (real LLM + Kaggle) on every task in tasks.json, auto-approving each human review,
# then prints a results table and saves evals/results.json. Usage: uv run python -m evals.run_evals [max_tasks]
import json
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.agent import build_graph, find_run_id
from app.config import RECURSION_LIMIT, RUNS_DIR

TASKS_FILE = Path(__file__).parent / "tasks.json"
RESULTS_FILE = Path(__file__).parent / "results.json"


def score_run(messages: list, expected_task_type: str) -> dict:
    """Judge one finished conversation using the run that train_baselines saved."""
    steps = sum(isinstance(message, ToolMessage) for message in messages)
    run_id = find_run_id(messages)
    if run_id is None:  # the agent never trained a model
        return {"finished": False, "task_type_ok": False, "beats_dummy": False, "steps": steps}

    results = json.loads((RUNS_DIR / run_id / "results.json").read_text())["results"]
    scores = {result["model"]: result["score"] for result in results}
    return {
        "finished": True,
        "task_type_ok": results[0]["task_type"] == expected_task_type,
        "beats_dummy": max(scores.values()) > scores["dummy"],
        "steps": steps,
    }


def run_task(graph, task: dict) -> dict:
    """Run one task, approving every pause, and return its scores plus timing."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}
    started = time.time()
    row = {"prompt": task["prompt"][:60], "error": ""}
    try:
        result = graph.invoke({"messages": [("user", task["prompt"])]}, config)
        while "__interrupt__" in result:
            result = graph.invoke(Command(resume={"action": "approve"}), config)
        row.update(score_run(result["messages"], task["expected_task_type"]))
    except Exception as error:  # one failing task must not stop the whole eval
        row.update(finished=False, task_type_ok=False, beats_dummy=False, steps=0, error=str(error)[:80])
    row["seconds"] = round(time.time() - started, 1)
    return row


def summarise(rows: list[dict]) -> dict:
    """Overall rates and averages across all tasks."""
    table = pd.DataFrame(rows)
    return {
        "tasks": len(table),
        "finished_rate": round(table["finished"].mean(), 2),
        "task_type_ok_rate": round(table["task_type_ok"].mean(), 2),
        "beats_dummy_rate": round(table["beats_dummy"].mean(), 2),
        "avg_steps": round(table["steps"].mean(), 1),
        "avg_seconds": round(table["seconds"].mean(), 1),
    }


if __name__ == "__main__":
    tasks = json.loads(TASKS_FILE.read_text())
    if len(sys.argv) > 1:
        tasks = tasks[: int(sys.argv[1])]

    graph = build_graph()
    rows = []
    for task in tasks:
        rows.append(run_task(graph, task))
        print(f"[{len(rows)}/{len(tasks)}] {rows[-1]['prompt']}  finished={rows[-1]['finished']}", flush=True)

    summary = summarise(rows)
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print("\n" + json.dumps(summary, indent=2))
    RESULTS_FILE.write_text(json.dumps({"summary": summary, "tasks": rows}, indent=2))
