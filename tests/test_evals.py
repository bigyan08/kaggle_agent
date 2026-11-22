import json

from langchain_core.messages import AIMessage, ToolMessage

from evals import run_evals


def save_fake_run(runs_dir, task_type: str, scores: dict) -> str:
    """Write a results.json like ml.save_run does, and return its run id."""
    results = [{"model": name, "task_type": task_type, "score": score} for name, score in scores.items()]
    (runs_dir / "run1").mkdir()
    (runs_dir / "run1" / "results.json").write_text(json.dumps({"results": results}))
    return "run1"


def conversation(run_id: str | None) -> list:
    messages = [ToolMessage(content="Downloaded", tool_call_id="1", name="download_dataset")]
    if run_id:
        messages.append(ToolMessage(content=f"Best model: x. Run id: {run_id}", tool_call_id="2", name="train_baselines"))
    return messages + [AIMessage("done")]


def test_good_run_scores_all_true(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evals, "RUNS_DIR", tmp_path)
    run_id = save_fake_run(tmp_path, "classification", {"dummy": 0.2, "random_forest": 0.9})
    score = run_evals.score_run(conversation(run_id), "classification")
    assert score == {"finished": True, "task_type_ok": True, "beats_dummy": True, "steps": 2}


def test_wrong_task_type_and_no_improvement_are_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evals, "RUNS_DIR", tmp_path)
    run_id = save_fake_run(tmp_path, "classification", {"dummy": 0.5, "random_forest": 0.5})
    score = run_evals.score_run(conversation(run_id), "regression")
    assert not score["task_type_ok"]
    assert not score["beats_dummy"]


def test_run_that_never_trained_is_unfinished():
    score = run_evals.score_run(conversation(None), "classification")
    assert score["finished"] is False


def test_summary_rates():
    rows = [
        {"finished": True, "task_type_ok": True, "beats_dummy": True, "steps": 4, "seconds": 10.0},
        {"finished": False, "task_type_ok": False, "beats_dummy": False, "steps": 2, "seconds": 20.0},
    ]
    summary = run_evals.summarise(rows)
    assert summary["finished_rate"] == 0.5
    assert summary["avg_seconds"] == 15.0
