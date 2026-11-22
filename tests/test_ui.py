# Runs the Streamlit script headlessly with a fake API, so no server or LLM is needed.
from pathlib import Path

from streamlit.testing.v1 import AppTest

UI_SCRIPT = str(Path(__file__).parent.parent / "ui" / "streamlit_app.py")

PAUSED = {
    "thread_id": "t1",
    "status": "awaiting_approval",
    "reply": None,
    "run_id": None,
    "approval": {"tool_calls": [{"id": "c1", "name": "download_dataset", "args": {"dataset_ref": "o/iris"}}]},
}
DONE = {"thread_id": "t1", "status": "completed", "reply": "All done", "approval": None, "run_id": "run1"}
RUN = {"dataset": "o/iris", "target": "species", "best_model": "random_forest", "results": [{"model": "dummy", "score": 0.1}]}


class FakeResponse:
    def __init__(self, body: dict):
        self.status_code = 200
        self.body = body

    def json(self) -> dict:
        return self.body


def make_app(monkeypatch, sent: list) -> AppTest:
    """Fake API: /chat pauses for approval, /resume completes, /runs returns saved metrics."""

    def fake_post(url, json=None, timeout=None):
        sent.append((url.rsplit("/", 1)[-1], json))
        return FakeResponse(PAUSED if url.endswith("/chat") else DONE)

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", lambda url, timeout=None: FakeResponse(RUN))
    return AppTest.from_file(UI_SCRIPT, default_timeout=10).run()


def test_approval_panel_then_results(monkeypatch):
    sent = []
    app = make_app(monkeypatch, sent)

    app.chat_input[0].set_value("find iris data").run()
    assert any("download_dataset" in m.value for m in app.markdown)
    assert app.chat_input[0].disabled  # can't chat while an approval is pending

    approve = next(b for b in app.button if b.label == "Approve")
    approve.click().run()
    assert sent[-1] == ("resume", {"action": "approve"})
    assert app.chat_message[-1].markdown[0].value == "All done"
    assert app.dataframe  # metrics table is shown


def test_edit_sends_changed_args(monkeypatch):
    sent = []
    app = make_app(monkeypatch, sent)
    app.chat_input[0].set_value("find iris data").run()

    app.text_area[0].set_value('{"dataset_ref": "o/other"}')
    next(b for b in app.button if b.label == "Edit & approve").click().run()
    assert sent[-1] == ("resume", {"action": "edit", "args": {"c1": {"dataset_ref": "o/other"}}})


def test_invalid_json_is_not_sent(monkeypatch):
    sent = []
    app = make_app(monkeypatch, sent)
    app.chat_input[0].set_value("find iris data").run()

    app.text_area[0].set_value("{not json")
    next(b for b in app.button if b.label == "Edit & approve").click().run()
    assert app.error
    assert len(sent) == 1  # only the first /chat call
