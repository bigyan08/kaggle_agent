# FastAPI app: start a chat, resume a paused run after human review, read saved results.
import json
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.agent import build_graph, find_run_id
from app.config import RECURSION_LIMIT, RUNS_DIR
from app.guardrails import GuardrailViolation, check_user_input, resolve_inside

app = FastAPI(title="Kaggle Agent API")
graph = build_graph()


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    action: Literal["approve", "edit", "reject"]
    args: dict | None = None  # for "edit": {tool_call_id: new_args}
    feedback: str | None = None  # for "reject"


class ChatResponse(BaseModel):
    thread_id: str
    status: str  # "completed" or "awaiting_approval"
    reply: str | None = None
    approval: dict | None = None  # what the human is being asked to review
    run_id: str | None = None  # latest saved training run in this thread, if any


def run_graph(graph_input, thread_id: str) -> ChatResponse:
    """Run (or resume) the graph, then report whether it finished or paused for approval."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    result = graph.invoke(graph_input, config=config)
    if "__interrupt__" in result:
        return ChatResponse(thread_id=thread_id, status="awaiting_approval", approval=result["__interrupt__"][0].value)
    reply = result["messages"][-1].content
    return ChatResponse(thread_id=thread_id, status="completed", reply=reply, run_id=find_run_id(result["messages"]))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        message = check_user_input(request.message)
    except GuardrailViolation as error:
        raise HTTPException(status_code=400, detail=str(error))
    return run_graph({"messages": [("user", message)]}, request.thread_id or str(uuid.uuid4()))


@app.post("/chat/{thread_id}/resume", response_model=ChatResponse)
def resume(thread_id: str, request: ResumeRequest):
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not state.next:
        raise HTTPException(status_code=404, detail="No paused run for this thread_id.")
    return run_graph(Command(resume=request.model_dump(exclude_none=True)), thread_id)


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    try:
        results_file = resolve_inside(RUNS_DIR, run_id) / "results.json"
    except GuardrailViolation as error:
        raise HTTPException(status_code=400, detail=str(error))
    if not results_file.is_file():
        raise HTTPException(status_code=404, detail="Run not found.")
    return json.loads(results_file.read_text())


@app.get("/health")
def health():
    return {"status": "healthy"}
