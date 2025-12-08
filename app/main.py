from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY', '')
os.environ['LANGSMITH_TRACING'] = "true"

from .agent import create_agent

app = FastAPI(title="Kaggle Agent API")
graph = create_agent()

class QueryRequest(BaseModel):
    message: str
    thread_id: str | None = None

class QueryResponse(BaseModel):
    response: str
    thread_id: str

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    result = graph.invoke({"messages": request.message}, config=config)
    last_message = result["messages"][-1].content
    
    return QueryResponse(response=last_message, thread_id=thread_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}