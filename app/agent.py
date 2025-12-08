from langchain.chat_models import init_chat_model
from app.tools import search_download_tool,eda_report_tool
from langgraph.graph import StateGraph, START,END
from app.state import State
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage

def create_agent():
    llm = init_chat_model(model="groq:llama-3.3-70b-versatile")
    tools = [search_download_tool, eda_report_tool]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    memory = MemorySaver()

    def should_continue(state: State):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def agent_node(state: State):
        system_msg = SystemMessage(content="""You are a helpful assistant that can search and download Kaggle datasets and perform EDA.
        
        Rules:
        - First search and download the dataset using search_download_tool
        - Then create EDA report using eda_report_tool
        - After EDA report is created, DO NOT call any more tools. Simply respond that the task is complete.
        - Never repeat a tool call that has already succeeded.
            """)
        
        messages = [system_msg] + state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": response}

    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")
    
    return builder.compile(checkpointer=memory)