from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from utilities.generator import generate_chat_responses

load_dotenv()

model = ChatOpenAI(model="gpt-4.1")
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]
memory = MemorySaver()
llm_with_tools = model.bind_tools(tools=tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages": [result],
    }

async def tools_router(state: State):
    last_message = state["messages"][-1]
    
    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    
    return END
    

# The tool_node can either be this:
# tool_node = ToolNode(tools=tools) or the function below. 
# ToolNode is from langgraph.prebuilt import ToolNode
async def tool_node(state: State):
    """Custom tool node that handles tool calls from the LLM."""
    # Get the tool calls from the last messages
    tool_calls = state["messages"][-1].tool_calls

    # Initialize list to store tool messages
    tool_messages = []

    # Process each tool call
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"] # The search query lives here
        tool_id = tool_call["id"]

        # Handle the search tool
        if tool_name == "tavily_search_results_json":
            # Execute the search tool with the provided arguments
            search_results = await search_tool.ainvoke(tool_args)

            # Create a ToolMessage for this result
            tool_message = ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            )

            tool_messages.append(tool_message)
    
    # Add the tool messages to the state
    return {"messages": tool_messages}

graph_builder = StateGraph(State)
graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")

graph_builder.add_conditional_edges("model", tools_router, {
        "tool_node": "tool_node",
        "__end__": END
    })
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)

app = FastAPI()

# Add CORS middleware with settings that match frontend requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"]
)

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(graph, message, checkpoint_id), 
        media_type="text/event-stream"
    )

# SSE - server-sent events