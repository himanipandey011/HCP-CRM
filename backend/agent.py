import os
import json
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from tools import log_interaction, edit_interaction, suggest_followup, search_materials, sentiment_analyzer

load_dotenv()

# ── State ──────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    form_state: Dict[str, Any]
    tool_used: Optional[str]
    form_updates: Optional[Dict[str, Any]]
    suggestions: Optional[List[str]]
    final_reply: Optional[str]

# ── Tools list ─────────────────────────────────────────
tools = [log_interaction, edit_interaction, suggest_followup, search_materials, sentiment_analyzer]

# ── LLM with tools bound ───────────────────────────────
def get_llm_with_tools():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )
    return llm.bind_tools(tools)

# ── Nodes ──────────────────────────────────────────────
def agent_node(state: AgentState) -> AgentState:
    llm = get_llm_with_tools()
    system = SystemMessage(content="""You are an AI assistant for a Life Sciences CRM system helping field representatives log HCP (Healthcare Professional) interactions.

You have access to these tools:
1. log_interaction - Extract details from natural language and populate the form
2. edit_interaction - Update specific form fields based on user corrections
3. suggest_followup - Suggest next best actions after an interaction
4. search_materials - Recommend relevant materials to share with HCPs
5. sentiment_analyzer - Analyze sentiment from meeting notes

RULES:
- When user describes a meeting or interaction → use log_interaction
- When user says "change", "fix", "update", "correct" → use edit_interaction
- When user asks for follow-up suggestions → use suggest_followup
- When user asks about materials or brochures → use search_materials
- When user asks about sentiment or tone → use sentiment_analyzer
- Always be helpful, concise, and professional.
- After using a tool, explain what you did in a friendly way.""")

    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

def tool_node_func(state: AgentState) -> AgentState:
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)
    return result

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def process_tool_results(state: AgentState) -> AgentState:
    """Parse tool results and extract form updates."""
    form_updates = None
    tool_used = None
    suggestions = None

    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            try:
                data = json.loads(msg.content)
                tool_used = data.get("tool")

                if tool_used == "log_interaction" and data.get("status") == "success":
                    form_updates = data.get("extracted_fields", {})

                elif tool_used == "edit_interaction" and data.get("status") == "success":
                    form_updates = data.get("field_updates", {})

                elif tool_used == "suggest_followup" and data.get("status") == "success":
                    suggestions = data.get("suggestions", [])

                elif tool_used == "search_materials" and data.get("status") == "success":
                    materials = data.get("materials", [])
                    suggestions = [f"{m['type']}: {m['name']}" for m in materials]

                elif tool_used == "sentiment_analyzer" and data.get("status") == "success":
                    report = data.get("sentiment_report", {})
                    suggestions = [
                        f"Sentiment: {report.get('overall_sentiment', 'N/A')}",
                        f"Prescribing Intent: {report.get('prescribing_intent', 'N/A')}",
                        f"Confidence: {int(float(report.get('confidence_score', 0)) * 100)}%"
                    ]
                    if form_updates is None:
                        form_updates = {"sentiment": report.get("overall_sentiment")}
                break
            except:
                continue

    return {
        "tool_used": tool_used,
        "form_updates": form_updates,
        "suggestions": suggestions
    }

def final_reply_node(state: AgentState) -> AgentState:
    """Generate a friendly final reply after tool use."""
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    tool_used = state.get("tool_used")
    form_updates = state.get("form_updates")
    suggestions = state.get("suggestions")

    context = f"Tool used: {tool_used}\n"
    if form_updates:
        context += f"Form fields updated: {list(form_updates.keys())}\n"
    if suggestions:
        context += f"Suggestions generated: {suggestions}\n"

    system = SystemMessage(content="""You are a friendly CRM AI assistant. 
Write a brief, natural reply (2-3 sentences) confirming what you just did.
Be specific about what fields were filled or what was analyzed.
End with a helpful offer or question.""")

    response = llm.invoke([
        system,
        HumanMessage(content=f"Context of what just happened:\n{context}\nWrite a friendly confirmation message.")
    ])

    return {"final_reply": response.content}

# ── Build Graph ────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_func)
    graph.add_node("process_results", process_tool_results)
    graph.add_node("final_reply", final_reply_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        END: END
    })

    graph.add_edge("tools", "process_results")
    graph.add_edge("process_results", "final_reply")
    graph.add_edge("final_reply", END)

    return graph.compile()

# ── Main run function ──────────────────────────────────
async def run_agent(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    current_form_state: Dict[str, Any]
) -> Dict[str, Any]:

    app = build_graph()

    messages = []
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))

    initial_state = AgentState(
        messages=messages,
        form_state=current_form_state,
        tool_used=None,
        form_updates=None,
        suggestions=None,
        final_reply=None
    )

    result = await app.ainvoke(initial_state)

    reply = result.get("final_reply")
    if not reply:
        last = result["messages"][-1]
        reply = last.content if hasattr(last, "content") else "Done!"

    return {
        "reply": reply,
        "form_updates": result.get("form_updates"),
        "tool_used": result.get("tool_used"),
        "suggestions": result.get("suggestions")
    }

def get_interaction_state():
    return {}