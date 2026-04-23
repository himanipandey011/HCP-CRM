from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent

app = FastAPI(title="HCP CRM AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    current_form_state: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    reply: str
    form_updates: Optional[Dict[str, Any]] = None
    tool_used: Optional[str] = None
    suggestions: Optional[List[str]] = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatMessage):
    try:
        result = await run_agent(
            user_message=payload.message,
            conversation_history=payload.conversation_history,
            current_form_state=payload.current_form_state
        )
        return ChatResponse(
            reply=result["reply"],
            form_updates=result.get("form_updates"),
            tool_used=result.get("tool_used"),
            suggestions=result.get("suggestions")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "HCP CRM AI Backend is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)