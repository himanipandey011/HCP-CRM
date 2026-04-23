import os
import json
import re
from datetime import date
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

def get_llm(model: str = "llama-3.3-70b-versatile"):
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=model,
        temperature=0.1,
    )

def clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw

@tool
def log_interaction(natural_language_description: str) -> str:
    """Extract HCP interaction details from natural language and return structured JSON to populate form fields."""
    llm = get_llm()
    today = date.today().isoformat()
    prompt = f"""You are a CRM data extraction assistant for Life Sciences field reps.
Extract interaction details from this description and return ONLY valid JSON.

Description: "{natural_language_description}"

Return JSON with these fields (use null for missing):
{{
  "hcp_name": "string or null",
  "interaction_type": "Meeting|Phone Call|Email|Conference|null",
  "date": "YYYY-MM-DD or null (today is {today})",
  "time": "HH:MM or null",
  "attendees": ["list of names"],
  "topics_discussed": "summary string or null",
  "sentiment": "Positive|Negative|Neutral or null",
  "materials_shared": ["list of materials"],
  "follow_up_required": true,
  "notes": "any additional notes or null"
}}
Return ONLY the JSON, no explanation."""
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = clean_json(response.content)
    try:
        data = json.loads(raw)
        return json.dumps({"status": "success", "extracted_fields": data, "tool": "log_interaction"})
    except:
        return json.dumps({"status": "error", "message": "Could not parse", "raw": raw})

@tool
def edit_interaction(edit_instruction: str) -> str:
    """Update only specific form fields based on user correction. Input is the edit instruction from the user."""
    llm = get_llm()
    today = date.today().isoformat()
    prompt = f"""You are a CRM form editor.

User's edit instruction: "{edit_instruction}"

Today's date is {today}.

Return ONLY valid JSON with just the fields that need to change.
Examples:
- "change date to today" → {{"date": "{today}"}}
- "fix name to Dr. Himani" → {{"hcp_name": "Dr. Himani"}}
- "change type to Phone Call" → {{"interaction_type": "Phone Call"}}

Return ONLY the JSON, no explanation."""
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = clean_json(response.content)
    try:
        updates = json.loads(raw)
        return json.dumps({"status": "success", "field_updates": updates, "tool": "edit_interaction"})
    except:
        return json.dumps({"status": "error", "message": "Could not parse edit", "raw": raw})

@tool
def suggest_followup(interaction_summary: str) -> str:
    """Suggest the next best follow-up actions for the field rep after an HCP interaction."""
    llm = get_llm()
    prompt = f"""You are a Life Sciences CRM advisor.

Interaction summary: {interaction_summary}

Give 3 specific follow-up suggestions as a JSON array of strings.
Return ONLY the JSON array, no explanation."""
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = clean_json(response.content)
    try:
        suggestions = json.loads(raw)
        return json.dumps({"status": "success", "suggestions": suggestions, "tool": "suggest_followup"})
    except:
        return json.dumps({"status": "success", "suggestions": ["Schedule follow-up meeting", "Send product literature", "Check sample needs"], "tool": "suggest_followup"})

@tool
def search_materials(topics: str) -> str:
    """Recommend relevant marketing materials or clinical studies to share with the HCP based on topics discussed."""
    llm = get_llm()
    prompt = f"""You are a Life Sciences materials advisor.

Topics discussed: {topics}

Recommend 4 relevant materials. Return ONLY a valid JSON array:
[{{"type": "Brochure", "name": "material name", "relevance": "why relevant"}}]
Return ONLY the JSON array."""
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = clean_json(response.content)
    try:
        materials = json.loads(raw)
        return json.dumps({"status": "success", "materials": materials, "tool": "search_materials"})
    except:
        return json.dumps({"status": "success", "materials": [{"type": "Brochure", "name": "Product Overview", "relevance": "General info"}], "tool": "search_materials"})

@tool
def sentiment_analyzer(meeting_notes: str) -> str:
    """Analyze sentiment and emotional signals from meeting notes to provide a detailed sentiment report."""
    llm = get_llm()
    prompt = f"""Analyze the sentiment of these HCP meeting notes.

Notes: "{meeting_notes}"

Return ONLY valid JSON:
{{
  "overall_sentiment": "Positive|Negative|Neutral|Mixed",
  "confidence_score": 0.85,
  "key_positive_signals": ["list"],
  "concerns_raised": ["list"],
  "prescribing_intent": "High|Medium|Low|Unknown"
}}
Return ONLY the JSON."""
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = clean_json(response.content)
    try:
        result = json.loads(raw)
        return json.dumps({"status": "success", "sentiment_report": result, "tool": "sentiment_analyzer"})
    except:
        return json.dumps({"status": "success", "sentiment_report": {"overall_sentiment": "Neutral", "confidence_score": 0.5}, "tool": "sentiment_analyzer"})