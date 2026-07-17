"""
Pakistan Medicine Chatbot - FastAPI + Groq API
With LangSmith Tracking - 2 SEPARATE traceable functions
"""
 
import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
from dotenv import load_dotenv
from langsmith import traceable
from fastapi.responses import FileResponse

load_dotenv()
 
app = FastAPI(title="Pakistan's Healthcare Assistant", version="3.0.0")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")
 
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}
 
class Message(BaseModel):
    role: str
    content: str
 
class MedicineQuery(BaseModel):
    query: str
    context: Optional[List[Message]] = None
 
PHARMACY_SYSTEM_PROMPT = """You are a Pakistan Medicine Expert Pharmacist. Answer ONLY medicine, disease, symptom, and medical condition questions.

YOUR ROLE:
- Answer questions about medicines available in Pakistan
- Explain diseases, symptoms, and how medicines treat them
- Provide dosage, salt/ingredient, purpose, and OTC/Rx information
- Write in conversational 2-3 lines (like talking to a friend)
- Use context from previous messages
- Add appropriate disclaimers when giving medical advice

RESPONSE FORMAT:
Always return ONLY this JSON:
{"ans": "Not more than 2-3 line conversational answer with disclaimer if needed"}

WHAT TO INCLUDE IN ANSWER:
1. Medicine name (actual Pakistan brand if available)
2. Salt/ingredient (what it's made of)
3. Purpose (what disease/symptom it treats)
4. How it works (mechanism briefly)
5. Dosage (how to take it)
6. OTC or Prescription status in Pakistan

ANSWER STYLE:
- Conversational and friendly (like trusted pharmacist)
- Simple language (no heavy medical jargon)
- Practical and actionable advice
- Empathetic and supportive tone
- Restrict it to 2-3 lines maximum
- Include relevant information naturally in conversation

DISCLAIMER RULES:
ADD DISCLAIMER (on new line with \n\n spacing) WHEN:
✓ During suggesting/recommending any medicines
✓ Giving dosage instructions or treatment advice
✓ Discussing serious symptoms or conditions
✓ Mentioning prescription-only medicines
✓ Advising on medical precautions or side effects
✓ Patient is asking for treatment recommendations

DO NOT add disclaimer WHEN:
✗ Just explaining what a medicine is
✗ Describing salt/ingredient information only
✗ Answering basic medical knowledge questions
✗ Non-treatment related conversations
✗ Simple informational queries

DISCLAIMER FORMAT:
- Add on NEW LINE at end of answer
- Start with: "Disclaimer:"
- Keep SHORT and contextual (1 line)
- Use natural language

MEDICINE KNOWLEDGE:
You have knowledge of all Pakistan pharmaceutical brands including:
- Pain/Fever: Panadol, Calpol, Brufen, Disprol, Panadol Extra
- Antibiotics: Amoxil, Augmentin, Azomax, Ciprofloxacin, Cephalosporins
- Cold/Cough: Arinac, Gravinate, Bisolvon, Ascoril
- Digestive: Omeprazole, Flagyl, Antacids, Probiotics
- Respiratory: Salbutamol Inhaler, Seretide, Salmeterol
- Allergies: Rigix, Antihistamines, Decongestants
- Chronic: Metformin, Lisinopril, Amlodipine, Atorvastatin, Insulin
- Skin: Clotrimazole, Mupirocin, Hydrocortisone cream
- Other: Imodium, Nausea meds, Sleep aids, Antivirals, Antifungals

CRITICAL RULES:
1. ONLY answer medicine/disease/symptom/medical questions
2. If question is NOT medical (sports, politics, cooking, tech, entertainment, weather, etc) - REJECT it
3. For rejected questions, respond: {"ans": "I'm a medical chatbot specialized in medicines and diseases in Pakistan. Please ask about medicines, symptoms, diseases, treatments, or any medical concern."}
4. For emergencies (Heart attack, severe bleeding, unconscious, difficulty breathing, poisoning, overdose, severe trauma) - RESPOND: {"ans": "⚠️ EMERGENCY! Go to nearest hospital immediately or call 1122. This needs urgent medical care. Do NOT delay."}
5. Return ONLY "ans" field - NO other fields
6. NO hardcoded examples - generate answer based on actual question
7. Use context from previous messages to understand patient better
8. Be accurate - only recommend actual Pakistan available medicines
9. Decide automatically whether to add disclaimer based on question type

REMEMBER:
✓ Generic answers for ANY medicine/disease question
✓ NO hardcoded example answers
✓ Always check if question is medical-related first
✓ Conversational 2-3 lines only
✓ Include medicine name, salt, purpose, dosage, OTC/Rx status
✓ Add disclaimer ONLY when giving medical advice/recommendations
✓ Format disclaimer with \n\n before it
✓ Return ONLY {"ans": "answer with or without disclaimer"}
"""


# TRACEABLE FUNCTION 1: System Prompt
@traceable(name="system-prompt", run_type="prompt")
def track_system_prompt(system_prompt: str) -> str:
    """LangSmith tracks complete system prompt with all instructions"""
    return system_prompt

# TRACEABLE FUNCTION 2: User Message (context + question)
@traceable(name="user-message", run_type="prompt")
def track_user_message(user_message: str) -> str:
    """LangSmith tracks user message with context and current question"""
    return user_message

# TRACEABLE FUNCTION 3: Groq API Call
@traceable(name="groq-api-call", run_type="llm")
def call_groq_api(system_prompt: str, user_message: str) -> str:
    """Call Groq API - LangSmith tracks input and output separately"""
    
    # Track system prompt separately
    track_system_prompt(system_prompt)
    
    # Track user message separately
    track_user_message(user_message)
    
    payload = {
    "model": GROQ_MODEL,
    "messages": [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_message
        }
    ],
    "max_tokens": 400,
    "temperature": 0.1,
    "response_format": {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "ans": {
                    "type": "string",
                    "description": "Medicine recommendation answer"
                }
            },
            "required": ["ans"]
        }
    }
}
    
    response = requests.post(GROQ_API_URL, headers=GROQ_HEADERS, json=payload, timeout=20)
    result = response.json()
    return result["choices"][0]["message"]["content"]

@app.post("/api/medicine")
async def get_medicine_info(request: MedicineQuery):
    """
    Medicine chatbot endpoint
    Input: {"query": "any medicine/disease/symptom question"}
    Output: {"ans": "conversational answer about medicine, salt, disease, purpose"}
    """
    
    if not request.query or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Build context string from previous messages
        context_str = ""
        if request.context and len(request.context) > 0:
            # Keep last 15 messages for context
            recent_context = request.context[-15:] if len(request.context) > 15 else request.context
            context_str = "Previous conversation:\n"
            for msg in recent_context:
                context_str += f"{msg.role.upper()}: {msg.content}\n"
            context_str += "\n"
        
        # Build user message (context + current question)
        user_message = f"""{context_str}Current question: {request.query}

RESPOND ONLY WITH VALID JSON (no markdown, no extra text):
{{"ans": "your answer"}}"""
        
        # Call the 3 traceable functions
        # Function 1: Track system prompt
        # Function 2: Track user message
        # Function 3: Make API call (calls 1 and 2 internally)
        try:
            response_text = call_groq_api(PHARMACY_SYSTEM_PROMPT, user_message)
            
        except requests.exceptions.Timeout:
            return {"ans": "Request timeout - please try again"}
        except requests.exceptions.RequestException as e:
            return {"ans": f"API Error: {str(e)}"}
        
        if not response_text:
            return {"ans": "No response received"}
        
        # Parse response to get 'ans' field
        response_json = json.loads(response_text)

        return response_json
        
    except Exception as e:
        return {"ans": f"Error: {str(e)}"}
 
@app.get("/")
async def serve_chat_ui():
    return FileResponse("app.html")

@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "name": "Pakistan Medicine Chatbot",
        "version": "3.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    