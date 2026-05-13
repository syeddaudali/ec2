"""
Pakistan Medicine Chatbot - FastAPI + Google Gemini API
GENERIC: No Hardcoded Answers - Handles ANY Medicine/Disease Question
Using Google's genai library instead of Groq
"""
 
from http import client
import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from dotenv import load_dotenv
 
# STEP 1: Load environment variables (API key must be in .env as GEMINI_API_KEY)
load_dotenv()
 
# STEP 2: Initialize FastAPI
app = FastAPI(title="Pakistan Medicine Chatbot", version="4.0.0")
 
# STEP 3: Add CORS middleware (same as before)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# STEP 4: Initialize Google Gemini Client (CHANGED FROM GROQ)
try:
    genai_client = genai.Client()  # Gets GEMINI_API_KEY from environment
    print("✅ Google Gemini API client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Gemini client: {e}")
    raise ValueError("Failed to initialize Google Gemini API")
 
# STEP 5: Pydantic Models (SAME AS BEFORE)
class Message(BaseModel):
    role: str
    content: str
 
class MedicineQuery(BaseModel):
    query: str
    context: Optional[List[Message]] = None

# GENERIC SYSTEM PROMPT - NO HARDCODED ANSWERS
PHARMACY_SYSTEM_PROMPT = """You are a Pakistan Medicine Expert Pharmacist. Answer ONLY medicine, disease, symptom, and medical condition questions.

YOUR ROLE:
- Answer questions about medicines available in Pakistan
- Explain diseases, symptoms, and how medicines treat them
- Provide dosage, salt/ingredient, purpose, and OTC/Rx information
- Write in conversational 3-4 lines (like talking to a friend)
- Use context from previous messages
- Add appropriate disclaimers when giving medical advice

RESPONSE FORMAT:
Always return ONLY this JSON:
{"ans": "3-4 line conversational answer with disclaimer if needed"}

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
- 3-4 lines maximum
- Include relevant information naturally in conversation

DISCLAIMER RULES:
ADD DISCLAIMER (on new line with \n\n spacing) WHEN:
✓ Suggesting/recommending specific medicines
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
4. For emergencies (Heart attack, severe bleeding, unconscious, difficulty breathing, poisoning, overdose, severe trauma) - RESPOND: {"ans": "⚠️ EMERGENCY! Go to hospital or call 1122 immediately. This needs urgent medical care. Do NOT delay."}
5. Return ONLY "ans" field - NO other fields
6. NO hardcoded examples - generate answer based on actual question
7. Use context from previous messages to understand patient better
8. Be accurate - only recommend actual Pakistan available medicines
9. Decide automatically whether to add disclaimer based on question type

REMEMBER:
✓ Generic answers for ANY medicine/disease question
✓ NO hardcoded example answers
✓ Always check if question is medical-related first
✓ Conversational 3-4 lines only
✓ Include medicine name, salt, purpose, dosage, OTC/Rx status
✓ Add disclaimer ONLY when giving medical advice/recommendations
✓ Format disclaimer with \n\n before it
✓ Return ONLY {"ans": "answer with or without disclaimer"}
"""

 
# STEP 7: Parse response function (SLIGHTLY CHANGED FOR GEMINI)
def parse_gemini_response(response_text: str) -> str:
    """Parse Gemini response and extract 'ans' field"""
    try:
        response_text = str(response_text).strip()
        
        # Remove markdown code blocks
        response_text = re.sub(r'```json\n?', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)
        response_text = response_text.strip()
        
        # Find and parse JSON
        json_match = re.search(r'\{[^{}]*"ans"[^{}]*\}', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            if "ans" in parsed:
                return str(parsed["ans"]).strip()
        
        # Fallback: try to extract any JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if "ans" in parsed:
                return str(parsed["ans"]).strip()
            return str(list(parsed.values())[0]).strip() if parsed else response_text
        
        # If no JSON found, return text as is
        return response_text
        
    except Exception:
        return response_text
 
# STEP 8: Main endpoint (CHANGED TO USE GEMINI)

from langsmith import traceable

@traceable(name="medicine-query", run_type="llm")
def call_gemini_api(full_prompt: str) -> str:
    """Call Gemini API - tracked with input prompt and output ans"""
    response = genai_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=full_prompt
    )
    return response.text


# In your async endpoint
@app.post("/api/medicine")
async def get_medicine_info(request: MedicineQuery):
    """Medicine chatbot endpoint"""
    
    if not request.query or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Build context
        context_str = ""
        if request.context and len(request.context) > 0:
            recent_context = request.context[-5:] if len(request.context) > 5 else request.context
            context_str = "Previous conversation:\n"
            for msg in recent_context:
                context_str += f"{msg.role.upper()}: {msg.content}\n"
            context_str += "\n"
        
        # Build prompt
        full_prompt = f"""{PHARMACY_SYSTEM_PROMPT}

{context_str}Current question: {request.query}

RESPOND ONLY WITH VALID JSON (no markdown, no extra text):
{{"ans": "your answer"}}"""
        
        # CALL TRACEABLE FUNCTION (input = full_prompt, output = response_text)
        response_text = call_gemini_api(full_prompt)
        
        if not response_text:
            return {"ans": "No response received from API"}
        
        # Parse to get ans field
        ans = parse_gemini_response(response_text)
        
        return {"ans": ans}
        
    except Exception as e:
        return {"ans": f"Error: {str(e)}"}
    
    
# STEP 12: Health check endpoint (SAME AS BEFORE)
@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "running",
        "name": "Pakistan Medicine Chatbot (Gemini)",
        "version": "4.0.0",
        "model": "Google Gemini 2.0 Flash"
    }
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 