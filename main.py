import os
import base64
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import googlemaps
from openai import OpenAI

# --- CONFIGURATION ---
# We get these from Render's "Environment Variables" later
app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
gmaps = googlemaps.Client(key=os.environ.get("GOOGLE_MAPS_API_KEY"))

# ALLOW EVERYTHING (So your Netlify app can talk to this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class LocationRequest(BaseModel):
    lat: float
    lng: float

class ChatRequest(BaseModel):
    user_message: str
    menu_data: Dict[str, Any]
    history: List[Dict[str, str]]
    restaurant_name: str
    current_order: List[str]

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "System Operational", "president": "Clinton"}

@app.post("/identify-restaurant")
def identify_restaurant(loc: LocationRequest):
    print(f"Finding restaurant at {loc.lat}, {loc.lng}")
    try:
        places_result = gmaps.places_nearby(
            location=(loc.lat, loc.lng),
            radius=50,
            type='restaurant'
        )
        if not places_result['results']:
            return {"name": "Unknown Location", "address": "N/A"}
        
        place = places_result['results'][0]
        return {
            "name": place.get('name'),
            "address": place.get('vicinity')
        }
    except Exception as e:
        print(e)
        return {"name": "Unknown Location (Error)", "address": "N/A"}

@app.post("/scan-menu")
async def scan_menu(file: UploadFile = File(...)):
    print(f"Analyzing image...")
    image_data = await file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    prompt = """
    Extract menu items from this image into JSON:
    {"drinks": [], "appetizers": [], "entrees": [], "sides": []}
    Include price if visible.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(e)
        return {"drinks": ["Coke", "Water"], "entrees": ["Burger", "Salad"]} # Fallback

@app.post("/chat")
def chat_with_bill(request: ChatRequest):
    # 1. Tools
    tools = [{
        "type": "function",
        "function": {
            "name": "update_order",
            "description": "Add or remove item from order",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                    "action": {"type": "string", "enum": ["add", "remove"]}
                },
                "required": ["item_name", "action"]
            }
        }
    }]

    # 2. System Prompt
    current_order_str = ", ".join(request.current_order) if request.current_order else "Nothing yet"
    system_prompt = f"""
    You are Bill Clinton dining at {request.restaurant_name}.
    Current Tab: {current_order_str}.
    Menu Data: {json.dumps(request.menu_data)}
    
    Persona: Raspy voice, southern charm, loves unhealthy food but pretends to diet.
    Rule: Keep responses SHORT (under 2 sentences).
    If user wants to order, call the tool.
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(request.history[-6:])
    messages.append({"role": "user", "content": request.user_message})

    # 3. AI Call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    msg = response.choices[0].message
    final_text = msg.content
    updated_order = list(request.current_order)

    # 4. Handle Tool Use
    if msg.tool_calls:
        for tool in msg.tool_calls:
            args = json.loads(tool.function.arguments)
            if args["action"] == "add": updated_order.append(args["item_name"])
            if args["action"] == "remove" and args["item_name"] in updated_order: updated_order.remove(args["item_name"])
        
        # Get verbal confirmation
        messages.append(msg)
        messages.append({"role": "tool", "tool_call_id": msg.tool_calls[0].id, "content": "Order Updated."})
        resp2 = client.chat.completions.create(model="gpt-4o", messages=messages)
        final_text = resp2.choices[0].message.content

    return {"text": final_text, "updated_order": updated_order}
