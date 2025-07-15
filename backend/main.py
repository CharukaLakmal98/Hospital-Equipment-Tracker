from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Hospital Equipment Tracker", version="1.0.0")

# Enable CORS for your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM using environment variables from .env file
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

try:
    if USE_LLM and OPENAI_API_KEY:
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE
        )
        print(f"✅ LLM initialized with model: {OPENAI_MODEL}")
    else:
        llm = None
        if USE_LLM:
            print("⚠️ LLM enabled but OPENAI_API_KEY not found in .env file")
        else:
            print("ℹ️ LLM disabled - using rule-based responses")
except Exception as e:
    llm = None
    print(f"❌ Failed to initialize LLM: {e}")

# Define the state for our LangGraph agent


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query_type: str
    equipment_data: Dict[str, Any]
    next_action: str


# Sample equipment database (in a real app, this would be a proper database)
EQUIPMENT_DB = {
    "ventilators": {
        "total": 25,
        "available": 18,
        "in_use": 7,
        "maintenance": 0,
        "locations": ["ICU", "Emergency", "Surgery"]
    },
    "monitors": {
        "total": 50,
        "available": 35,
        "in_use": 12,
        "maintenance": 3,
        "locations": ["ICU", "Emergency", "General Ward"]
    },
    "defibrillators": {
        "total": 15,
        "available": 12,
        "in_use": 2,
        "maintenance": 1,
        "locations": ["Emergency", "Surgery", "ICU"]
    }
}

# Agent Functions


def analyze_query(state: AgentState) -> AgentState:
    """Analyze the user query to determine what type of equipment information they need"""
    last_message = state["messages"][-1]
    query = last_message.content.lower()

    if any(equipment in query for equipment in EQUIPMENT_DB.keys()):
        state["query_type"] = "equipment_lookup"
        state["next_action"] = "lookup_equipment"
    elif "status" in query or "available" in query or "how many" in query:
        state["query_type"] = "status_check"
        state["next_action"] = "check_status"
    elif "location" in query or "where" in query:
        state["query_type"] = "location_check"
        state["next_action"] = "check_location"
    else:
        state["query_type"] = "general"
        state["next_action"] = "general_response"

    return state


def lookup_equipment(state: AgentState) -> AgentState:
    """Look up specific equipment information"""
    last_message = state["messages"][-1]
    query = last_message.content.lower()

    response = "Here's what I found:\n\n"

    for equipment_type, data in EQUIPMENT_DB.items():
        if equipment_type in query:
            response += f"**{equipment_type.title()}:**\n"
            response += f"• Total units: {data['total']}\n"
            response += f"• Available: {data['available']}\n"
            response += f"• Currently in use: {data['in_use']}\n"
            response += f"• Under maintenance: {data['maintenance']}\n"
            response += f"• Locations: {', '.join(data['locations'])}\n\n"
            state["equipment_data"] = data
            break
    else:
        response = f"I couldn't find specific information about that equipment. Available equipment types: {', '.join(EQUIPMENT_DB.keys())}"

    state["messages"].append(AIMessage(content=response))
    state["next_action"] = "end"
    return state


def check_status(state: AgentState) -> AgentState:
    """Provide overall equipment status"""
    response = "**Hospital Equipment Status Overview:**\n\n"

    for equipment_type, data in EQUIPMENT_DB.items():
        availability_rate = (data['available'] / data['total']) * 100
        response += f"**{equipment_type.title()}:** {data['available']}/{data['total']} available ({availability_rate:.1f}%)\n"

    state["messages"].append(AIMessage(content=response))
    state["next_action"] = "end"
    return state


def check_location(state: AgentState) -> AgentState:
    """Provide location-based equipment information"""
    last_message = state["messages"][-1]
    query = last_message.content.lower()

    response = "**Equipment by Location:**\n\n"

    # Collect all unique locations
    all_locations = set()
    for data in EQUIPMENT_DB.values():
        all_locations.update(data['locations'])

    for location in sorted(all_locations):
        if location.lower() in query or "all" in query:
            response += f"**{location}:**\n"
            for equipment_type, data in EQUIPMENT_DB.items():
                if location in data['locations']:
                    response += f"• {equipment_type.title()}: Available\n"
            response += "\n"

    if response == "**Equipment by Location:**\n\n":
        response = f"Available locations: {', '.join(sorted(all_locations))}"

    state["messages"].append(AIMessage(content=response))
    state["next_action"] = "end"
    return state


def general_response(state: AgentState) -> AgentState:
    """Handle general queries - with optional LLM enhancement"""

    if llm:
        # AI-powered response using LLM
        try:
            last_message = state["messages"][-1]
            user_query = last_message.content

            system_prompt = f"""You are a helpful hospital equipment tracking assistant. 
            
Available equipment in our database:
{EQUIPMENT_DB}

Please provide a helpful response to the user's query about hospital equipment. 
Be concise, friendly, and professional. If they ask about specific equipment, 
refer to the data above. If you can't find the information, guide them on what they can ask about.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            ai_response = llm.invoke(messages)
            response = ai_response.content

        except Exception as e:
            response = f"AI temporarily unavailable. Error: {str(e)}\n\n" + \
                get_fallback_response()
    else:
        # Fallback to rule-based response
        response = get_fallback_response()

    state["messages"].append(AIMessage(content=response))
    state["next_action"] = "end"
    return state


def get_fallback_response() -> str:
    """Fallback response when LLM is not available"""
    return """I'm your Hospital Equipment Tracker assistant! I can help you with:

🏥 **Equipment Information:**
• Check availability of ventilators, monitors, defibrillators
• View equipment status and maintenance info
• Find equipment locations

📍 **Location Services:**
• Find equipment in specific departments (ICU, Emergency, Surgery, General Ward)
• Check what's available in each location

📊 **Status Reports:**
• Overall equipment availability
• Usage statistics
• Maintenance schedules

Try asking me things like:
• "How many ventilators are available?"
• "Show me equipment in the ICU"
• "What's the status of all equipment?"
• "Where can I find monitors?"
"""

# Create the LangGraph workflow


def create_equipment_agent():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("lookup", lookup_equipment)
    workflow.add_node("status", check_status)
    workflow.add_node("location", check_location)
    workflow.add_node("general", general_response)

    # Set entry point
    workflow.set_entry_point("analyze")

    # Add conditional edges
    def route_action(state: AgentState) -> str:
        return state["next_action"]

    workflow.add_conditional_edges(
        "analyze",
        route_action,
        {
            "lookup_equipment": "lookup",
            "check_status": "status",
            "check_location": "location",
            "general_response": "general"
        }
    )

    # Add edges to END
    workflow.add_edge("lookup", END)
    workflow.add_edge("status", END)
    workflow.add_edge("location", END)
    workflow.add_edge("general", END)

    return workflow.compile()


# Initialize the agent
equipment_agent = create_equipment_agent()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {"message": "Hospital Equipment Tracker API with LangGraph Agent"}


@app.post("/ask-agent/", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    """Process equipment queries using LangGraph agent"""
    try:
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "query_type": "",
            "equipment_data": {},
            "next_action": ""
        }

        # Run the agent
        result = equipment_agent.invoke(initial_state)

        # Get the last AI message
        ai_messages = [msg for msg in result["messages"]
                       if isinstance(msg, AIMessage)]
        if ai_messages:
            response_text = ai_messages[-1].content
        else:
            response_text = "I'm sorry, I couldn't process your request. Please try again."

        return QueryResponse(response=response_text)

    except Exception as e:
        return QueryResponse(response=f"Sorry, I encountered an error: {str(e)}")


@app.get("/equipment/status")
async def get_equipment_status():
    """Get overall equipment status"""
    return {"equipment_database": EQUIPMENT_DB}


@app.get("/equipment/{equipment_type}")
async def get_specific_equipment(equipment_type: str):
    """Get information about specific equipment type"""
    if equipment_type.lower() in EQUIPMENT_DB:
        return {equipment_type: EQUIPMENT_DB[equipment_type.lower()]}
    else:
        return {"error": f"Equipment type '{equipment_type}' not found", "available_types": list(EQUIPMENT_DB.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
