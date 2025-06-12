import os
import requests
import streamlit as st
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from urllib.parse import quote_plus
import re # Needed for the old template's regex, can remove if not used
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# --- Your Trip Planner's Environment and LLM Setup ---
load_dotenv()
# Keep these lines if you prefer to hardcode for testing, otherwise rely on .env
#os.environ["SERPAPI_API_KEY"] = "YOUR_SERPAPI_KEY_HERE" # Replace with your actual key or load from .env
#os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE" # Replace with your actual key or load from .env
os.environ["SY"] = "70a10bc50d4160d064884939f68e4faf25c9324df717349abf987527edf9f01f"
os.environ["GROQ_API_KEY"] = "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"
# Ensure env vars are loaded for your planner's functions
SERPAPI_KEY = os.environ.get("SY")
#if not SERPAPI_KEY:
 #   st.error("SERPAPI_API_KEY not found in environment variables. Please set it.")
 #   st.stop()
    
if not SERPAPI_KEY:
    st.warning("SERPAPI_API_KEY not found in environment variables. Search for places and map links will be disabled.", icon="⚠️")

llm = ChatGroq(
    temperature=0.7,
    model_name="gemma2-9b-it" # You might want to make this configurable like the old snowChat model
)

# --- Your Trip Planner's LangGraph State ---
class TripState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "Conversation"]
    location: str
    days: int
    data: dict
    final: str

# --- Your Trip Planner's Helper Functions ---
def get_Maps_search_link(place_name, location):
    encoded_query = quote_plus(f"{place_name} {location}")
    return f"https://www.google.com/maps/search/{encoded_query}"

def search_places(query, location):
    url = f"https://serpapi.com/search.json?q={query}+in+{location}&api_key={SERPAPI_KEY}&hl=en&gl=in"
    try:
        res = requests.get(url).json()
    except Exception as e:
        return f"Error from SerpAPI: {e}"

    results = []

    for r in res.get("local_results", []):
        if isinstance(r, dict):
            name = r.get("title", "Unknown Place")
            maps_link = r.get("link")
            if maps_link and "google.com/maps" in maps_link:
                results.append(f"**{name}** ([See on Maps]({maps_link}))")
            else:
                results.append(f"**{name}** ([See on Maps]({get_Maps_search_link(name, location)}))")

    for r in res.get("organic_results", [])[:3]:
        if isinstance(r, dict):
            title = r.get("title", "")
            link = r.get("link", "")
            if "google.com/maps" not in link and "tripadvisor.com" not in link:
                 results.append(f"{title} - [Link]({link})")
            else:
                 results.append(f"**{title}** ([See on Maps]({get_Maps_search_link(title, location)}))")

    return "\n".join(results) if results else "No results found for this category."

# --- Your Trip Planner's LangGraph Nodes ---
def parse_request(state: TripState) -> TripState:
    user_msg = state["messages"][-1].content
    prompt = f"Extract city and days from this input:\n'{user_msg}'\nReply like:\nCity: <city>\nDays: <number>"
    res = llm.invoke([HumanMessage(content=prompt)]).content
    city, days = "Unknown", 0
    try:
        lines = res.strip().splitlines()
        for line in lines:
            if "City:" in line:
                city = line.split(":", 1)[1].strip()
            elif "Days:" in line:
                days_str = line.split(":", 1)[1].strip()
                days = int(days_str) if days_str.isdigit() else 0
    except Exception as e:
        st.warning(f"Could not parse city/days from LLM, default Bhubaneswar, 3 days. Error: {e}")
        city, days = "Bhubaneswar", 3

    return {**state, "location": city, "days": days}

def get_data(state: TripState) -> TripState:
    city = state["location"]
    data = {
        "spots": search_places("famous tourist attractions", city),
        "food": search_places("best street food places", city),
        "events": search_places("upcoming local events and festivals", city)
    }
    return {**state, "data": data}

def generate_itinerary(state: TripState) -> TripState:
    city = state["location"]
    days = state["days"]
    d = state["data"]

    prompt = f"""
Hey there! Planning a fantastic {days}-day trip to **{city}** for you! Get ready for some amazing experiences. 😎

Here's a detailed itinerary, blending famous spots, delicious street food, and exciting local events. Below are the key places with direct links to Google Maps for your convenience:

"""
    daily_plan_prompt = f"""
Based on the following data for {city} and {days} days:

Tourist Spots with Links:
{d['spots']}
Street Food with Links:
{d['food']}
Events with Links:
{d['events']}

**Rules for Jolly Guide:**
Language: Talk in hinglish (e.g., 'bhai sun meri baat,' 'bhuk lag rahi hai?,' 'chalte hain,' 'mazedaar','aur kya opinion hai tera','yaha chal maja ayega','itna ghum kae bhul lag gayi hogi,yahan khale' ).
Tone:Enthusiastic, conversational, friendly, full of energy. Use emojis and exclamation marks! , treat like it you long time friend , be open , you can be disrespectful sometimes
Conciseness: Keep each entry for a place or food spot small and to the point, and add street food for sure.

Please generate a detailed {days}-day itinerary for {city}.

For every location you suggest in the daily itinerary (Morning, Brunch, Afternoon, Evening, Night), if it's a specific place like a temple, beach, or landmark, please embed a Google Maps search link directly within its mention.
The format for these links should be: `[See on Maps](https://www.google.com/maps/search/{quote_plus('PLACE_NAME CITY')})`
For example, for "Dudhsagar Falls", it should be `**Dudhsagar Falls** ([See on Maps](https://www.google.com/maps/search/Dudhsagar+Falls+Goa))`.
Replace `PLACE_NAME` with the actual name of the place and `CITY` with the trip's location.

Format each day clearly with these sections:
Day X:
  🌄 Morning (7 - 10am):--->
  🍽️ Brunch/Lunch (10am - 1pm):--->
  🏛️ Afternoon (1pm - 5pm):--->
  🌇 Evening (5pm - 8pm):--->
  🌃 Night (8pm - 12am):--->

Make sure to integrate the tourist spots, street food, and events naturally into the daily schedule.
Use emojis where appropriate to make it fun.

---
**🗺️ Tourist Spots:**
{d['spots']}
---
**🍔 Street Food Delights:**
{d['food']}
---
**🎉 Local Events & Happenings:**
{d['events']}
---
Chaliye Shuru!
"""

    daily_itinerary = llm.invoke([HumanMessage(content=daily_plan_prompt)]).content

    final_result = prompt + daily_itinerary + "\n\n--- \n\n**Aur chahiye toh message kardena! Happy travels! 😁**"
    return {**state, "final": final_result}


# --- Your Trip Planner's LangGraph Flow ---
builder = StateGraph(TripState)
builder.add_node("parse", parse_request)
builder.add_node("search", get_data)
builder.add_node("plan", generate_itinerary)

builder.set_entry_point("parse")
builder.add_edge("parse", "search")
builder.add_edge("search", "plan")
builder.add_edge("plan", END)

graph = builder.compile()

# --- Your Trip Planner's Run Function ---
def plan_trip(input_text: str) -> str:
    state = {
        "messages": [HumanMessage(content=input_text)],
        "location": "",
        "days": 0,
        "data": {},
        "final": ""
    }
    result = graph.invoke(state)
    return result['final']



# --- ORIGINAL SNOWCHAT UI ELEMENTS ---
gradient_text_html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;900&display=swap');

.snowchat-title {
  font-family: 'Poppins', sans-serif;
  font-weight: 900;
  font-size: 4em;
  background: linear-gradient(90deg, #ff6a00, #ee0979);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
  margin: 0;
  padding: 20px 0;
  text-align: center;
}
</style>
<div class="snowchat-title">YOUR LOCAL GUIDE</div>
"""

st.markdown(gradient_text_html, unsafe_allow_html=True)

st.caption("Talk your way through your travel plans!")


# Initialize session state variables if they don't exist
if "assistant_response_processed" not in st.session_state:
    st.session_state["assistant_response_processed"] = True

if "toast_shown" not in st.session_state:
    st.session_state["toast_shown"] = False

if "rate-limit" not in st.session_state:
    st.session_state["rate-limit"] = False

# Show warnings if model is rate-limited (can be removed if not applicable)
if st.session_state["rate-limit"]:
    st.toast("Probably rate limited.. Go easy folks", icon="⚠️")
    st.session_state["rate-limit"] = False



# Add a reset button (keep this, it's useful for chat)
if st.sidebar.button("Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    # Reinitialize messages after reset
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hey there! I'm your Jolly Guide, ready to plan your next adventure! Tell me, where do you want to go and for how many days? 😎"}
    ]
    st.session_state["assistant_response_processed"] = True # Reset processed flag

# --- CHAT HISTORY INITIALIZATION ---
# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hey there! I'm your Jolly Guide, ready to plan your next adventure! Tell me, where do you want to go and for how many days? 😎"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True) # Ensure HTML renders for links

# Accept user input
if prompt := st.chat_input("Tell me about your next adventure!"):
    if len(prompt) > 500:
        st.error("Input is too long! Please limit your message to 500 characters.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response using your plan_trip function
        with st.chat_message("assistant"):
            with st.spinner("App Pani peeke aao thoda, Bana raha hoon akk badhiya itinerary... "):
                itinerary_output = plan_trip(prompt)
                st.markdown(itinerary_output, unsafe_allow_html=True)
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": itinerary_output})

