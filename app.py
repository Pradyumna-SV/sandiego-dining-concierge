import streamlit as st
import time
from model_engine import RecSysEngine

# 1. Page Configuration (Must be the first line)
st.set_page_config(
    page_title="San Diego Dining Concierge",
    page_icon="🌮",
    layout="centered"
)

# 2. Sidebar for System Status
with st.sidebar:
    st.header("System Status")
    st.success("Model Engine: Online")
    st.info("Dataset: San Diego Restaurants (2021)")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 3. Title and Introduction
st.title("🌮 San Diego Dining Concierge")
st.markdown(
    """
    Welcome! I can help you find the perfect place to eat in San Diego.
    *   **Ask for predictions:** "Will I like The Taco Stand?"
    *   **Ask for recommendations:** "Where should I go for dinner?"
    *   **Ask for categories:** "Find me a sushi place."
    """
)

# 4. Load the Engine (Cached!)
# This is the most important part. It prevents the data from reloading
# every time you type a message.
@st.cache_resource
def load_engine():
    # This only runs once when the server starts
    return RecSysEngine()

# Show a spinner while loading (only happens on first run)
with st.spinner("Waking up the AI and loading review data..."):
    engine = load_engine()

# 5. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm ready to help. What are you craving today?"}
    ]

# 6. Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. Handle User Input
if prompt := st.chat_input("Type your question here..."):
    # A. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. Generate Response (The AI Logic)
    # We use a small sleep to simulate "thinking" so it feels natural
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # --- CALL YOUR ENGINE HERE ---
        response = engine.generate_response(prompt)
        
        # Simulate typing effect (optional, but looks cool)
        # time.sleep(0.5) 
        
        message_placeholder.markdown(response)

    # C. Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": response})