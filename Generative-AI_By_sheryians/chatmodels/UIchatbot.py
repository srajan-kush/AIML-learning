import streamlit as st
from dotenv import load_dotenv
import time

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Srajan AI Agent",
    page_icon="🤡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d0d0d;
    color: #f0f0f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111;
    border-right: 1px solid #2a2a2a;
}

/* Chat messages */
.chat-msg {
    display: flex;
    gap: 12px;
    margin: 12px 0;
    align-items: flex-start;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.chat-bubble {
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.55;
    font-family: 'Space Mono', monospace;
}

.user-bubble {
    background: #1e1e2e;
    border: 1px solid #3b3b5c;
    color: #c0caf5;
    margin-left: auto;
    border-top-right-radius: 4px;
}

.bot-bubble {
    background: #1a1a1a;
    border: 1px solid #ff6b35;
    color: #f0f0f0;
    border-top-left-radius: 4px;
}

.avatar {
    font-size: 1.5rem;
    min-width: 36px;
    text-align: center;
    margin-top: 4px;
}

/* Input area */
.stTextInput > div > div > input {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #f0f0f0 !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 2px rgba(255,107,53,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: #ff6b35 !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(255,107,53,0.35) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #1a1a1a !important;
    border-color: #333 !important;
    color: #f0f0f0 !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    padding-top: 4px;
}

/* Divider */
hr { border-color: #222; }

/* Header */
.page-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.page-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(90deg, #ff6b35, #f7c59f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.page-header p {
    color: #555;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin-top: 4px;
}

/* Token badge */
.badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #333;
    color: #888;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []          # display messages
    if "lc_messages" not in st.session_state:   # langchain messages
        st.session_state.lc_messages = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "you are a funny AI agent"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.9
    if "model_name" not in st.session_state:
        st.session_state.model_name = "mistral-small-2506"
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # Model selection
    model_choice = st.selectbox(
        "Model",
        ["mistral-small-2506", "mistral-medium-2505", "mistral-large-2411"],
        index=["mistral-small-2506", "mistral-medium-2505", "mistral-large-2411"].index(
            st.session_state.model_name
        ),
    )
    if model_choice != st.session_state.model_name:
        st.session_state.model_name = model_choice

    # Temperature
    temperature = st.slider("Temperature 🌡️", 0.0, 1.0, st.session_state.temperature, 0.05)
    if temperature != st.session_state.temperature:
        st.session_state.temperature = temperature

    # System prompt
    st.markdown("**System Prompt**")
    new_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=100,
        label_visibility="collapsed",
    )
    if new_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = new_prompt
        st.session_state.lc_messages = []  # reset history on prompt change

    st.divider()

    # Stats
    st.markdown(f"**💬 Messages:** {len(st.session_state.messages)}")
    st.markdown(f"**🔢 Est. tokens:** {st.session_state.total_tokens}")

    st.divider()

    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.lc_messages = []
            st.session_state.total_tokens = 0
            st.rerun()
    with col2:
        # Export chat
        if st.session_state.messages:
            chat_text = "\n\n".join(
                f"{'You' if m['role'] == 'user' else 'Bot'}: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "⬇️ Export",
                data=chat_text,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("⬇️ Export", disabled=True, use_container_width=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🤡 Funny AI Agent</h1>
    <p>powered by mistral · langchain</p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── Chat Display ──────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown(
            "<p style='text-align:center;color:#444;font-family:Space Mono,monospace;font-size:0.85rem;'>"
            "Say something... I dare you. 😈</p>",
            unsafe_allow_html=True,
        )
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-msg" style="justify-content:flex-end">
                <div class="chat-bubble user-bubble">{msg['content']}</div>
                <div class="avatar">🧑</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-msg">
                <div class="avatar">🤡</div>
                <div class="chat-bubble bot-bubble">{msg['content']}</div>
            </div>""", unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────────────────────
st.divider()
with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Type your message...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

if submitted and user_input.strip():
    # Build lc_messages fresh each time (system + history)
    lc_msgs = [SystemMessage(content=st.session_state.system_prompt)]
    
    lc_msgs += st.session_state.lc_messages
    lc_msgs.append(HumanMessage(content=user_input))

    # Save user message to display
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.lc_messages.append(HumanMessage(content=user_input))

    print("User: ",user_input)

    # Call model
    model = ChatMistralAI(
        model=st.session_state.model_name,
        temperature=st.session_state.temperature,
    )

    with st.spinner("Thinking... 🤔"):
        response = model.invoke(lc_msgs)

    bot_reply = response.content
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    print("Bot:",bot_reply)

    st.session_state.lc_messages.append(AIMessage(content=bot_reply))

    # Rough token estimate
    st.session_state.total_tokens += len(user_input.split()) + len(bot_reply.split())

    st.rerun()

