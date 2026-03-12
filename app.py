import pickle
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.messages import HumanMessage

from config import XGBOOST_MODEL_PATH, BERT_MODEL_PATH
from agent import build_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASD Detection AI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading  (cached so they load only once per Streamlit session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_xgboost_model():
    """Load XGBoost model. Returns (model, error_string)."""
    try:
        with open(XGBOOST_MODEL_PATH, "rb") as f:
            return pickle.load(f), None
    except FileNotFoundError:
        return None, f"XGBoost model not found at: {XGBOOST_MODEL_PATH}"
    except Exception as e:
        return None, f"Error loading XGBoost model: {str(e)}"


@st.cache_resource
def load_bert_model():
    """Load BERT tokenizer + model. Returns (tokenizer, model, error_string)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        model     = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, f"Error loading BERT model: {str(e)}"


@st.cache_resource
def load_agent(_xgb, _tok, _bert):
    """Build the LangGraph agent (cached after first build)."""
    return build_agent(_xgb, _tok, _bert)


# ── Load everything ───────────────────────────────────────────────────────────
xgboost_model,              xgb_error  = load_xgboost_model()
bert_tokenizer, bert_model, bert_error = load_bert_model()

models_ready = (xgboost_model is not None) and (bert_model is not None)

if models_ready:
    agents = load_agent(xgboost_model, bert_tokenizer, bert_model)
else:
    agents = None


# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    """Reset conversation to a clean slate."""
    st.session_state.messages    = []          # list of (role, text) for display
    st.session_state.agent_state = {           # LangGraph state dict
        "messages": [],
        "answer"  : "",
        "stage"   : "idle",
    }

if "messages" not in st.session_state:
    init_session()

    # Auto-trigger the opening greeting on first load
    if agents:
        st.session_state.agent_state = agents.invoke(st.session_state.agent_state)
        greeting = st.session_state.agent_state.get("answer", "")
        if greeting:
            st.session_state.messages.append(("assistant", greeting))


# ══════════════════════════════════════════════════════════════════════════════
# Core: send a message through the LangGraph agent
# ══════════════════════════════════════════════════════════════════════════════

def send_message(user_text: str):
    """
    Append user message to state, invoke the agent, store reply for display.
    """
    if not agents:
        st.session_state.messages.append(
            ("assistant", "❌ Models not loaded. Cannot process request.")
        )
        return

    # Add user message to both display list and LangGraph state
    st.session_state.messages.append(("user", user_text))
    st.session_state.agent_state["messages"].append(HumanMessage(content=user_text))

    # Run the graph
    st.session_state.agent_state = agents.invoke(st.session_state.agent_state)

    reply = st.session_state.agent_state.get("answer", "")
    if reply:
        st.session_state.messages.append(("assistant", reply))


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🧠 ASD Detection System")
        st.markdown("---")

        # Model status indicators
        st.subheader("Model Status")
        col1, col2 = st.columns(2)
        with col1:
            if xgboost_model:
                st.success("✅ XGBoost")
            else:
                st.error("❌ XGBoost")
                if xgb_error:
                    st.caption(xgb_error[:60])
        with col2:
            if bert_model:
                st.success("✅ BERT")
            else:
                st.error("❌ BERT")
                if bert_error:
                    st.caption(bert_error[:60])

        st.markdown("---")

        # Quick action buttons
        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Questionnaire", use_container_width=True):
                send_message("questionnaire")
                st.rerun()
        with col2:
            if st.button("📝 Text Analysis", use_container_width=True):
                send_message("text")
                st.rerun()

        if st.button("🔄 Clear Chat", type="secondary", use_container_width=True):
            init_session()
            # Re-trigger greeting after reset
            if agents:
                st.session_state.agent_state = agents.invoke(st.session_state.agent_state)
                greeting = st.session_state.agent_state.get("answer", "")
                if greeting:
                    st.session_state.messages.append(("assistant", greeting))
            st.rerun()

        st.markdown("---")

        # How to use
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Start** — the agent greets you automatically
            2. **Say yes** — to begin an assessment
            3. **Choose** — questionnaire or text method
            4. **Provide** — your answers or description
            5. **Get** — prediction with confidence scores
            6. **Continue** — run another or type *no* to exit
            """)


        st.markdown("---")
        st.warning("""
        ⚠️ **Medical Disclaimer**

        Screening tool only — not a diagnostic instrument.
        Always consult a qualified healthcare professional.
        """)

    # ── Main chat area ────────────────────────────────────────────────────────
    st.title("Autism Behavioral Trait Detection AI Agent")
    st.markdown("---")

    # Chat history display
    chat_container = st.container(height=500)
    with chat_container:
        for role, message in st.session_state.messages:
            with st.chat_message(role):
                st.markdown(message)

    # Chat input box
    if prompt := st.chat_input("Type your message here..."):
        with st.spinner("Thinking..."):
            send_message(prompt)
        st.rerun()

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption("Built with ❤️ using Streamlit | For screening purposes only")


if __name__ == "__main__":
    main()
