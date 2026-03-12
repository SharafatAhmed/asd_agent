"""
agent.py
--------
LangGraph-based ASD screening agent.

Nodes
-----
  supervisor_agent    – reads stage, calls LLM for routing/reply
  questionnaire_agent – handles Q-CHAT-10 flow + XGBoost prediction
  text_agent          – handles free-text flow + BERT prediction

Improvements over v1
---------------------
  • Confidence scores on both models (full probability breakdown)
  • Input relevance gate (LLM yes/no filter before BERT runs)
  • Post-result flow (ask user if they want another assessment)
  • Robust JSON parsing with regex fallback
"""

import re
import json
import numpy as np
import torch
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import groq_api_key


# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)


# ── State ─────────────────────────────────────────────────────────────────────
# Shared memory passed between every node in the graph.
#
# stage values:
#   idle                 → greeting shown, waiting for yes/no
#   choose_method        → user said yes, waiting for method choice
#   awaiting_answers     → questionnaire shown, waiting for 11 values
#   awaiting_description → text prompt shown, waiting for description
#   post_result          → prediction shown, asking if user wants another

class State(TypedDict):
    messages : Annotated[list, add_messages]   # full conversation history
    answer   : str                              # latest reply shown to user
    stage    : str                              # current conversation stage


# ── Feature labels (same order the XGBoost model was trained on) ──────────────
QCHAT_LABELS = ['A9', 'A6', 'A5', 'A7', 'A4', 'A1', 'A2', 'A8', 'A3', 'A10', 'Sex']


# ══════════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ══════════════════════════════════════════════════════════════════════════════

def questionnaire_predict(xgboost_model, features: list) -> str:
    """
    Run the XGBoost model on 11 Q-CHAT-10 answers.
    Returns a formatted result string with confidence scores.

    Args:
        xgboost_model : loaded XGBoost model object
        features      : list of 11 binary ints
    """
    proba      = xgboost_model.predict_proba(np.array([features]))[0]
    pred       = int(np.argmax(proba))
    label      = "ASD" if pred == 1 else "Non-ASD"
    confidence = proba[pred] * 100

    return (
        f"✅ **Questionnaire Analysis Complete**\n\n"
        f"**Assessment  :** {label}\n\n"
        f"**Confidence  :** {confidence:.1f}%\n\n"
        f"⚠️ **Disclaimer:** This is a screening tool only, not a medical diagnosis.\n"
        f"Always consult a qualified healthcare professional."
    )


def text_predict(bert_tokenizer, bert_model, description: str) -> str:
    """
    Run the BERT model on a free-text behaviour description.
    Returns a formatted result string with confidence scores.

    Args:
        bert_tokenizer : loaded HuggingFace tokenizer
        bert_model     : loaded BERT classification model
        description    : natural language string
    """
    inputs = bert_tokenizer(
        description,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        proba  = torch.softmax(logits, dim=1)[0]
        pred   = torch.argmax(proba).item()

    label      = "ASD" if pred == 1 else "Non-ASD"
    confidence = proba[pred].item() * 100
    preview    = description[:200] + ("..." if len(description) > 200 else "")

    return (
        f"✅ **Text Analysis Complete**\n\n"
        f"**Description :** \"{preview}\"\n\n"
        f"**Assessment  :** {label}\n\n"
        f"**Confidence  :** {confidence:.1f}%\n\n"
        f"⚠️ **Disclaimer:** This is a screening tool only, not a medical diagnosis.\n"
        f"Always consult a qualified healthcare professional."
    )


def parse_answers(raw: str):
    """
    Validate and parse comma-separated Q-CHAT-10 answers.

    Returns:
        list of 11 ints  → success
        str              → error message
    """
    try:
        values = [int(x.strip()) for x in raw.split(",")]
    except ValueError:
        return "Please use only 0 or 1 separated by commas.\nExample: 0,1,0,1,1,0,0,0,1,0,0"

    if len(values) != 11:
        return f"I need exactly 11 values but got {len(values)}. Please re-enter all 11 answers."

    bad = [v for v in values if v not in (0, 1)]
    if bad:
        return f"All values must be 0 or 1. Found invalid values: {bad}. Please re-enter."

    return values


def is_valid_description(text: str) -> bool:
    """
    Input relevance gate — asks the LLM if the text is a child behaviour description.
    Prevents irrelevant inputs (e.g. geography facts) from reaching BERT.

    Returns:
        True  → valid behavioural description, BERT should run
        False → irrelevant input, reject and ask again
    """
    check = llm.invoke([
        SystemMessage(content=(
            "You are a strict input filter for a medical screening tool.\n"
            "Reply ONLY with yes or no — nothing else, no punctuation.\n"
            "Question: Is the following text a description of a child's behaviour, "
            "development, communication, or social traits?"
        )),
        HumanMessage(content=text)
    ])
    return check.content.strip().lower().startswith("yes")


# ══════════════════════════════════════════════════════════════════════════════
# Supervisor system prompt
# ══════════════════════════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM = """
You are the supervisor of an ASD (Autism Spectrum Disorder) screening assistant.
Your ONLY domain is ASD screening in children. Politely refuse all off-topic questions.

You will receive the current conversation stage and must reply with ONLY valid JSON:
{"next": "<routing>", "reply": "<message to user>"}

Routing values:
  "questionnaire_agent" -> user wants the Q-CHAT-10 questionnaire method
  "text_agent"          -> user wants the text description method
  "end"                 -> no sub-agent needed, just reply to user

Stage rules:
  idle                  -> Greet warmly, explain ASD screening, ask yes/no. next=end
  idle + yes/sure/ok    -> Ask them to choose: questionnaire or text. next=end  (stage->choose_method)
  idle + no             -> Acknowledge, say you are available. next=end
  choose_method + questionnaire/qchat/1 -> next=questionnaire_agent
  choose_method + text/describe/2       -> next=text_agent
  post_result + yes/another/again/sure  -> Ask them to choose method again. next=end  (stage->choose_method)
  post_result + no/done/exit/quit       -> Thank them and say goodbye. next=end  (stage->idle)
  Any stage + hi/hello                  -> Greet and introduce yourself. next=end
  Any stage + who are you / what do you do -> Explain your purpose. next=end
  Any stage + off-topic                 -> Politely decline and redirect. next=end
  Any stage + bye/exit/quit             -> Say goodbye. next=end
"""


# ══════════════════════════════════════════════════════════════════════════════
# Agent node functions
# ══════════════════════════════════════════════════════════════════════════════

def build_supervisor(xgboost_model, bert_tokenizer, bert_model_obj):
    """
    Factory: returns the supervisor_agent function with models in closure scope.
    This lets us pass loaded models in without using global variables.
    """

    def supervisor_agent(state: State) -> State:
        stage    = state.get("stage", "idle")
        messages = state.get("messages", [])

        # ── Fast-path: skip LLM, route straight to sub-agent ──────────────
        if stage in ("awaiting_answers", "awaiting_description"):
            state["answer"] = ""
            return state

        # ── Build LLM message history ──────────────────────────────────────
        history = [
            SystemMessage(content=SUPERVISOR_SYSTEM),
            SystemMessage(content=f"Current stage: {stage}"),
        ]
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                history.append(msg)
        if not messages:
            history.append(HumanMessage(content="[START]"))

        # ── Call LLM ──────────────────────────────────────────────────────
        raw = llm.invoke(history)

        # ── Parse JSON with regex fallback ────────────────────────────────
        try:
            text = raw.content.strip().strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            text  = match.group(0) if match else text
            data  = json.loads(text)
            next_ = data.get("next", "end")
            reply = data.get("reply", "")
        except (json.JSONDecodeError, AttributeError):
            next_ = "end"
            reply = raw.content.strip()

        if next_ not in ("questionnaire_agent", "text_agent", "end"):
            next_ = "end"

        # ── Update stage ──────────────────────────────────────────────────
        if next_ == "questionnaire_agent":
            state["stage"] = "awaiting_answers"
        elif next_ == "text_agent":
            state["stage"] = "awaiting_description"
        elif stage == "idle" and next_ == "end":
            if any(w in reply.lower() for w in ["questionnaire", "text", "method", "choose", "which"]):
                state["stage"] = "choose_method"
        elif stage == "post_result" and next_ == "end":
            if any(w in reply.lower() for w in ["questionnaire", "text", "method", "choose", "which"]):
                state["stage"] = "choose_method"
            else:
                state["stage"] = "idle"

        state["messages"] = messages + [AIMessage(content=reply)]
        state["answer"]   = reply if next_ == "end" else ""
        return state

    def questionnaire_agent(state: State) -> State:
        messages   = state.get("messages", [])
        last_human = next(
            (m.content.strip() for m in reversed(messages) if isinstance(m, HumanMessage)),
            None
        )

        QUESTIONS = (
            "📋 **Q-CHAT-10 Questionnaire**\n\n"
            "Answer in **0** (No) or **1** (Yes) for each question:\n\n"
            " 1.   Does your child use simple gestures? (e.g. wave goodbye)\n"
            " 2.   Does your child follow where you're looking?\n"
            " 3.   Does your child pretend? (e.g. care for dolls, toy phone)\n"
            " 4.   If someone is upset, does your child try to comfort them?\n"
            " 5.   Does your child point to share interest with you?\n"
            " 6.   Does your child look at you when you call his/her name?\n"
            " 7.   How easy is it to get eye contact with your child?\n"
            " 8.   Would you describe your child's first words as normal?\n"
            " 9.   Does your child point to indicate they want something?\n"
            "10.   Does your child stare at nothing with no apparent purpose?\n"
            "11.   Child biological sex (0=Female, 1=Male)\n\n"
            "Enter all 11 answers as comma-separated numbers.\n"
            "Example: `0,1,0,1,1,0,0,0,1,0,0`\n\n"
            "**Your answers:**"
        )

        POST_RESULT = (
            "\n\n---\n"
            "Would you like to run **another assessment**?\n"
            "• Type **yes** — to try again\n"
            "• Type **no**  — to exit"
        )

        if state.get("stage") == "awaiting_answers" and last_human and "," in last_human:
            result = parse_answers(last_human)
            if isinstance(result, str):
                reply = f"❌ {result}\n\n{QUESTIONS}"
                state["messages"] = messages + [AIMessage(content=reply)]
                state["answer"]   = reply
                state["stage"]    = "awaiting_answers"
            else:
                answer = questionnaire_predict(xgboost_model, result) + POST_RESULT
                state["messages"] = messages + [AIMessage(content=answer)]
                state["answer"]   = answer
                state["stage"]    = "post_result"
            return state

        state["messages"] = messages + [AIMessage(content=QUESTIONS)]
        state["answer"]   = QUESTIONS
        state["stage"]    = "awaiting_answers"
        return state

    def text_agent(state: State) -> State:
        messages   = state.get("messages", [])
        last_human = next(
            (m.content.strip() for m in reversed(messages) if isinstance(m, HumanMessage)),
            None
        )

        PROMPT = (
            "📝 **Text Description Method**\n\n"
            "Describe the child's behaviour in your own words.\n"
            "Include details about:\n"
            "• Social interactions\n"
            "• Communication patterns\n"
            "• Repetitive behaviours\n"
            "• Response to surroundings\n\n"
            "**Example:** *'My 3-year-old rarely makes eye contact and does not respond to his name.'*\n\n"
            "**Your description:**"
        )

        POST_RESULT = (
            "\n\n---\n"
            "Would you like to run **another assessment**?\n"
            "• Type **yes** — to try again\n"
            "• Type **no**  — to exit"
        )

        if state.get("stage") == "awaiting_description" and last_human and len(last_human.split()) >= 3:
            # ── Relevance gate ────────────────────────────────────────────
            if not is_valid_description(last_human):
                reply = (
                    "❌ That doesn't look like a behavioural description.\n\n"
                    "Please describe the **child's behaviour**, communication, or social traits.\n\n"
                    "**Example:** *'My 3-year-old rarely makes eye contact and does not respond to his name.'*\n\n"
                    "**Your description:**"
                )
                state["messages"] = messages + [AIMessage(content=reply)]
                state["answer"]   = reply
                state["stage"]    = "awaiting_description"
                return state

            # ── Run BERT ──────────────────────────────────────────────────
            answer = text_predict(bert_tokenizer, bert_model_obj, last_human) + POST_RESULT
            state["messages"] = messages + [AIMessage(content=answer)]
            state["answer"]   = answer
            state["stage"]    = "post_result"
            return state

        state["messages"] = messages + [AIMessage(content=PROMPT)]
        state["answer"]   = PROMPT
        state["stage"]    = "awaiting_description"
        return state

    return supervisor_agent, questionnaire_agent, text_agent


# ══════════════════════════════════════════════════════════════════════════════
# Routing function
# ══════════════════════════════════════════════════════════════════════════════

def routing_logic(state: State) -> Literal["questionnaire_agent", "text_agent", "end"]:
    """Reads stage from state and tells LangGraph which node to go to next."""
    stage = state.get("stage", "idle")
    if stage == "awaiting_answers":
        return "questionnaire_agent"
    elif stage == "awaiting_description":
        return "text_agent"
    return "end"


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder  (called once from app.py after models are loaded)
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(xgboost_model, bert_tokenizer, bert_model_obj):
    """
    Builds and compiles the LangGraph agent.
    Call this after loading the ML models.

    Returns:
        compiled LangGraph app ready for .invoke()
    """
    supervisor, questionnaire, text = build_supervisor(
        xgboost_model, bert_tokenizer, bert_model_obj
    )

    workflow = StateGraph(State)

    workflow.add_node("supervisor_agent",    supervisor)
    workflow.add_node("questionnaire_agent", questionnaire)
    workflow.add_node("text_agent",          text)

    workflow.add_edge(START, "supervisor_agent")

    workflow.add_conditional_edges("supervisor_agent", routing_logic, {
        "questionnaire_agent": "questionnaire_agent",
        "text_agent"         : "text_agent",
        "end"                : END,
    })

    workflow.add_edge("questionnaire_agent", END)
    workflow.add_edge("text_agent",          END)

    return workflow.compile()
