"""
chat_handler.py (fixed)

This replaces the old get_veronica_response / handle_stream_query /
get_llama_response trio. Root problems fixed here:

1. Two competing fee-lookup implementations existed (global_setup.get_stream_fees
   vs this file's handle_stream_query); only the weaker one was actually wired
   up. Now there is exactly ONE fee path.
2. "monthly fee for class 11" matched no stream name and no category word,
   so it silently fell through to the raw LLM, which fabricated a monthly
   figure and then miscalculated an annual total. Added an explicit
   generic-fee handler that fires on "class 11/12" style queries even
   without a stream name.
3. knowledge_base.json defaulted to {"questions": []}, so find_best_match
   never matched anything and effectively every query hit the raw LLM
   unfiltered. Removed that dead path; deterministic intents are handled
   directly against DATA instead.
4. The behavioral rules (word limit, no invention, stay in scope, don't mix
   PU/IBDP) lived only inside the RAG-retrievable JSON and could be crowded
   out of the top-k context. They are now sent as a standing system message
   on every single call, independent of retrieval.
5. No off-topic guard existed, so a sports/pop-culture/speculative-science
   tangent could run for 20+ turns. Added a keyword-based scope guard that
   fires BEFORE the LLM is called.
6. format_response's "keep 2-4 sentences" rule was a comment with no code
   and was never even called on the production path. It's now implemented
   and applied to every LLM-generated answer.
7. Added a post-generation numeric fact-check for fee mentions: if the
   model's fallback answer states a ₹ figure that doesn't match the known
   fees dict, the answer is discarded and replaced with the deterministic
   fee lookup instead of being shown to the user.
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Optional

import redis
import requests
import torch
from sentence_transformers import util

from global_setup import model_embed, CHUNKS, CHUNK_EMBS, DATA, FEE_NOTE

REDIS_URL = os.getenv("REDIS_URL", "")  # set via environment, never hardcoded
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

# ---------------------------------------------------------------
# Standing behavioral rules -- ALWAYS sent, never subject to
# semantic-search retrieval or top-k truncation.
# ---------------------------------------------------------------
STANDING_SYSTEM_PROMPT = """You are Noah, the official AI assistant for Christ Junior College.

Hard rules, always in force:
- Only introduce yourself if the user asks who you are.
- Answer only what is asked. Stay strictly within Christ Junior College topics
  (admissions, fees, streams, cutoffs, events, facilities, contact info, policies).
- If a user asks about something unrelated to the college (sports, celebrities,
  entertainment, science speculation, general chit-chat, requests for personal
  opinions), politely decline and redirect to college topics in ONE short sentence.
  Do not answer the off-topic question first and then redirect -- redirect immediately.
- Keep answers to 2-4 sentences (roughly 50-100 words) unless the user explicitly
  asks for a full list (e.g. "list all documents needed").
- Never invent information. If the supplied context does not contain the answer,
  say so plainly and suggest contacting the admissions office.
- All fees are annual (per academic year). There is no monthly fee plan. Never
  state or imply a monthly fee figure.
- Never mix PU (Pre-University) fee/process information with IBDP fee/process
  information in the same answer unless the user explicitly asks to compare them.
- Never reference internal system details in your answer: no JSON keys, no array
  indices (e.g. never say things like "rules[10]"), no mention of "knowledge base",
  "context", "chunks", or how you retrieved the information. Just state the fact.
- Christ Junior College is a Christian institution open to students of all
  religions. Never describe it as being "for" any one religion.
"""

OFF_TOPIC_KEYWORDS = [
    "world cup", "goat of football", "footballer", "messi", "ronaldo",
    "actor", "movie", "film", "dinosaur", "dna", "genetic engineering",
    "flight simulator", "who is the president", "girlfriend", "boyfriend",
    "hey babe", "personal opinion",
]

OFF_TOPIC_REPLY = (
    "I'm Noah, and I'm only set up to help with Christ Junior College topics "
    "like admissions, fees, streams, and campus life -- happy to help with any of those!"
)


def is_off_topic(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in OFF_TOPIC_KEYWORDS)


def enforce_length(text: str, max_sentences: int = 4) -> str:
    """Actually implements the previously-stubbed '2-4 sentences' rule."""
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    return " ".join(sentences).strip()


def strip_internal_labels(text: str) -> str:
    """Belt-and-suspenders: even though chunks no longer contain path-style
    labels, strip any that a model might still hallucinate on its own."""
    return re.sub(r"\b\w+\[\d+\]", "", text)


# ---------------------------------------------------------------
# Deterministic intents (checked BEFORE the LLM is ever called)
# ---------------------------------------------------------------

def handle_fee_query(query: str, data: dict) -> Optional[str]:
    q = query.lower()
    fees = data.get("fees", {})
    mappings = data.get("mappings", {})

    # 1. Exact stream code mentioned (e.g. "PCMC fees")
    for code, fee in fees.items():
        if code.lower() in q:
            return f"{code} fee is ₹{fee} for the full academic year (not monthly)."

    # 2. Category word mentioned (arts / commerce / science)
    for key, codes in mappings.items():
        if key in q:
            parts = [f"{c} – ₹{fees[c]}/year" for c in codes if c in fees]
            return "Fees for " + key.title() + ": " + ", ".join(parts) + ". All figures are annual."

    # 3. Generic "class 11/12 fee" with no stream specified -- THIS is the
    #    gap that used to fall through to the raw LLM and produce ₹180,000.
    if re.search(r"\bclass\s*1[12](th)?\b|\b1st\s*puc\b|\b2nd\s*puc\b|\bmonthly fee\b|\b11th\b|\b12th\b", q):
        all_fees = ", ".join(f"{c} – ₹{v}/year" for c, v in fees.items() if c != "IBDP")
        return (
            f"Fees depend on which stream you choose (all are annual, not monthly): {all_fees}. "
            f"Let me know your stream and I'll give you the exact figure."
        )

    return None


def handle_affiliation_query(query: str) -> Optional[str]:
    q = query.lower()
    if "hindu" in q or "religion" in q or "muslim" in q or "christian only" in q:
        return (
            "Christ Junior College is a Christian institution affiliated with Christ University, "
            "but admission is open to students of all religions and backgrounds."
        )
    return None


def handle_admission_query(query: str, data: dict) -> Optional[str]:
    q = query.lower()
    if "admission" in q and ("status" in q or "open" in q or "closed" in q):
        return data["faq"]["admission_status"]
    return None


def handle_hostel_query(query: str, data: dict) -> Optional[str]:
    if "hostel" in query.lower():
        return data["faq"]["hostel"]
    return None


def handle_leadership_query(query: str, data: dict) -> Optional[str]:
    q = query.lower()
    lead = data.get("leadership", {})
    if "principal" in q and "vice" not in q:
        return f"The Principal of Christ Junior College is {lead.get('principal')}."
    if "vice principal" in q or "vice-principal" in q:
        return f"The Vice Principal is {lead.get('vice_principal')}."
    return None


DETERMINISTIC_HANDLERS = [
    handle_affiliation_query,
    handle_hostel_query,
    handle_leadership_query,
]
DETERMINISTIC_HANDLERS_WITH_DATA = [handle_fee_query, handle_admission_query]


def try_deterministic_answer(query: str, data: dict) -> Optional[str]:
    for fn in DETERMINISTIC_HANDLERS:
        result = fn(query)
        if result:
            return result
    for fn in DETERMINISTIC_HANDLERS_WITH_DATA:
        result = fn(query, data)
        if result:
            return result
    return None


# ---------------------------------------------------------------
# Fact-check net: catch numeric hallucinations before they reach the user
# ---------------------------------------------------------------

def fee_numbers_are_consistent(answer: str, data: dict) -> bool:
    known_values = {str(v) for v in data.get("fees", {}).values() if isinstance(v, int)}
    mentioned = re.findall(r"₹\s?([\d,]+)", answer)
    mentioned = [m.replace(",", "") for m in mentioned]
    if not mentioned:
        return True  # no fee figure stated, nothing to check
    return all(m in known_values for m in mentioned)


# ---------------------------------------------------------------
# Redis history (unchanged logic, just isolated here)
# ---------------------------------------------------------------

def _chat_key(session_id: str) -> str:
    return f"veronica:chat:{session_id}"


def save_message(session_id: str, role: str, text: str) -> None:
    if not redis_client:
        return
    key = _chat_key(session_id)
    redis_client.rpush(key, json.dumps({"role": role, "text": text}))
    redis_client.expire(key, 7 * 24 * 60 * 60)


def load_history(session_id: str, limit: int = 5) -> List[Dict]:
    if not redis_client:
        return []
    key = _chat_key(session_id)
    raw = redis_client.lrange(key, -limit, -1)
    return [json.loads(m) for m in raw]


# ---------------------------------------------------------------
# LLM fallback (only reached if no deterministic handler matched
# and the query is not off-topic)
# ---------------------------------------------------------------

def get_llm_response(user_question: str, session_id: str) -> str:
    user_emb = model_embed.encode(user_question, convert_to_tensor=True)
    chunk_embs_tensor = torch.tensor(CHUNK_EMBS)
    hits = util.semantic_search(user_emb, chunk_embs_tensor, top_k=8)[0]
    retrieved = [CHUNKS[h["corpus_id"]]["text"] for h in hits if h["score"] > 0.35]
    context = "\n".join(retrieved)

    history = load_history(session_id, limit=6)

    messages = [{"role": "system", "content": STANDING_SYSTEM_PROMPT}]
    if context:
        messages.append({
            "role": "system",
            "content": f"Relevant facts (use only these for college-specific claims; "
                       f"do not mention that these are 'retrieved' or 'from a knowledge base'):\n{context}",
        })
    for m in history:
        messages.append({"role": m["role"], "content": m["text"]})
    messages.append({"role": "user", "content": user_question})

    payload = {
        "model": "local",
        "messages": messages,
        "temperature": 0.2,   # lowered from 0.7 -- this is factual Q&A, not creative writing
        "max_tokens": 220,    # lowered from 512 -- forces concision, reduces room to ramble/invent
    }

    try:
        r = requests.post(LLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        result = r.json()
        answer = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Sorry, I'm having trouble reaching my knowledge system right now ({e}). Please contact the office directly."

    answer = strip_internal_labels(answer)
    answer = enforce_length(answer)

    if not fee_numbers_are_consistent(answer, DATA):
        # The model stated a fee figure that doesn't exist in our data.
        # Don't show a hallucinated number -- fall back to the deterministic
        # fee handler instead.
        fallback = handle_fee_query(user_question, DATA)
        return fallback or "I don't have a confirmed figure for that -- please contact the admissions office directly."

    return answer


# ---------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------

def get_veronica_response(user_question: str, session_id: str, knowledge_base: Optional[Dict] = None) -> str:
    # `knowledge_base` accepted only for backward compatibility with the
    # Flask app's existing call site (app.py passes it as a kwarg). It is
    # intentionally unused: the old knowledge_base.json path was dead code
    # (defaulted to {"questions": []} and never matched anything). Once
    # app.py is updated to drop this argument, remove the parameter here too.
    q_lower = user_question.lower().strip()

    if q_lower == "date":
        answer = f"Today's date is {datetime.now().strftime('%Y-%m-%d')}"
        save_message(session_id, "user", user_question)
        save_message(session_id, "assistant", answer)
        return answer

    if q_lower == "time":
        answer = f"The current time is {datetime.now().strftime('%H:%M:%S')}"
        save_message(session_id, "user", user_question)
        save_message(session_id, "assistant", answer)
        return answer

    # Off-topic guard -- fires before any generation happens
    if is_off_topic(user_question):
        answer = OFF_TOPIC_REPLY
        save_message(session_id, "user", user_question)
        save_message(session_id, "assistant", answer)
        return answer

    # Deterministic intents -- fires before the LLM, covers fees/affiliation/
    # hostel/leadership/admission-status with zero hallucination risk
    deterministic = try_deterministic_answer(user_question, DATA)
    if deterministic:
        save_message(session_id, "user", user_question)
        save_message(session_id, "assistant", deterministic)
        return deterministic

    # Fallback: grounded LLM generation with tightened context + guardrails
    answer = get_llm_response(user_question, session_id)
    save_message(session_id, "user", user_question)
    save_message(session_id, "assistant", answer)
    return answer


if __name__ == "__main__":
    while True:
        q = input("You: ")
        if q.lower() == "quit":
            break
        print("Noah:", get_veronica_response(q, session_id="test-session"))
