"""
global_setup.py (fixed)

Root problems this rewrite targets:
1. Naive `flatten_json` turned every JSON path into visible text like
   "rules[10]: ..." which the model then echoed back to users verbatim.
2. Bare numbers like fees.PCMC = 90000 had no unit attached, so the model
   invented "monthly" fees and did fabricated math.
3. Related facts (a stream's fee + its cutoffs + its full name) were split
   across disconnected chunks, so the model had to guess which numbers
   belonged together.
4. Internal-only data (creator identity block, redis config, command
   mappings) had no business being embedded and retrievable at all.

Fix: replace the generic recursive flattener with a small set of
hand-written, self-contained "fact cards" -- one per stream, one per policy
topic -- that state units and context explicitly and contain zero internal
path/array labels. This is a ~30-fact domain; hand-authored chunks beat a
generic flattener here.
"""

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

model_embed = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Fields that must NEVER be embedded / retrievable by the RAG layer.
# They are either internal plumbing or should only be surfaced through an
# explicit, hand-checked code path (see chat_handler.py), never via
# freeform LLM generation grounded on semantic search.
NON_RETRIEVABLE_TOP_LEVEL_KEYS = {
    "identity",       # creator/org info -- surfaced only via explicit rule, not RAG
    "commands",       # UI button wiring, not a fact
    "response_logic", # behavioral instructions -> goes in system prompt, not RAG
    "rules_culture",  # folded into a hand-written chunk below instead
    "logic",          # behavioral instructions -> system prompt
}

FEE_NOTE = "This is the total fee for the full academic year (₹, paid once per year, not monthly)."


def load_json(memory_path="veronica_memory.json"):
    mem_file = Path(memory_path)
    if not mem_file.exists():
        raise FileNotFoundError(f"{memory_path} not found")
    return json.loads(mem_file.read_text(encoding="utf-8"))


# -----------------------------------------------------------------
# STRUCTURED CHUNK BUILDERS
# Each function returns fully-formed, human-readable sentences with
# units and context baked in -- never "key: value" or "path[i]: value".
# -----------------------------------------------------------------

def _stream_full_name(code: str) -> str:
    names = {
        "PCMC": "Science - Physics, Chemistry, Maths, Computer Science",
        "PCMB": "Science - Physics, Chemistry, Maths, Biology",
        "CAMS": "Commerce - Accountancy, Maths, Statistics",
        "CAME": "Commerce - Accountancy, Maths, Economics",
        "HESP": "Arts - History, Economics, Sociology, Political Science",
        "HEPP": "Arts - History, Economics, Political Science, Psychology",
        "PPES": "Arts - Political Science, Psychology, Economics, Sociology",
    }
    return names.get(code, code)


def build_stream_chunks(data: dict) -> list[dict]:
    """One chunk per stream, binding code + full name + stream family +
    fee (with explicit /year unit) + cutoffs together, so the model never
    has to reassemble scattered facts."""
    chunks = []
    fees = data.get("fees", {})
    streams = data.get("streams", {})

    code_to_family = {}
    for family, codes in streams.items():
        for c in codes:
            code_to_family[c] = family

    # cutoffs live as separate rule strings in the source JSON; we already
    # have them hand-transcribed here per stream so they stay attached.
    cutoffs = {
        "PCMB": "SC 77.12%, ST NA, CAT I 82.72%, CAT IIA 56.4%, CAT IIB 90.4%, CAT IIIA 88.48%, CAT IIIB 83.68%",
        "PCMC": "SC 91.52%, ST 71.12%, CAT I 90.72%, CAT IIA 92.64%, CAT IIB 94.56%, CAT IIIA 93.76%, CAT IIIB 95.68%",
        "HESP": "SC 40%, ST 56.8%, CAT I NA, CAT IIA 68.64%, CAT IIB 59.52%, CAT IIIA 56.16%, CAT IIIB 48.64%",
        "HEPP": "SC 54.72%, ST 72.96%, CAT I NA, CAT IIA 67.84%, CAT IIB 49.5%, CAT IIIA 66.4%, CAT IIIB 64%",
        "PPES": "SC 65.76%, ST NA, CAT I NA, CAT IIA 66.72%, CAT IIB 68.32%, CAT IIIA 66.4%, CAT IIIB 68.64%",
    }

    for code, fee in fees.items():
        if code == "IBDP":
            continue
        family = code_to_family.get(code, "")
        text = (
            f"{code} ({_stream_full_name(code)}) is a 1st/2nd PUC {family} stream. "
            f"Its total fee is ₹{fee} for the full academic year (not monthly, not per semester). "
        )
        if code in cutoffs:
            text += f"Class 10 cutoff percentages for {code}: {cutoffs[code]}."
        chunks.append({"text": text, "topic": "stream_fee_cutoff", "key": code})

    ibdp_fee = fees.get("IBDP")
    if ibdp_fee:
        chunks.append({
            "text": f"The IB Diploma Programme (IBDP) fee is {ibdp_fee} for the full academic year, "
                    f"separate from PUC stream fees. IBDP and PUC fee structures must never be mixed together.",
            "topic": "stream_fee_cutoff",
            "key": "IBDP",
        })

    return chunks


def build_policy_chunks(data: dict) -> list[dict]:
    """Hand-written chunks for admissions, definitions, facilities, events,
    contact info, and the college's religious/institutional affiliation --
    the exact category that caused the Hindu/Christian mix-up, because the
    college's identity was never stated as one explicit standalone fact."""
    chunks = []

    chunks.append({
        "text": "Christ Junior College is a Christian institution affiliated with Christ University, Bangalore. "
                "It is open to students of all religions and backgrounds; it is not a Hindu institution and is not "
                "restricted to any one religion. The list of second-language options (Kannada, Hindi, Sanskrit, "
                "French) reflects the Karnataka PU Board's language requirements, not a religious affiliation.",
        "topic": "affiliation",
        "key": "religion",
    })

    faq = data.get("faq", {})
    chunks.append({
        "text": f"PUC admission status: {faq.get('admission_status', 'unknown')}.",
        "topic": "admissions", "key": "pu_status",
    })
    chunks.append({
        "text": f"Hostel: {faq.get('hostel', 'unknown')}.",
        "topic": "faq", "key": "hostel",
    })
    chunks.append({
        "text": f"A student cannot join 12th (2nd PUC) directly: {faq.get('direct_12th', '')}.",
        "topic": "faq", "key": "direct_12th",
    })
    timings = faq.get("timings", {})
    chunks.append({
        "text": f"College timings: {timings.get('college', '')}. Library: {timings.get('library', '')}. "
                f"Office: {timings.get('office', '')}.",
        "topic": "faq", "key": "timings",
    })
    contact = faq.get("contact", {})
    chunks.append({
        "text": f"Contact: phone {contact.get('phone', '')}, email {contact.get('email', '')}, "
                f"location {contact.get('location', '')}.",
        "topic": "faq", "key": "contact",
    })

    lead = data.get("leadership", {})
    chunks.append({
        "text": f"Principal: {lead.get('principal', '')}. Vice Principal: {lead.get('vice_principal', '')}.",
        "topic": "leadership", "key": "leadership",
    })

    for key, val in data.get("definitions", {}).items():
        if key.endswith("_full"):
            continue
        chunks.append({"text": val, "topic": "definition", "key": key})

    events = data.get("student_life", {}).get("events", [])
    clubs = data.get("student_life", {}).get("clubs", [])
    if events:
        chunks.append({
            "text": f"Christ Junior College's known annual events are: {', '.join(events)}. "
                    f"No other event names should be presented as official college events.",
            "topic": "events", "key": "events",
        })
    if clubs:
        chunks.append({
            "text": f"Student clubs at Christ Junior College: {', '.join(clubs)}.",
            "topic": "clubs", "key": "clubs",
        })

    docs_rule = (
        "Documents required at admission: Class X marks statement, Class X hall ticket, "
        "Transfer Certificate, Caste Certificate (only if claiming caste-category benefit), "
        "Aadhar Card copy, PEN (Permanent Education Number). After verification a payment link "
        "is emailed. ICSE/CBSE/other-state-board candidates additionally need a Consumer Number "
        "from the Karnataka PU eligibility portal; Karnataka State Board candidates do not need this."
    )
    chunks.append({"text": docs_rule, "topic": "admissions", "key": "documents"})

    facilities = data.get("facilities", [])
    if facilities:
        chunks.append({
            "text": f"Campus facilities: {', '.join(facilities)}.",
            "topic": "facilities", "key": "facilities",
        })

    return chunks


def build_all_chunks(data: dict) -> list[dict]:
    chunks = []
    chunks.extend(build_stream_chunks(data))
    chunks.extend(build_policy_chunks(data))

    # Safety net: assert nothing that looks like an internal path/index
    # label (e.g. "rules[10]", "fees.PCMC:") ever made it into a chunk.
    leak_pattern = re.compile(r"\b\w+\[\d+\]|(^|\s)\w+\.\w+:")
    for c in chunks:
        assert not leak_pattern.search(c["text"]), f"Internal-looking label leaked into chunk: {c['text'][:80]}"

    return chunks


def compute_embeddings(chunks, emb_path="chunk_embs.npy", force_recompute=False):
    emb_file = Path(emb_path)
    if emb_file.exists() and not force_recompute:
        return np.load(emb_path)
    texts = [c["text"] for c in chunks]
    embs = model_embed.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    np.save(emb_path, embs)
    return embs


def search_memory(query, chunks, chunk_embs, top_k=8):
    """top_k lowered from 20 -> 8. In a ~25-chunk corpus, 20 was returning
    almost the entire knowledge base as 'context' for every query, drowning
    out the relevant facts and giving the model room to blend unrelated
    chunks together (e.g. blending a stream's fee with another stream's
    cutoff). Fewer, more relevant chunks = less blending."""
    query_emb = model_embed.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = np.dot(chunk_embs, query_emb)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices if scores[i] > 0.35]  # relevance floor


DATA = load_json()
CHUNKS = build_all_chunks(DATA)
CHUNK_EMBS = compute_embeddings(CHUNKS)
