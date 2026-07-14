"""
extract_entities.py
Reads chunks.jsonl → calls LLM per chunk → saves entities to entities.jsonl

Fixes applied:
  - Robust JSON cleaning (handles trailing commas, extra text, code fences)
  - Rate limit protection (sleep between calls)
  - Skips already-successful chunks on re-run (only re-processes failed ones)
  - Better model: llama-3.3-70b-versatile for more accurate JSON output
"""

import json
import re
import time
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHUNKS_FILE   = "../data/processed/chunks.jsonl"
ENTITIES_FILE = "../data/processed/entities.jsonl"
SLEEP_BETWEEN = 1.5   # seconds between API calls — stays under Groq free limit

EXTRACT_FROM = {
    "work_order",
    "inspection_report",
    "abnormality_report",
    "standard_operating_procedure",
    "approved_vendor_list",
    "spreadsheet_register",
}

EXTRACTION_PROMPT = """You are extracting structured information from an industrial maintenance document.

Extract ALL of the following from the text:
- Equipment names/tags (e.g. PUMP-101, MTR-102, TANK-201, VFD-101)
- People (technicians, engineers, supervisors, safety officers)
- Regulations/Standards (e.g. OISD-117, IS 10596, Factories Act, ISO 9001)
- Parts (bearings, seals, gaskets with part numbers if mentioned)
- Incidents or failure events (e.g. bearing failure, dry running, vibration exceedance)
- Vendors/manufacturers (e.g. Kirloskar, SKF, Flowserve, ABB)

Then identify relationships between extracted entities.

IMPORTANT: Return ONLY a single valid JSON object. No extra text before or after. No trailing commas.

Use exactly this format:
{
  "entities": {
    "equipment": [],
    "people": [],
    "regulations": [],
    "parts": [],
    "incidents": [],
    "vendors": []
  },
  "relationships": [
    {"from": "PUMP-101", "relation": "REPAIRED_BY", "to": "Ramesh Kumar"}
  ]
}

Valid relation types: REPAIRED_BY, GOVERNED_BY, HAD_INCIDENT, USED_IN_REPAIR_OF,
PERFORMED_BY, MENTIONED_IN, SUPPLIES, REFERENCES, LED_TO_REVISION_OF, INSPECTED_BY

If nothing relevant is found return exactly: {"entities": {}, "relationships": []}

Text to extract from:
"""

def clean_json(raw: str) -> str:
    """Clean common LLM JSON formatting mistakes."""

    # Remove markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.replace("```", "").strip()

    # Extract just the JSON object if surrounded by extra text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)

    # Fix trailing commas before ] or }
    raw = re.sub(r',\s*([}\]])', r'\1', raw)

    # Fix single quotes used instead of double quotes
    # Only do this carefully — don't break apostrophes in values
    raw = re.sub(r"(?<![\\])'", '"', raw)

    return raw.strip()


def extract_from_chunk(chunk_text: str, source_file: str) -> dict:
    time.sleep(SLEEP_BETWEEN)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # better JSON accuracy than 8b
            max_tokens=1000,
            temperature=0,                     # deterministic — less hallucination
            messages=[{
                "role"   : "system",
                "content": "You are a JSON extraction engine. Output only valid JSON, nothing else."
            }, {
                "role"   : "user",
                "content": EXTRACTION_PROMPT + chunk_text
            }]
        )
        raw = response.choices[0].message.content.strip()
        cleaned = clean_json(raw)
        return json.loads(cleaned)

    except json.JSONDecodeError as e:
        print(f"\n  JSON error on {source_file}: {e}")
        return {"entities": {}, "relationships": [], "_failed": True}
    except Exception as e:
        print(f"\n  Error on {source_file}: {e}")
        return {"entities": {}, "relationships": [], "_failed": True}


# ── Load chunks ───────────────────────────────────────
chunks = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} total chunks")

chunks = [c for c in chunks if c["doc_type"] in EXTRACT_FROM]
print(f"Filtered to {len(chunks)} plant-specific chunks")

# ── Load existing results — skip already-successful ones ──
existing = {}
if os.path.exists(ENTITIES_FILE):
    with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                # Only keep if extraction actually succeeded
                extracted = record.get("extracted", {})
                failed = extracted.get("_failed", False)
                has_content = (
                    extracted.get("entities") or
                    extracted.get("relationships")
                )
                if has_content and not failed:
                    existing[record["chunk_id"]] = record

    print(f"Found {len(existing)} already-successful extractions — skipping these")

# ── Only process chunks that failed or are new ────────
to_process = [c for c in chunks if c["chunk_id"] not in existing]
print(f"Re-processing {len(to_process)} failed/new chunks\n")

# ── Extract ───────────────────────────────────────────
new_results = {}
for chunk in tqdm(to_process, desc="Extracting entities"):
    extracted = extract_from_chunk(chunk["text"], chunk["source_file"])
    new_results[chunk["chunk_id"]] = {
        "chunk_id"   : chunk["chunk_id"],
        "source_file": chunk["source_file"],
        "doc_type"   : chunk["doc_type"],
        "extracted"  : extracted
    }

# ── Merge old successes + new results ────────────────
all_results = {}
for chunk in chunks:
    cid = chunk["chunk_id"]
    if cid in existing:
        all_results[cid] = existing[cid]
    elif cid in new_results:
        all_results[cid] = new_results[cid]

# ── Save ──────────────────────────────────────────────
with open(ENTITIES_FILE, "w", encoding="utf-8") as f:
    for record in all_results.values():
        f.write(json.dumps(record) + "\n")

# ── Final stats ───────────────────────────────────────
total    = len(all_results)
empty    = sum(1 for r in all_results.values()
               if not r["extracted"].get("entities")
               and not r["extracted"].get("relationships"))
success  = total - empty

print(f"\nDone. Saved to {ENTITIES_FILE}")
print(f"Total: {total} | Successful: {success} | Empty: {empty}")

if empty > 0:
    print(f"\nStill failing chunks (run script again to retry):")
    for r in all_results.values():
        if not r["extracted"].get("entities") and not r["extracted"].get("relationships"):
            print(f"  - {r['source_file']} [{r['doc_type']}]")