"""
feedback.py
────────────────────────────────────────────────────────
Stores thumbs up/down feedback on RAG answers and provides
a per-chunk score adjustment that retrieve_context() applies
before ranking results.
"""

import json
import os
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(BASE_DIR, "../data/processed/feedback.jsonl")

# How much each net vote shifts a chunk's retrieval score.
# Kept small and capped so one enthusiastic downvoter can't
# completely bury a chunk that's actually correct most of the time.
VOTE_WEIGHT = 0.05
MAX_ADJUSTMENT = 0.3


def submit_feedback(question: str, answer: str, chunk_ids: list, rating: str, correction: str = None) -> dict:
    """
    rating must be "up" or "down". Writes one record per chunk_id involved
    in the answer, so each cited chunk accumulates its own vote history.
    """
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")

    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        for chunk_id in chunk_ids:
            record = {
                "chunk_id": chunk_id,
                "question": question,
                "answer": answer,
                "rating": rating,
                "correction": correction,
                "timestamp": timestamp,
            }
            f.write(json.dumps(record) + "\n")

    return {"status": "ok", "chunks_recorded": len(chunk_ids)}


def get_chunk_adjustments() -> dict:
    """
    Reads feedback.jsonl and returns {chunk_id: net_adjustment}.
    +1 vote nudges score up slightly, -1 nudges it down, capped at
    MAX_ADJUSTMENT in either direction so it demotes/promotes rather
    than fully removing or guaranteeing a chunk.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return {}

    net_votes = {}
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            chunk_id = record["chunk_id"]
            delta = 1 if record["rating"] == "up" else -1
            net_votes[chunk_id] = net_votes.get(chunk_id, 0) + delta

    adjustments = {}
    for chunk_id, net in net_votes.items():
        adjustment = net * VOTE_WEIGHT
        adjustment = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adjustment))
        adjustments[chunk_id] = adjustment

    return adjustments