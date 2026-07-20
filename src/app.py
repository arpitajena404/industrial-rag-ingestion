import os
import sys
import shutil
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add current folder to sys.path — must happen before importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import generate_answer
from ingest_new_document import ingest_new_document
from feedback import submit_feedback
from compliance import get_deviations, get_preset_readings, get_available_items
from graph_query import (
    get_everything_about,
    get_equipment_incidents_regulations,
    get_parts_for_equipment,
    get_compliance_chain,
    get_all_equipment,
    get_graph_visualization
)

try:
    from graph_query import (
        get_everything_about,
        get_equipment_incidents_regulations,
        get_parts_for_equipment,
        get_compliance_chain,
        get_all_equipment
    )
    GRAPH_AVAILABLE = True
except Exception as e:
    print(f"Warning: Knowledge graph unavailable — {e}")
    GRAPH_AVAILABLE = False

app = FastAPI(
    title="Industrial Knowledge Intelligence Center API",
    description="Backend API for querying industrial document vectors and generating answers via Gemini."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Request schemas ───────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class GraphRequest(BaseModel):
    entity: str
    query_type: str

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    chunk_ids: List[str]
    rating: str  # "up" or "down"
    correction: Optional[str] = None

class DeviationRequest(BaseModel):
    equipment: str
    readings: dict

# ── RAG endpoint ──────────────────────────────────────
@app.post("/api/query")
def query_rag_pipeline(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = generate_answer(request.question)
        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Graph endpoints ───────────────────────────────────
@app.post("/api/graph")
def query_knowledge_graph(request: GraphRequest):
    if not GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge graph is not available.")
    if not request.entity.strip():
        raise HTTPException(status_code=400, detail="Entity cannot be empty.")
    try:
        if request.query_type == "everything":
            results = get_everything_about(request.entity)
        elif request.query_type == "incidents":
            results = get_equipment_incidents_regulations()
        elif request.query_type == "parts":
            results = get_parts_for_equipment(request.entity)
        elif request.query_type == "compliance":
            results = get_compliance_chain(request.entity)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown query_type: {request.query_type}")
        return {"results": results, "entity": request.entity, "query_type": request.query_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/equipment")
def list_equipment():
    if not GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge graph is not available.")
    return {"equipment": get_all_equipment()}

@app.get("/api/graph/visualize")
def graph_visualize(equipment: str = None, limit: int = 150):
    if not GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge graph is not available.")
    try:
        return get_graph_visualization(equipment_name=equipment, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Ingest endpoint ────────────────────────────────────
@app.post("/api/ingest")
def ingest_document(file: UploadFile = File(...)):
    # Filename is kept exactly as uploaded (not renamed) — metadata.py's
    # classify_doc_type() depends on the filename prefix (e.g. "WO_" -> work_order),
    # so renaming here would break doc_type classification for every new upload.
    saved_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = ingest_new_document(saved_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Feedback endpoint ─────────────────────────────────
@app.post("/api/feedback")
def feedback_endpoint(request: FeedbackRequest):
    if request.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    try:
        result = submit_feedback(
            question=request.question,
            answer=request.answer,
            chunk_ids=request.chunk_ids,
            rating=request.rating,
            correction=request.correction,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
def get_stats():
    chunks_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/processed/chunks.jsonl")
    total_chunks = 0
    source_files = set()
    if os.path.exists(chunks_file):
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                total_chunks += 1
                source_files.add(record.get("source_file", ""))
    return {"total_chunks": total_chunks, "total_documents": len(source_files)}

@app.post("/api/compliance/check")
def check_compliance(request: DeviationRequest):
    if not request.equipment.strip():
        raise HTTPException(status_code=400, detail="Equipment name cannot be empty.")
    if not request.readings:
        raise HTTPException(status_code=400, detail="No readings provided.")
    try:
        result = get_deviations(request.equipment, request.readings)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compliance/preset/{equipment}")
def get_preset(equipment: str):
    readings = get_preset_readings(equipment)
    if not readings:
        raise HTTPException(status_code=404, detail=f"No preset readings for {equipment}")
    return {"equipment": equipment, "readings": readings}


@app.get("/api/compliance/items")
def list_inspection_items():
    return {"items": get_available_items()}

# ── Static files ──────────────────────────────────────
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    print("=" * 60)
    print("Industrial RAG FastAPI Server Launching")
    print("Available locally at: http://localhost:8000")
    print("=" * 60)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)