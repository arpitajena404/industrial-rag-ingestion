import os
import sys
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add current folder to sys.path — must happen before importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import generate_answer
from ingest_new_document import ingest_new_document

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