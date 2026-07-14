"""
graph_load.py
Reads entities.jsonl → loads nodes and relationships into Neo4j
"""

import json
from neo4j import GraphDatabase
from tqdm import tqdm

# ── Connection ────────────────────────────────────────
URI      = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "12345678"   # whatever you set in Neo4j Desktop

ENTITIES_FILE = "../data/processed/entities.jsonl"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# ── Helper: run a Cypher query ────────────────────────
def run(session, query, **params):
    session.run(query, **params)

# ── Node creation functions ───────────────────────────
# MERGE means: create this node only if it doesn't already exist
# This prevents duplicate PUMP-101 nodes from different chunks

def create_equipment(session, name):
    run(session,
        "MERGE (e:Equipment {name: $name})",
        name=name.strip())

def create_person(session, name):
    run(session,
        "MERGE (p:Person {name: $name})",
        name=name.strip())

def create_regulation(session, name):
    run(session,
        "MERGE (r:Regulation {name: $name})",
        name=name.strip())

def create_part(session, name):
    run(session,
        "MERGE (pt:Part {name: $name})",
        name=name.strip())

def create_incident(session, name):
    run(session,
        "MERGE (i:Incident {description: $name})",
        name=name.strip())

def create_vendor(session, name):
    run(session,
        "MERGE (v:Vendor {name: $name})",
        name=name.strip())

def create_document(session, source_file, doc_type):
    run(session,
        "MERGE (d:Document {source_file: $source_file}) "
        "SET d.doc_type = $doc_type",
        source_file=source_file, doc_type=doc_type)

# ── Relationship creation ─────────────────────────────
# Maps relationship strings from LLM output to Cypher queries

RELATIONSHIP_QUERIES = {
    "REPAIRED_BY": """
        MATCH (a {name: $from_name}), (b {name: $to_name})
        MERGE (a)-[:REPAIRED_BY]->(b)
    """,
    "GOVERNED_BY": """
        MATCH (a {name: $from_name}), (b {name: $to_name})
        MERGE (a)-[:GOVERNED_BY]->(b)
    """,
    "HAD_INCIDENT": """
        MATCH (a {name: $from_name}), (b {description: $to_name})
        MERGE (a)-[:HAD_INCIDENT]->(b)
    """,
    "MENTIONED_IN": """
        MATCH (a {name: $from_name}), (b {source_file: $to_name})
        MERGE (a)-[:MENTIONED_IN]->(b)
    """,
    "USED_IN_REPAIR_OF": """
        MATCH (a {name: $from_name}), (b {name: $to_name})
        MERGE (a)-[:USED_IN_REPAIR_OF]->(b)
    """,
    "PERFORMED": """
        MATCH (a {name: $from_name}), (b {source_file: $to_name})
        MERGE (a)-[:PERFORMED]->(b)
    """,
    "SUPPLIES": """
        MATCH (a {name: $from_name}), (b {name: $to_name})
        MERGE (a)-[:SUPPLIES]->(b)
    """,
    "LED_TO_REVISION_OF": """
        MATCH (a {description: $from_name}), (b {source_file: $to_name})
        MERGE (a)-[:LED_TO_REVISION_OF]->(b)
    """,
    "REFERENCES": """
        MATCH (a {source_file: $from_name}), (b {name: $to_name})
        MERGE (a)-[:REFERENCES]->(b)
    """,
}

# ── Load entities.jsonl and populate graph ────────────
records = []
with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} extraction results")

with driver.session() as session:

    # Pass 1 — create all nodes first
    print("\nPass 1: Creating nodes...")
    for record in tqdm(records):
        source_file = record["source_file"]
        doc_type    = record["doc_type"]
        entities    = record["extracted"].get("entities", {})

        create_document(session, source_file, doc_type)

        for name in entities.get("equipment",   []): create_equipment (session, name)
        for name in entities.get("people",      []): create_person    (session, name)
        for name in entities.get("regulations", []): create_regulation(session, name)
        for name in entities.get("parts",       []): create_part      (session, name)
        for name in entities.get("incidents",   []): create_incident  (session, name)
        for name in entities.get("vendors",     []): create_vendor    (session, name)

    # Pass 2 — create all relationships
    print("\nPass 2: Creating relationships...")
    for record in tqdm(records):
        relationships = record["extracted"].get("relationships", [])
        for rel in relationships:
            rel_type  = rel.get("relation", "")
            from_name = rel.get("from", "")
            to_name   = rel.get("to",   "")

            # Skip if LLM returned a list instead of a string
            if isinstance(from_name, list):
                from_name = from_name[0] if from_name else ""
            if isinstance(to_name, list):
                to_name = to_name[0] if to_name else ""
            if isinstance(rel_type, list):
                rel_type = rel_type[0] if rel_type else ""

            from_name = str(from_name).strip()
            to_name   = str(to_name).strip()
            rel_type  = str(rel_type).strip().upper()

            if not from_name or not to_name or not rel_type:
                continue

            query = RELATIONSHIP_QUERIES.get(rel_type)
            if query:
                try:
                    run(session,
                        query,
                        from_name=from_name,
                        to_name=to_name)
                except Exception as e:
                    pass  # node might not exist — skip silently

print("\nDone! Graph loaded into Neo4j.")
print("Open Neo4j Browser at http://localhost:7474 to explore.")