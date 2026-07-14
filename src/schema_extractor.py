"""
schema_extractor.py
────────────────────────────────────────────────────────
Connects to Neo4j and extracts the full graph schema:
  - All node types and their properties
  - All relationship types and which nodes they connect
  - Sample values for key properties
  - Entity counts per type
  - Outputs a clean schema.json + prints a readable summary

Run this AFTER graph_load.py has populated Neo4j.

Usage:
  python src/schema_extractor.py

Output:
  data/processed/schema.json   ← machine-readable, used by RAG pipeline
  prints human-readable summary to console
"""

import json
from neo4j import GraphDatabase
from collections import defaultdict

# ── Connection ────────────────────────────────────────
URI      = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "12345678"    # change this

OUTPUT_FILE = "../data/processed/schema.json"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

schema = {
    "node_types"        : {},
    "relationship_types": [],
    "entity_counts"     : {},
    "sample_values"     : {},
}

with driver.session() as session:

    # ── 1. Get all node labels and their properties ───
    print("=" * 55)
    print("  Extracting node types and properties...")
    print("=" * 55)

    labels_result = session.run("CALL db.labels()")
    labels = [r["label"] for r in labels_result]

    for label in labels:
        # Get property keys for this label
        props_result = session.run(
            f"MATCH (n:{label}) UNWIND keys(n) AS key "
            f"RETURN DISTINCT key ORDER BY key"
        )
        properties = [r["key"] for r in props_result]

        # Count nodes of this type
        count_result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
        count = count_result.single()["count"]

        schema["node_types"][label] = {
            "properties": properties,
            "count"     : count
        }
        schema["entity_counts"][label] = count

        # Get sample values for each property (up to 5)
        sample_values = {}
        for prop in properties:
            sample_result = session.run(
                f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL "
                f"RETURN DISTINCT n.{prop} AS val LIMIT 5"
            )
            sample_values[prop] = [r["val"] for r in sample_result]
        schema["sample_values"][label] = sample_values

        print(f"\n  [{label}] — {count} nodes")
        print(f"    Properties : {', '.join(properties)}")
        for prop, vals in sample_values.items():
            print(f"    {prop} samples: {vals}")

    # ── 2. Get all relationship types ─────────────────
    print("\n" + "=" * 55)
    print("  Extracting relationship types...")
    print("=" * 55)

    rel_result = session.run("""
        MATCH (a)-[r]->(b)
        RETURN DISTINCT
            labels(a)[0]  AS from_label,
            type(r)        AS relationship,
            labels(b)[0]  AS to_label,
            count(r)       AS count
        ORDER BY count DESC
    """)

    for record in rel_result:
        entry = {
            "from"        : record["from_label"],
            "relationship": record["relationship"],
            "to"          : record["to_label"],
            "count"       : record["count"]
        }
        schema["relationship_types"].append(entry)
        print(f"  ({record['from_label']})"
              f"-[:{record['relationship']}]->"
              f"({record['to_label']})"
              f"  ×{record['count']}")

    # ── 3. Get all equipment names (key for demo) ─────
    print("\n" + "=" * 55)
    print("  All Equipment nodes in graph:")
    print("=" * 55)

    equip_result = session.run(
        "MATCH (e:Equipment) RETURN e.name AS name ORDER BY e.name"
    )
    equipment_names = [r["name"] for r in equip_result]
    schema["all_equipment"] = equipment_names
    for name in equipment_names:
        print(f"  • {name}")

    # ── 4. Get all regulation nodes ───────────────────
    print("\n" + "=" * 55)
    print("  All Regulation nodes in graph:")
    print("=" * 55)

    reg_result = session.run(
        "MATCH (r:Regulation) RETURN r.name AS name ORDER BY r.name"
    )
    regulation_names = [r["name"] for r in reg_result]
    schema["all_regulations"] = regulation_names
    for name in regulation_names:
        print(f"  • {name}")

    # ── 5. Get all document nodes ─────────────────────
    print("\n" + "=" * 55)
    print("  All Document nodes in graph:")
    print("=" * 55)

    doc_result = session.run(
        "MATCH (d:Document) "
        "RETURN d.source_file AS file, d.doc_type AS type "
        "ORDER BY d.doc_type, d.source_file"
    )
    documents = [{"file": r["file"], "type": r["type"]} for r in doc_result]
    schema["all_documents"] = documents
    for d in documents:
        print(f"  • [{d['type']}] {d['file']}")

    # ── 6. Connectivity summary (most connected nodes) ─
    print("\n" + "=" * 55)
    print("  Most connected nodes (top 10):")
    print("=" * 55)

    conn_result = session.run("""
        MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        ORDER BY degree DESC
        LIMIT 10
        RETURN
            labels(n)[0] AS label,
            COALESCE(n.name, n.source_file, n.description, "unnamed") AS id,
            degree
    """)

    connectivity = []
    for record in conn_result:
        entry = {
            "label" : record["label"],
            "id"    : record["id"],
            "degree": record["degree"]
        }
        connectivity.append(entry)
        print(f"  {record['label']}: {record['id']}  "
              f"({record['degree']} connections)")

    schema["most_connected"] = connectivity

# ── Save schema to JSON ───────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, default=str)

print("\n" + "=" * 55)
print(f"  Schema saved to {OUTPUT_FILE}")
print("=" * 55)

# ── Print a compact schema string (useful for LLM prompts) ─
print("\n" + "=" * 55)
print("  COMPACT SCHEMA (copy this into your LLM system prompt)")
print("=" * 55)

print("\nNODE TYPES:")
for label, info in schema["node_types"].items():
    print(f"  ({label}) props: {info['properties']}  [{info['count']} nodes]")

print("\nRELATIONSHIP TYPES:")
for rel in schema["relationship_types"]:
    print(f"  ({rel['from']})-[:{rel['relationship']}]->({rel['to']})")

driver.close()
print("\nDone.")