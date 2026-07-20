"""
graph_query.py
Clean query interface for app.py to import and use.
"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USERNAME", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "")
    )
)

def get_everything_about(entity_name: str) -> list:
    with driver.session() as session:
        result = session.run("""
            MATCH (n)-[r]-(connected)
            WHERE toLower(n.name) CONTAINS toLower($name)
               OR toLower(n.description) CONTAINS toLower($name)
            RETURN
                type(r)            AS relationship,
                labels(connected)[0] AS connected_type,
                COALESCE(connected.name, connected.description,
                         connected.source_file) AS connected_id
            ORDER BY type(r)
            LIMIT 30
        """, name=entity_name)
        return [dict(r) for r in result]


def get_equipment_incidents_regulations() -> list:
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Equipment)-[:HAD_INCIDENT]->(i:Incident)
            MATCH (e)-[:GOVERNED_BY]->(reg:Regulation)
            RETURN
                e.name          AS equipment,
                i.description   AS incident,
                reg.name        AS regulation
        """)
        return [dict(r) for r in result]


def get_parts_for_equipment(equipment_name: str) -> list:
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Part)-[:USED_IN_REPAIR_OF]->(e:Equipment)
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN p.name AS part, e.name AS equipment
        """, name=equipment_name)
        return [dict(r) for r in result]


def get_all_equipment() -> list:
    with driver.session() as session:
        result = session.run(
            "MATCH (e:Equipment) RETURN e.name AS name ORDER BY e.name"
        )
        return [r["name"] for r in result]


def get_compliance_chain(equipment_name: str) -> list:
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Equipment)-[:GOVERNED_BY]->(reg:Regulation)
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e.name AS equipment, reg.name AS regulation
        """, name=equipment_name)
        return [dict(r) for r in result]
    
def get_graph_visualization(equipment_name: str = None, limit: int = 150) -> dict:
    """
    Returns nodes and edges for graph visualization.
    If equipment_name is given, returns its 2-hop neighborhood (machine, its
    parts, incidents, people, regulations, etc). Otherwise returns a capped
    view of the whole graph for an overview.
    """
    with driver.session() as session:
        if equipment_name:
            query = """
                MATCH (e)-[r*1..2]-(connected)
                WHERE toLower(e.name) CONTAINS toLower($name)
                UNWIND r AS rel
                WITH startNode(rel) AS a, endNode(rel) AS b, type(rel) AS rtype
                RETURN DISTINCT
                    labels(a)[0] AS a_label, COALESCE(a.name, a.description, a.source_file) AS a_id,
                    labels(b)[0] AS b_label, COALESCE(b.name, b.description, b.source_file) AS b_id,
                    rtype
                LIMIT $limit
            """
            result = session.run(query, name=equipment_name, limit=limit)
        else:
            query = """
                MATCH (a)-[r]->(b)
                RETURN DISTINCT
                    labels(a)[0] AS a_label, COALESCE(a.name, a.description, a.source_file) AS a_id,
                    labels(b)[0] AS b_label, COALESCE(b.name, b.description, b.source_file) AS b_id,
                    type(r) AS rtype
                LIMIT $limit
            """
            result = session.run(query, limit=limit)

        nodes = {}
        edges = []
        for record in result:
            a_key = f"{record['a_label']}:{record['a_id']}"
            b_key = f"{record['b_label']}:{record['b_id']}"
            nodes[a_key] = {"id": a_key, "label": record["a_id"], "group": record["a_label"]}
            nodes[b_key] = {"id": b_key, "label": record["b_id"], "group": record["b_label"]}
            edges.append({"from": a_key, "to": b_key, "label": record["rtype"]})

        return {"nodes": list(nodes.values()), "edges": edges}