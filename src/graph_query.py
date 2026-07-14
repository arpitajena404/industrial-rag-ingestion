"""
graph_query.py
Clean query interface for app.py to import and use.
"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", os.getenv("NEO4J_PASSWORD", "your_password_here"))
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