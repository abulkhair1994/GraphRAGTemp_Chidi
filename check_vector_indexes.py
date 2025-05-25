from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))
)

with driver.session() as session:
    try:
        # Try the newer SHOW INDEXES syntax
        result = session.run('SHOW INDEXES WHERE type = "VECTOR"')
        indexes = list(result)
        print('Current Vector Indexes:')
        for idx in indexes:
            print(f'  - {idx["name"]}: {idx["labelsOrTypes"]} -> {idx["properties"]}')
        
        if not indexes:
            print('  No vector indexes found!')
            
    except Exception as e:
        print(f"Error checking indexes: {e}")
        # Try alternative approach
        try:
            result = session.run('CALL db.schema.visualization()')
            print("Database schema available, but vector index check failed")
        except Exception as e2:
            print(f"Schema check also failed: {e2}")

driver.close() 