"""Test Neo4j connection using environment variables.

This script tests connecting to Neo4j using credentials from environment
variables loaded from the .env file.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import environment utilities
from src.utils.env_manager import load_env_vars, EnvManager

def test_neo4j_connection():
    """Test connection to Neo4j database using environment variables."""
    try:
        # Load environment variables
        load_env_vars()
        
        # Import Neo4j driver
        from neo4j import GraphDatabase
        
        # Get Neo4j credentials from environment
        neo4j_creds = EnvManager.get_neo4j_credentials()
        
        # Print connection details (obscuring password)
        print("\nNeo4j Connection Details:")
        print(f"  URI: {neo4j_creds['uri']}")
        print(f"  Username: {neo4j_creds['username']}")
        print(f"  Password: {'*' * (len(neo4j_creds['password']) if neo4j_creds['password'] else 0)}")
        print(f"  Database: {neo4j_creds['database']}")
        
        # Check if credentials are valid
        if not neo4j_creds['password']:
            print("❌ NEO4J_PASSWORD is not set in environment variables")
            print("   Make sure you have a .env file in the project root with NEO4J_PASSWORD specified")
            return False
        
        # Connect to Neo4j
        with GraphDatabase.driver(neo4j_creds['uri'], auth=(neo4j_creds['username'], neo4j_creds['password'])) as driver:
            # Verify connectivity
            driver.verify_connectivity()
            print("✅ Connected to Neo4j server")
            
            # Test database connection
            with driver.session(database=neo4j_creds['database']) as session:
                result = session.run("RETURN 'Connection successful!' AS message")
                message = result.single()["message"]
                print(f"✅ {message}")
                
                # Create test node
                session.run("""
                CREATE (n:Test {
                    name: 'GraphRAG Test Node',
                    created: datetime()
                })
                RETURN n
                """)
                print("✅ Created test node")
                
        return True
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    test_neo4j_connection() 