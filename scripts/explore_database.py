"""
Neo4j Database Explorer Example

This example shows how to use the DatabaseExplorer class to explore
your Neo4j database contents for GraphRAG development.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import DatabaseExplorer and environment utilities
from src.retrieval import DatabaseExplorer
from src.utils.env_manager import load_env_vars, EnvManager

def main():
    """Run database exploration example."""
    # Load environment variables
    load_env_vars()
    
    # Get Neo4j credentials
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    print("\n🔍 Using DatabaseExplorer to explore Neo4j...")
    
    # Create DatabaseExplorer instance
    explorer = DatabaseExplorer(
        url=neo4j_creds["uri"],
        username=neo4j_creds["username"],
        password=neo4j_creds["password"],
        database=neo4j_creds["database"]
    )
    
    # Explore the database (with automatic console output)
    results = explorer.explore_database(print_output=True)
    
    # Access specific results if needed
    labels = results["schema"]["labels"]
    print(f"\n📈 Found {len(labels)} node labels in database")
    
    # You can also get specific data programmatically
    if "Document" in labels:
        document_samples = explorer.get_sample_nodes("Document", limit=2)
        print(f"\n📄 Retrieved {len(document_samples)} Document node samples programmatically")
    
    print("\n✅ Database exploration complete!")

if __name__ == "__main__":
    main() 