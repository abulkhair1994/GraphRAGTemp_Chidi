"""
Neo4j Setup Example

This script demonstrates how to use the Neo4j setup utility from src/utils.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import utilities
from src.utils.env_manager import load_env_vars
from src.utils.setup_neo4j import setup_neo4j

def main():
    """Run the Neo4j setup example."""
    # Load environment variables
    load_env_vars()
    
    print("\n===== Neo4j Setup Example =====\n")
    
    # Run the setup with interactive prompts
    success = setup_neo4j(interactive=True)
    
    if success:
        print("\n✅ Example complete! Your Neo4j database is now ready for GraphRAG.")
        print("\nNext steps:")
        print("1. Run the GraphRAG test script: python examples/graphrag_test.py")
        print("2. Try the full example: python examples/graphrag_retrieval_example.py")
    else:
        print("\n❌ Example failed. Please check the logs for details.")

if __name__ == "__main__":
    main() 