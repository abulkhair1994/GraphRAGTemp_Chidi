#!/usr/bin/env python3
"""
Fix the vector dimension mismatch between OpenAI embeddings (1536) and fastRP embeddings (512).

This module uses the DatabaseExplorer to automatically detect and fix the issue
by either:
1. Copying fastRP_embedding values to the embedding property expected by the index, or
2. Recreating the vector index to directly use fastRP_embedding with proper dimensions

This should be run before using the GraphRAG retriever to ensure proper vector search.
"""

import os
import sys
from typing import Dict, Any

# Import from project modules
from src.retrieval.database_explorer import DatabaseExplorer
from src.utils.env_manager import load_env_vars, EnvManager

def fix_embedding_mismatch() -> Dict[str, Any]:
    """Fix the dimension mismatch between OpenAI embeddings and fastRP embeddings in Neo4j.
    
    Returns:
        Dictionary containing the results of the fix operation
    """
    print("\n🔧 FIXING EMBEDDING DIMENSION MISMATCH...")
    
    # Load environment variables
    load_env_vars()
    
    # Get Neo4j credentials
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    # Create explorer instance
    explorer = DatabaseExplorer(
        url=neo4j_creds["uri"],
        username=neo4j_creds["username"],
        password=neo4j_creds["password"],
        database=neo4j_creds["database"]
    )
    
    # Check and fix fastRP embeddings with auto_fix=True
    results = explorer.check_fastrp_embeddings(auto_fix=True)
    
    if results.get("fix_applied") == "option1":
        print(f"\n✅ Success! Applied fix option 1: Copied fastRP_embedding to embedding for {results.get('updated_nodes', 0)} nodes")
    elif results.get("fix_applied") == "option2":
        print(f"\n✅ Success! Applied fix option 2: Recreated vector index '{results.get('index_name')}' to use fastRP_embedding")
    elif results.get("fix_applied") == "none_needed":
        print("\n✅ No fix needed: Vector dimensions already compatible")
    else:
        print("\n❌ Could not apply any fixes automatically")
        
    print("\n📊 Database embeddings status after fix:")
    explorer.print_embedding_properties()
    
    print("\n🔍 Vector index status after fix:")
    vector_indexes = explorer.get_vector_indexes()
    if vector_indexes:
        for idx in vector_indexes:
            print(f"  - Name: {idx['name']}")
            print(f"    Labels: {idx['labels']}")
            print(f"    Properties: {idx['properties']}")
    else:
        print("  No vector indexes found.")
    
    print("\nDone! Vector search should now work correctly in GraphRAG retriever.")
    return results


if __name__ == "__main__":
    fix_embedding_mismatch() 