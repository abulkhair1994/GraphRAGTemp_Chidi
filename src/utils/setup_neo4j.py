"""
Neo4j Setup Module

This utility module provides functions to set up and configure Neo4j 
for the GraphRAG application. It handles connection testing and vector index creation.
"""

import sys
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import utilities - use relative imports since we're in the utils package
from .env_manager import EnvManager
from .neo4j_utils import test_connection, setup_vector_index

def setup_neo4j(interactive=True):
    """Set up Neo4j for GraphRAG.
    
    Args:
        interactive: Whether to prompt for user input (default: True)
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Get Neo4j credentials
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    logger.info("Starting Neo4j setup for GraphRAG...")
    
    # Step 1: Test connection
    logger.info("Testing connection to Neo4j...")
    connection_successful = test_connection(
        uri=neo4j_creds["uri"],
        username=neo4j_creds["username"],
        password=neo4j_creds["password"],
        database=neo4j_creds["database"]
    )
    
    if not connection_successful:
        logger.error("Connection to Neo4j failed. Please check your credentials.")
        return False
    
    logger.info("Connection to Neo4j successful!")
    
    # Step 2: Set up vector index
    logger.info("Setting up vector index...")
    
    # Set default values
    index_name = "document_embeddings"
    node_label = "Document"
    embedding_property = "embedding"
    dimensions = 1536
    
    # If interactive, prompt for settings
    if interactive:
        print("\nVector Index Settings:")
        index_name = input(f"Index name (default: {index_name}): ") or index_name
        node_label = input(f"Node label (default: {node_label}): ") or node_label
        embedding_property = input(f"Embedding property (default: {embedding_property}): ") or embedding_property
        dimensions_input = input(f"Embedding dimensions (default: {dimensions}): ") or str(dimensions)
        dimensions = int(dimensions_input)
    
    # Create vector index
    setup_successful = setup_vector_index(
        uri=neo4j_creds["uri"],
        username=neo4j_creds["username"],
        password=neo4j_creds["password"],
        database=neo4j_creds["database"],
        index_name=index_name,
        node_label=node_label,
        embedding_property=embedding_property,
        dimensions=dimensions
    )
    
    if not setup_successful:
        logger.error("Vector index setup failed.")
        return False
    
    logger.info(f"Vector index '{index_name}' is set up on {node_label} nodes.")
    logger.info(f"This index will use the '{embedding_property}' property for vector search.")
    
    logger.info("Neo4j setup complete!")
    return True

# Direct execution for command-line usage
if __name__ == "__main__":
    from env_manager import load_env_vars
    
    # Load environment variables
    load_env_vars()
    
    print("\n===== Neo4j Setup Utility =====\n")
    
    success = setup_neo4j(interactive=True)
    
    if success:
        print("\n✅ Setup complete! Your Neo4j database is now ready for GraphRAG.")
        print("\nNext steps:")
        print("1. Run the GraphRAG test script: python examples/graphrag_test.py")
        print("2. Try the full example: python examples/graphrag_retrieval_example.py")
    else:
        print("\n❌ Setup failed. Please check the logs for details.") 