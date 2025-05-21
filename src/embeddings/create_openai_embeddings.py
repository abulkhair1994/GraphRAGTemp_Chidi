"""
Create OpenAI embeddings for all nodes with text content in the Neo4j database.

This module provides functionality to generate and store OpenAI embeddings
for all nodes that have text content in a Neo4j database.
"""

import os
import sys
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings
from src.utils.env_manager import load_env_vars, EnvManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_text_properties() -> List[str]:
    """Get list of potential text properties to check for content."""
    return [
        'text_content',
        'title',
        'content',
        'problem_text',
        'solution_text',
        'label',
        'description'
    ]

def get_text_content(node: Dict[str, Any]) -> Optional[str]:
    """Extract text content from a node using available text properties.
    
    Args:
        node: Neo4j node dictionary
        
    Returns:
        Combined text content or None if no text found
    """
    text_parts = []
    
    # Check each potential text property
    for prop in get_text_properties():
        if prop in node and node[prop]:
            text_parts.append(str(node[prop]))
    
    # Return combined text if any found
    return ' '.join(text_parts) if text_parts else None

def create_openai_embeddings(
    batch_size: int = 100,
    max_workers: int = 4,
    write_property: str = "openai_embedding"
) -> Dict[str, Any]:
    """
    Create OpenAI embeddings for all nodes with text content in the Neo4j database.
    
    Args:
        batch_size: Number of nodes to process in each batch
        max_workers: Maximum number of parallel workers for embedding generation
        write_property: Name of the property to store the embeddings
        
    Returns:
        dict: Result information including statistics
    """
    # Load environment variables
    load_env_vars()
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        neo4j_creds["uri"],
        auth=(neo4j_creds["username"], neo4j_creds["password"])
    )
    
    try:
        # Get total count of nodes with text content
        with driver.session(database=neo4j_creds["database"]) as session:
            # Build dynamic query to check all text properties
            text_props = get_text_properties()
            where_clause = " OR ".join([f"n.{prop} IS NOT NULL" for prop in text_props])
            
            result = session.run(f"""
                MATCH (n)
                WHERE {where_clause}
                RETURN COUNT(n) as count
            """)
            total_nodes = result.single()["count"]
            
            if total_nodes == 0:
                logger.warning("No nodes found with text content")
                return {"error": "No nodes with text content found"}
            
            logger.info(f"Found {total_nodes} nodes with text content")
            
            # Process nodes in batches
            processed = 0
            failed = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Process nodes in batches
                for offset in range(0, total_nodes, batch_size):
                    # Get batch of nodes
                    result = session.run(f"""
                        MATCH (n)
                        WHERE {where_clause}
                        RETURN n
                        SKIP {offset}
                        LIMIT {batch_size}
                    """)
                    
                    nodes = [record["n"] for record in result]
                    
                    # Generate embeddings in parallel
                    futures = []
                    for node in nodes:
                        text = get_text_content(node)
                        if text:
                            futures.append(
                                executor.submit(embeddings.embed_query, text)
                            )
                    
                    # Update nodes with embeddings
                    for node, future in zip(nodes, futures):
                        try:
                            embedding = future.result()
                            node_id = node.id
                            
                            # Update node with embedding
                            session.run(f"""
                                MATCH (n)
                                WHERE id(n) = {node_id}
                                SET n.{write_property} = $embedding
                            """, embedding=embedding)
                            
                            processed += 1
                        except Exception as e:
                            logger.error(f"Error processing node {node.id}: {str(e)}")
                            failed += 1
                    
                    logger.info(f"Processed {processed}/{total_nodes} nodes (failed: {failed})")
            
            # Create vector index for the embeddings
            try:
                session.run(f"""
                    CREATE VECTOR INDEX {write_property}_index IF NOT EXISTS
                    FOR (n)
                    ON (n.{write_property})
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: 1536,
                            `vector.similarity`: 'cosine'
                        }}
                    }}
                """)
                logger.info(f"Created vector index {write_property}_index")
            except Exception as e:
                logger.error(f"Error creating vector index: {str(e)}")
            
            return {
                "total_nodes": total_nodes,
                "processed": processed,
                "failed": failed,
                "write_property": write_property
            }
            
    finally:
        driver.close()

if __name__ == "__main__":
    create_openai_embeddings() 