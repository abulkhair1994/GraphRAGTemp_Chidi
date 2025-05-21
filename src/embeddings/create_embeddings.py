"""
Create FastRP embeddings for all nodes in the Neo4j database.

This module provides functionality to generate and store FastRP embeddings
for all nodes in a Neo4j database using the Graph Data Science library.
"""

import os
import sys
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from graphdatascience import GraphDataScience
from src.utils.env_manager import load_env_vars, EnvManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def remove_old_embeddings(gds: GraphDataScience, property_name: str) -> None:
    """Remove old FastRP embeddings from all nodes.
    
    Args:
        gds: GraphDataScience instance
        property_name: Name of the embedding property to remove
    """
    logger.info(f"Removing old embeddings ({property_name})...")
    
    cleanup_query = f"""
    MATCH (n)
    WHERE n.{property_name} IS NOT NULL
    WITH count(n) as total_nodes
    CALL {{
        MATCH (n)
        WHERE n.{property_name} IS NOT NULL
        REMOVE n.{property_name}
        RETURN count(n) as cleaned_nodes
    }}
    RETURN total_nodes, cleaned_nodes
    """
    
    try:
        result = gds.run_cypher(cleanup_query)
        total = result.iloc[0]['total_nodes']
        cleaned = result.iloc[0]['cleaned_nodes']
        logger.info(f"Removed embeddings from {cleaned} nodes (found {total} nodes with embeddings)")
    except Exception as e:
        logger.warning(f"Error while removing old embeddings: {str(e)}")

def create_text_features(gds: GraphDataScience) -> None:
    """Create numeric features from text content across all node types."""
    text_feature_query = """
    MATCH (n)
    WITH n, 
         CASE WHEN n.title IS NOT NULL THEN size(n.title) ELSE 0 END AS title_length,
         CASE WHEN n.content IS NOT NULL THEN size(n.content) ELSE 0 END AS content_length,
         CASE WHEN n.text_content IS NOT NULL THEN size(n.text_content) ELSE 0 END AS text_length,
         CASE WHEN n.problem_text IS NOT NULL THEN size(n.problem_text) ELSE 0 END AS problem_length,
         CASE WHEN n.solution_text IS NOT NULL THEN size(n.solution_text) ELSE 0 END AS solution_length,
         // Add word counts for richer text features
         CASE WHEN n.text_content IS NOT NULL 
              THEN size(split(n.text_content, ' ')) 
              ELSE 0 
         END AS word_count,
         // Add special character counts
         CASE WHEN n.text_content IS NOT NULL 
              THEN size(apoc.text.regexGroups(n.text_content, '[^a-zA-Z0-9\\s]')) 
              ELSE 0 
         END AS special_chars
    SET n.text_features = [
        title_length, 
        content_length, 
        text_length, 
        problem_length, 
        solution_length,
        word_count,
        special_chars
    ]
    RETURN count(n) as processed_count
    """
    try:
        result = gds.run_cypher(text_feature_query)
        logger.info(f"Created text features for {result.iloc[0]['processed_count']} nodes")
    except Exception as e:
        logger.warning(f"Could not create text features: {str(e)}")

def create_structural_features(gds: GraphDataScience) -> None:
    """Create features based on node structure and relationships."""
    structural_query = """
    MATCH (n)
    WITH n
    // Count all relationships
    OPTIONAL MATCH (n)-[r]-()
    WITH n, count(r) as total_degree
    
    // Count specific relationship types
    OPTIONAL MATCH (n)-[:CONTAINS]->(child)
    WITH n, total_degree, count(child) as num_children
    
    OPTIONAL MATCH (parent)-[:CONTAINS]->(n)
    WITH n, total_degree, num_children, count(parent) as num_parents
    
    // Count RELATED relationships
    OPTIONAL MATCH (n)-[:RELATED]-()
    WITH n, total_degree, num_children, num_parents, count(*) as related_count
    
    // Calculate hierarchical level
    OPTIONAL MATCH path = (root)-[:CONTAINS*]->(n)
    WHERE NOT ()-[:CONTAINS]->(root)
    WITH n, total_degree, num_children, num_parents, related_count,
         CASE WHEN path IS NULL THEN 0 ELSE length(path) END as hierarchy_level
    
    SET 
        n.node_index = CASE WHEN n.index IS NOT NULL THEN toInteger(n.index) ELSE -1 END,
        n.structural_features = [
            total_degree,
            num_children,
            num_parents,
            related_count,
            hierarchy_level
        ]
    RETURN count(n) as processed_count
    """
    try:
        result = gds.run_cypher(structural_query)
        logger.info(f"Created structural features for {result.iloc[0]['processed_count']} nodes")
    except Exception as e:
        logger.warning(f"Could not create structural features: {str(e)}")

def create_fastRP_embeddings(
    embedding_dim: int = 512,
    property_ratio: float = 1.0,
    random_seed: int = 42,
    iteration_weights: List[float] = [0.0, 0.25, 0.5, 1.0],
    write_property: str = "fastRP_embedding"
) -> Dict[str, Any]:
    """
    Create FastRP embeddings for all nodes in the Neo4j database.
    
    Args:
        embedding_dim: Dimension of the embeddings. Default is 512.
        property_ratio: Ratio of property-based features. Default is 1.0.
        random_seed: Random seed for reproducibility. Default is 42.
        iteration_weights: Weights for each iteration.
        write_property: Name of the property to store the embeddings.
        
    Returns:
        dict: Result information from the FastRP algorithm including statistics.
    """
    # Load environment variables
    load_env_vars()
    neo4j_creds = EnvManager.get_neo4j_credentials()

    logger.info("Connecting to Neo4j...")
    gds = GraphDataScience(
        neo4j_creds["uri"],
        auth=(neo4j_creds["username"], neo4j_creds["password"]),
        database=neo4j_creds["database"],
        aura_ds=True
    )
    logger.info(f"Connected to Neo4j: {gds.database()}")

    # Remove old embeddings first
    remove_old_embeddings(gds, write_property)

    # Create features
    create_text_features(gds)
    create_structural_features(gds)

    # Define properties to use for embedding
    numeric_properties = [
        'text_features',          # Text-based features
        'structural_features',    # Structural features
        'node_index'             # Original index if available
    ]
    
    logger.info("Projecting graph...")
    try:
        # Project the graph with all relationships
        gds.graph.project(
            'completeGraph',
            '*',  # Include all node labels
            {
                'CONTAINS': {
                    'type': 'CONTAINS',
                    'orientation': 'UNDIRECTED',
                    'aggregation': 'NONE'
                },
                'RELATED': {
                    'type': 'RELATED',
                    'orientation': 'UNDIRECTED',
                    'aggregation': 'NONE'
                }
            },
            nodeProperties=numeric_properties
        )

        G = gds.graph.get('completeGraph')
        graph_stats = gds.graph.list('completeGraph').iloc[0]
        logger.info(f"Graph projected with {graph_stats['nodeCount']} nodes and {graph_stats['relationshipCount']} relationships")

        # Run FastRP with optimized parameters
        logger.info("Running FastRP algorithm...")
        result = gds.fastRP.write(
            G,
            writeProperty=write_property,
            embeddingDimension=embedding_dim,
            propertyRatio=property_ratio,
            randomSeed=random_seed,
            featureProperties=numeric_properties,
            iterationWeights=iteration_weights
        )

        logger.info("FastRP embeddings created successfully!")
        logger.info(f"Node properties written: {result['nodePropertiesWritten']}")
        logger.info(f"Run time: {result['computeMillis']/1000:.2f} seconds")

        # Show example queries
        logger.info("\nExample Cypher queries for using the embeddings:")
        logger.info("\n1. Find similar content across any node type:")
        print(f"""
        MATCH (n)
        WHERE n.id = $nodeId
        CALL {{
            WITH n
            MATCH (other)
            WHERE other <> n
            WITH other, 
                 gds.similarity.cosine(n.{write_property}, other.{write_property}) AS similarity,
                 labels(other) as otherLabels
            RETURN other, similarity, otherLabels
            ORDER BY similarity DESC
            LIMIT 5
        }}
        RETURN 
            otherLabels[0] as type,
            other.title as title,
            other.id as id,
            similarity
        ORDER BY similarity DESC;
        """)

        # Verify embeddings with statistics
        logger.info("\nEmbedding statistics:")
        stats_query = f"""
        MATCH (n)
        WHERE n.{write_property} IS NOT NULL
        WITH 
            labels(n) as label,
            count(n) as count,
            avg(size(n.{write_property})) as avg_dim
        RETURN label, count, avg_dim
        ORDER BY count DESC
        LIMIT 5
        """
        stats = gds.run_cypher(stats_query)
        print(stats)

        return result
        
    finally:
        try:
            logger.info("Cleaning up...")
            gds.graph.drop('completeGraph')
            logger.info("Done!")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    create_fastRP_embeddings() 