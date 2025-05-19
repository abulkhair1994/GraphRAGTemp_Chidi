"""
Neo4j utilities for GraphRAG.

This module provides helper functions for Neo4j operations,
including vector index creation and management.
"""

import logging
from neo4j import GraphDatabase, Driver

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_index(driver, index_name, label, embedding_property, dimensions, similarity_fn="cosine"):
    """Create a vector index in Neo4j if it doesn't exist.
    
    Args:
        driver: Neo4j driver instance
        index_name: Name of the vector index
        label: Node label to create the index on
        embedding_property: Property name storing the embeddings
        dimensions: Dimensionality of the embeddings
        similarity_fn: Similarity function to use ('cosine', 'euclidean')
        
    Returns:
        bool: True if index was created, False if it already existed
    """
    with driver.session() as session:
        # Check if the index already exists
        index_check_query = """
        SHOW INDEXES
        YIELD name, type
        WHERE name = $index_name
        RETURN count(*) as count
        """
        
        result = session.run(index_check_query, index_name=index_name)
        count = result.single()["count"]
        
        if count > 0:
            logger.info(f"Vector index {index_name} already exists")
            return False
            
        # Create the vector index
        create_index_query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (d:{label})
        ON (d.{embedding_property})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: '{similarity_fn}'
            }}
        }}
        """
        
        session.run(create_index_query)
        logger.info(f"Vector index {index_name} created successfully")
        return True

def drop_vector_index(driver, index_name):
    """Drop a vector index if it exists.
    
    Args:
        driver: Neo4j driver instance
        index_name: Name of the vector index to drop
        
    Returns:
        bool: True if index was dropped, False if it didn't exist
    """
    with driver.session() as session:
        # Check if the index exists
        index_check_query = """
        SHOW INDEXES
        YIELD name, type
        WHERE name = $index_name
        RETURN count(*) as count
        """
        
        result = session.run(index_check_query, index_name=index_name)
        count = result.single()["count"]
        
        if count == 0:
            logger.info(f"Vector index {index_name} does not exist")
            return False
            
        # Drop the index
        drop_index_query = f"""
        DROP INDEX {index_name}
        """
        
        session.run(drop_index_query)
        logger.info(f"Vector index {index_name} dropped successfully")
        return True

def test_connection(uri, username, password, database="neo4j"):
    """Test connection to Neo4j database.
    
    Args:
        uri: Neo4j connection URI
        username: Neo4j username
        password: Neo4j password
        database: Neo4j database name
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session(database=database) as session:
            session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
            driver.close()
            return True
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        return False

def setup_vector_index(uri, username, password, database="neo4j", 
                      index_name="document_embeddings", 
                      node_label="Document", 
                      embedding_property="embedding", 
                      dimensions=1536,
                      similarity_fn="cosine"):
    """Set up a vector index in Neo4j.
    
    Args:
        uri: Neo4j connection URI
        username: Neo4j username
        password: Neo4j password
        database: Neo4j database name
        index_name: Name of the vector index
        node_label: Node label to create the index on
        embedding_property: Property name storing the embeddings
        dimensions: Dimensionality of the embeddings
        similarity_fn: Similarity function to use ('cosine', 'euclidean')
        
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connection
        with driver.session(database=database) as session:
            session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")
        
        # Create the vector index
        created = create_vector_index(
            driver=driver,
            index_name=index_name,
            label=node_label,
            embedding_property=embedding_property,
            dimensions=dimensions,
            similarity_fn=similarity_fn
        )
        
        # Close the driver
        driver.close()
        
        logger.info(f"Vector index setup {'completed' if created else 'already exists'}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up vector index: {str(e)}")
        return False 