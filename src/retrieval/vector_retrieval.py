"""
Vector Retrieval Engine for GraphRAG.

This module implements semantic search using vector embeddings stored in Neo4j.
It converts document chunks to vector embeddings and retrieves information 
based on semantic similarity.
"""

import logging
from typing import Dict, List, Optional, Any

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Neo4jVector
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class VectorRetrievalEngine:
    """Vector-based retrieval engine using Neo4j vector indexes."""
    
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        database: str = "neo4j",
        embedding_model: Optional[Embeddings] = None,
        index_name: str = "document_embeddings",
        text_node_property: str = "page_content",
        embedding_node_property: str = "embedding",
        embedding_dimension: int = 1536,
        distance_metric: str = "cosine",
        node_label: str = "Document"
    ):
        """Initialize the Vector Retrieval Engine.
        
        Args:
            url: Neo4j connection URL
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            embedding_model: LangChain embedding model (OpenAI, Cohere, etc.)
            index_name: Name of the vector index in Neo4j
            text_node_property: Node property containing the text
            embedding_node_property: Node property containing the embeddings
            embedding_dimension: Dimension of the embeddings
            distance_metric: Distance metric for similarity search ('cosine', 'euclidean')
            node_label: Label of nodes to search for embeddings (default: "Document")
        """
        self.url = url
        self.username = username
        self.password = password
        self.database = database
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.text_node_property = text_node_property
        self.embedding_node_property = embedding_node_property
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        self.node_label = node_label
        
        self.vector_store = None
        
    def get_document_node_labels(self):
        """Find all node labels that have text content suitable for embedding."""
        driver = GraphDatabase.driver(
            self.url, 
            auth=(self.username, self.password)
        )
        
        try:
            with driver.session(database=self.database) as session:
                # Look for nodes that have the text property we're looking for
                result = session.run(f"""
                    MATCH (n) 
                    WHERE n.{self.text_node_property} IS NOT NULL 
                    WITH DISTINCT labels(n) AS labels
                    RETURN labels
                    LIMIT 10
                """)
                
                all_labels = []
                for record in result:
                    all_labels.append(record["labels"])
                
                # If we found labels with the text property
                if all_labels:
                    logger.info(f"Found node labels with {self.text_node_property} property: {all_labels}")
                    # Return the first label from each list for simplicity
                    return [labels[0] if labels else None for labels in all_labels if labels]
                else:
                    logger.warning(f"No nodes found with property {self.text_node_property}")
                    return []
        except Exception as e:
            logger.error(f"Error finding document node labels: {e}")
            return []
        finally:
            driver.close()
        
    def create_vector_index(self):
        """Create the vector index in Neo4j if it doesn't exist."""
        # This uses the Neo4j Cypher query to create a vector index
        # Implementation depends on Neo4j version
        # For Neo4j 5.x:
        driver = GraphDatabase.driver(
            self.url, 
            auth=(self.username, self.password)
        )
        
        try:
            # First check if we have nodes with the expected properties
            with driver.session(database=self.database) as session:
                # Count nodes with the label and embedding property
                result = session.run(f"""
                    MATCH (d:{self.node_label}) 
                    WHERE d.{self.embedding_node_property} IS NOT NULL 
                    RETURN COUNT(d) as count
                """)
                count = result.single()["count"] if result.peek() else 0
                
                if count == 0:
                    # Check if we have the nodes but no embeddings
                    result = session.run(f"""
                        MATCH (d:{self.node_label}) 
                        RETURN COUNT(d) as count
                    """)
                    node_count = result.single()["count"] if result.peek() else 0
                    
                    if node_count > 0:
                        logger.warning(f"Found {node_count} nodes with label {self.node_label} but none have embeddings")
                    else:
                        logger.warning(f"No nodes found with label {self.node_label}")
                        
                        # Try to detect what node labels might have content
                        doc_labels = self.get_document_node_labels()
                        if doc_labels:
                            self.node_label = doc_labels[0]
                            logger.info(f"Switching to use node label {self.node_label} for vector index")
                
                # Create the vector index with the determined node label
                create_index_query = f"""
                CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
                FOR (d:{self.node_label})
                ON (d.{self.embedding_node_property})
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: '{self.distance_metric}'
                }}}}
                """
                
                logger.info(f"Creating vector index {self.index_name} for {self.node_label} nodes in Neo4j")
                result = session.run(create_index_query)
                logger.info(f"Vector index creation result: {result.consume().counters}")
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            raise
        finally:
            driver.close()
            
        logger.info(f"Vector index {self.index_name} created or already exists")
        
    def init_vector_store(self):
        """Initialize the vector store connection."""
        if not self.embedding_model:
            raise ValueError("Embedding model must be provided")
        
        # Create the vector index first
        self.create_vector_index()
            
        try:
            self.vector_store = Neo4jVector(
                embedding=self.embedding_model,
                url=self.url,
                username=self.username,
                password=self.password,
                database=self.database,
                index_name=self.index_name,
                node_label=self.node_label,
                text_node_property=self.text_node_property,
                embedding_node_property=self.embedding_node_property,
            )
            logger.info(f"Vector store initialized with node label {self.node_label}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def check_document_existence(self) -> bool:
        """Check if documents with embeddings exist in the database."""
        driver = GraphDatabase.driver(
            self.url, 
            auth=(self.username, self.password)
        )
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(f"""
                    MATCH (d:{self.node_label}) 
                    WHERE d.{self.embedding_node_property} IS NOT NULL 
                    RETURN COUNT(d) as count
                """)
                count = result.single()["count"] if result.peek() else 0
                
                if count > 0:
                    logger.info(f"Found {count} {self.node_label} nodes with embeddings")
                    return True
                else:
                    logger.warning(f"No {self.node_label} nodes with embeddings found")
                    return False
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False
        finally:
            driver.close()
    
    def add_documents(self, documents: List[Document], **kwargs):
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            **kwargs: Additional arguments to pass to the add_documents method
        
        Returns:
            List of IDs of the added documents
        """
        if not self.vector_store:
            self.init_vector_store()
            
        return self.vector_store.add_documents(documents, **kwargs)
    
    def similarity_search(self, query: str, k: int = 4, **kwargs):
        """Perform similarity search against the vector store.
        
        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments for the similarity search
            
        Returns:
            List of similar documents with their scores
        """
        if not self.vector_store:
            self.init_vector_store()
        
        # Check if we have documents with embeddings
        if not self.check_document_existence():
            logger.warning("No documents with embeddings found in the database. Vector search returning empty results.")
            return []
            
        try:
            return self.vector_store.similarity_search(query, k=k, **kwargs)
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        """Perform similarity search with scores against the vector store.
        
        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments for the similarity search
            
        Returns:
            List of tuples (document, score)
        """
        if not self.vector_store:
            self.init_vector_store()
            
        # Check if we have documents with embeddings
        if not self.check_document_existence():
            logger.warning("No documents with embeddings found in the database. Vector search returning empty results.")
            return []
            
        try:
            return self.vector_store.similarity_search_with_score(query, k=k, **kwargs)
        except Exception as e:
            logger.error(f"Error in similarity search with score: {e}")
            return []
    
    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs
    ):
        """Perform max marginal relevance search against the vector store.
        
        This helps reduce redundancy in search results by balancing relevance with diversity.
        
        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of results to consider before reranking
            lambda_mult: Controls tradeoff between relevance and diversity (0-1)
            **kwargs: Additional arguments for the search
            
        Returns:
            List of documents
        """
        if not self.vector_store:
            self.init_vector_store()
            
        # Check if we have documents with embeddings
        if not self.check_document_existence():
            logger.warning("No documents with embeddings found in the database. Vector search returning empty results.")
            return []
            
        try:
            return self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
            )
        except Exception as e:
            logger.error(f"Error in max marginal relevance search: {e}")
            return [] 