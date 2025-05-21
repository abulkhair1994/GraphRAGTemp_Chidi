"""
GraphRAG Retriever for combining vector and graph-based retrieval.

This module provides a hybrid retriever that combines both vector similarity search
and graph-based retrieval for enhanced context retrieval in RAG applications.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

from langchain.schema import Document
from langchain.schema.embeddings import Embeddings

from src.retrieval.vector_retrieval import VectorRetrievalEngine
from src.retrieval.graph_retrieval import GraphRetrievalEngine, GraphResult
from src.retrieval.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

@dataclass
class GraphRAGResult:
    """Combined result from GraphRAG retrieval."""
    vector_documents: List[Document]
    graph_result: GraphResult
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    
    def to_text(self) -> str:
        """Convert the combined result to a textual representation."""
        text_parts = []
        
        # Add vector results
        text_parts.append("Vector Search Results:")
        for i, doc in enumerate(self.vector_documents, 1):
            text_parts.append(f"Document {i}:")
            text_parts.append(f"  {doc.page_content[:200]}...")
            text_parts.append(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            text_parts.append("")
        
        # Add graph results
        text_parts.append("Graph Search Results:")
        text_parts.append(self.graph_result.to_text())
        
        return "\n".join(text_parts)


class GraphRAGRetriever:
    """Hybrid retriever combining vector, graph, and text search."""
    
    def __init__(
        self,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        embedding_model: Optional[Embeddings] = None,
        vector_text_node_property: str = "text_content",
        vector_embedding_node_property: str = "fastRP_embedding",
        vector_embedding_dimension: int = 512,
        vector_distance_metric: str = "cosine",
        vector_node_label: str = "Content",
        vector_search_type: str = "hybrid",
        graph_max_hops: int = 2,
        graph_max_nodes: int = 10,
        text_search_threshold: float = 0.7,
        vector_top_k: int = 3,
        text_top_k: int = 3
    ):
        """Initialize the GraphRAG Retriever.
        
        Args:
            neo4j_url: Neo4j connection URL
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            embedding_model: Embedding model for generating embeddings
            vector_text_node_property: Property containing text for vector search
            vector_embedding_node_property: Property containing embeddings
            vector_embedding_dimension: Dimension of embeddings
            vector_distance_metric: Distance metric for similarity
            vector_node_label: Label of document nodes
            vector_search_type: Type of search to perform ('hybrid', 'vector', 'text')
            graph_max_hops: Maximum number of hops for graph traversal
            graph_max_nodes: Maximum number of nodes to retrieve
            text_search_threshold: Similarity threshold for text search
            vector_top_k: Number of top vector results to return
            text_top_k: Number of top text results to return
        """
        self.neo4j_url = neo4j_url
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.embedding_model = embedding_model
        
        # Vector retrieval parameters
        self.vector_text_node_property = vector_text_node_property
        self.vector_embedding_node_property = vector_embedding_node_property
        self.vector_embedding_dimension = vector_embedding_dimension
        self.vector_distance_metric = vector_distance_metric
        self.vector_node_label = vector_node_label
        self.vector_search_type = vector_search_type
        self.vector_top_k = vector_top_k
        self.text_top_k = text_top_k
        
        # Graph retrieval parameters
        self.graph_max_hops = graph_max_hops
        self.graph_max_nodes = graph_max_nodes
        self.text_search_threshold = text_search_threshold
        
        # Initialize engines
        self.vector_engine = VectorRetrievalEngine(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
            embedding_model=embedding_model,
            index_name="content_embeddings",
            text_node_property=vector_text_node_property,
            embedding_node_property=vector_embedding_node_property,
            embedding_dimension=vector_embedding_dimension,
            distance_metric=vector_distance_metric,
            node_label=vector_node_label,
            search_type=vector_search_type
        )
        
        self.graph_engine = GraphRetrievalEngine(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database
        )
        
        # Initialize document processor for entity extraction
        self.document_processor = DocumentProcessor(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
        )
        
        # Get database schema information
        self.db_schema = self.graph_engine.get_db_schema()
        
        # Update the node label for vector search if not provided
        if not self.vector_node_label:
            self._discover_document_node_label()
    
    def _discover_document_node_label(self):
        """Attempt to discover the best node label for document searches."""
        if not self.db_schema["labels"]:
            logger.warning("No node labels found in database schema")
            return
            
        # Look for likely document node labels
        document_labels = []
        
        # Check for common document-related label names
        for label in self.db_schema["labels"]:
            label_lower = label.lower()
            if any(term in label_lower for term in ["document", "content", "page", "text", "chunk"]):
                document_labels.append(label)
                
        # Check for nodes with text content properties
        if not document_labels:
            for label, props in self.db_schema["properties"].items():
                if any(prop_name in [self.vector_text_node_property, "text", "content", "body"] for prop_name in props):
                    document_labels.append(label)
                    
        # If we found potential document labels, use the first one
        if document_labels:
            self.vector_node_label = document_labels[0]
            logger.info(f"Discovered document node label: {self.vector_node_label}")
            
            # Update the vector engine node label
            self.vector_engine.node_label = self.vector_node_label
    
    def add_documents(self, documents: List[Document], **kwargs):
        """Add documents to the vector store and extract entities for the graph.
        
        Args:
            documents: List of documents to add
            **kwargs: Additional arguments to pass to the add_documents method
        
        Returns:
            List of IDs of the added documents
        """
        # First add documents to vector store for similarity search
        try:
            vector_doc_ids = self.vector_engine.add_documents(documents, **kwargs)
            
            # Then process documents to extract entities and relationships for graph search
            graph_doc_ids = self.document_processor.process_documents(documents)
            
            # Return the document IDs from vector store
            return vector_doc_ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []
    
    def retrieve(
        self, 
        query: str,
        vector_k: Optional[int] = None,
        graph_entity_types: Optional[List[str]] = None,
        graph_relationship_types: Optional[List[str]] = None
    ) -> GraphRAGResult:
        """Perform hybrid retrieval using both vector and graph-based methods.
        
        Args:
            query: Query text
            vector_k: Number of vector results to return (overrides default)
            graph_entity_types: Types of entities to include in graph search
            graph_relationship_types: Types of relationships to include
            
        Returns:
            GraphRAGResult containing vector and graph results
        """
        vector_docs = []
        graph_result = GraphResult(entities=[], relationships=[], paths=[])
        
        # Retrieve from vector store
        try:
            logger.info(f"Performing vector search for query: {query}")
            vector_docs = self.vector_engine.similarity_search(
                query, top_k=vector_k or self.vector_top_k
            )
            if vector_docs:
                logger.info(f"Found {len(vector_docs)} documents via vector search")
            else:
                logger.warning("No documents found via vector search")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        # If no graph entity types provided, try to discover them from database schema
        if not graph_entity_types:
            # Use the most populated entity types
            sorted_labels = sorted(self.db_schema["label_counts"].items(), key=lambda x: x[1], reverse=True)
            graph_entity_types = [label for label, count in sorted_labels[:3]]
            logger.info(f"Using discovered entity types for graph retrieval: {graph_entity_types}")
            
        # Retrieve from graph
        try:
            logger.info(f"Performing graph retrieval for query: {query}")
            graph_result = self.graph_engine.knowledge_graph_query(
                query_text=query,
                entity_types=graph_entity_types,
                relationship_types=graph_relationship_types,
                limit=self.graph_max_nodes
            )
            
            if graph_result.entities or graph_result.relationships:
                logger.info(f"Found {len(graph_result.entities)} entities and {len(graph_result.relationships)} relationships via graph search")
            else:
                logger.warning("No graph results found for query")
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            graph_result = GraphResult(entities=[], relationships=[], paths=[])
        
        # Return combined results
        return GraphRAGResult(
            vector_documents=vector_docs,
            graph_result=graph_result
        )
    
    def retrieve_with_scores(
        self,
        query: str,
        vector_k: Optional[int] = None,
        graph_entity_types: Optional[List[str]] = None,
        graph_relationship_types: Optional[List[str]] = None
    ) -> GraphRAGResult:
        """Perform hybrid retrieval with relevance scores.
        
        Args:
            query: Query text
            vector_k: Number of vector results to return (overrides default)
            graph_entity_types: Types of entities to include in graph search
            graph_relationship_types: Types of relationships to include
            
        Returns:
            GraphRAGResult containing vector and graph results with scores
        """
        vector_docs = []
        vector_avg_score = 0.0
        graph_result = GraphResult(entities=[], relationships=[], paths=[])
        graph_score = 0.0
        
        # Retrieve from vector store with scores
        try:
            logger.info(f"Performing vector search with scores for query: {query}")
            vector_results = self.vector_engine.similarity_search_with_score(
                query, top_k=vector_k or self.vector_top_k
            )
            
            # Split documents and scores
            if vector_results:
                vector_docs = [doc for doc, _ in vector_results]
                vector_avg_score = sum(score for _, score in vector_results) / len(vector_results) if vector_results else 0
                logger.info(f"Found {len(vector_docs)} documents via vector search with avg score: {vector_avg_score}")
            else:
                logger.warning("No vector results found for query")
        except Exception as e:
            logger.error(f"Vector search with scores failed: {e}")
        
        # If no graph entity types provided, try to discover them from database schema
        if not graph_entity_types:
            # Use the most populated entity types
            sorted_labels = sorted(self.db_schema["label_counts"].items(), key=lambda x: x[1], reverse=True)
            graph_entity_types = [label for label, count in sorted_labels[:3]]
            logger.info(f"Using discovered entity types for graph retrieval: {graph_entity_types}")
        
        # Retrieve from graph
        try:
            logger.info(f"Performing graph retrieval for query: {query}")
            graph_result = self.graph_engine.knowledge_graph_query(
                query_text=query,
                entity_types=graph_entity_types,
                relationship_types=graph_relationship_types,
                limit=self.graph_max_nodes
            )
            
            # Calculate a simple graph score based on number of entities and relationships
            entity_count = len(graph_result.entities)
            rel_count = len(graph_result.relationships)
            
            if entity_count > 0 or rel_count > 0:
                graph_score = (entity_count + rel_count) / self.graph_max_nodes
                logger.info(f"Found {entity_count} entities and {rel_count} relationships with score: {graph_score}")
            else:
                logger.warning("No graph results found for query")
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
        
        # Return combined results
        return GraphRAGResult(
            vector_documents=vector_docs,
            graph_result=graph_result,
            vector_score=vector_avg_score,
            graph_score=graph_score
        )
    
    def close(self):
        """Close all connections."""
        try:
            self.graph_engine.close()
            logger.info("Successfully closed all connections")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using both vector and graph-based retrieval.
        
        Args:
            query: Query string to search for
            
        Returns:
            Combined list of relevant documents
        """
        vector_docs = []
        graph_docs = []
        
        # Start with vector retrieval
        logger.info(f"Performing vector search for query: {query}")
        try:
            # First initialize the vector store if needed
            if not self.vector_engine.is_initialized():
                self.vector_engine.init_vector_store()
                
            # Perform similarity search with fallback
            vector_docs = self.vector_engine.similarity_search(query, top_k=self.vector_top_k)
            
            if vector_docs:
                # Successfully retrieved vector docs
                logger.info(f"Found {len(vector_docs)} documents via vector search")
            else:
                logger.warning("No documents found via vector search")
        except Exception as e:
            logger.warning(f"Error in vector retrieval: {e}")
            # Continue execution to try graph retrieval
        
        # Then do graph-based retrieval using entity types discovered from database
        logger.info(f"Using discovered entity types for graph retrieval: {self.entity_types[:3]}...")
        logger.info(f"Performing graph retrieval for query: {query}")
        
        try:
            graph_results = self.graph_engine.retrieve(
                query=query,
                max_nodes=self.graph_max_nodes,
                max_depth=self.graph_max_depth,
                entity_types=self.entity_types
            )
            
            # Convert nodes to documents
            graph_docs = [
                Document(
                    page_content=self._extract_content(node),
                    metadata={
                        "id": node.get('id', ''),
                        "label": node.get('label', ''),
                        "title": node.get('title', ''),
                        "source": "graph"
                    }
                )
                for node in graph_results.get('nodes', [])
            ]
            
            # Format the relationships
            self.graph_relationships = [
                f"{rel['source_node'].get('title', 'Unknown')} --[{rel['type']}]--> {rel['target_node'].get('title', 'Unknown')}"
                for rel in graph_results.get('relationships', [])
            ]
            
            logger.info(f"Found {len(graph_docs)} entities and {len(self.graph_relationships)} relationships via graph search")
            
        except Exception as e:
            logger.warning(f"Error in graph retrieval: {e}")
            # Continue execution
            
        # Combine and deduplicate results
        combined_docs = self._combine_and_deduplicate(vector_docs, graph_docs)
        
        return combined_docs 