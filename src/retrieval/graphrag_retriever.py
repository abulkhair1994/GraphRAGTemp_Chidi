"""
GraphRAG Retriever for combining vector and graph-based retrieval.

This module provides a hybrid retriever that combines both vector similarity search
and graph-based retrieval for enhanced context retrieval in RAG applications.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase

from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain_neo4j import Neo4jVector

from src.retrieval.vector_retrieval import VectorRetrievalEngine
from src.retrieval.graph_retrieval import GraphRetrievalEngine, GraphResult
from src.retrieval.document_processor import DocumentProcessor
from src.retrieval.metadata_retrieval import MetadataRetriever

logger = logging.getLogger(__name__)

@dataclass
class GraphRAGResult:
    """Combined result from GraphRAG retrieval."""
    vector_documents: List[Document]
    graph_result: GraphResult
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    vector_query: Optional[str] = None
    hybrid_scores: Optional[Dict[str, List[float]]] = None
    search_strategy: str = "standard"
    
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
        vector_node_label: str = "Problem",  # Default but will query all educational types
        vector_search_type: str = "hybrid",
        vector_index_name: str = "educational_problem_embeddings",  # Default to one of the educational indexes
        graph_max_hops: int = 2,
        graph_max_nodes: int = 10,
        text_search_threshold: float = 0.7,
        vector_top_k: int = 3,
        text_top_k: int = 3,
        # Hybrid search parameters
        enable_hybrid_search: bool = True,
        hybrid_alpha: float = 0.5,
        fulltext_index_name: str = "fulltext_index"
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
            enable_hybrid_search: Whether to enable hybrid search
            hybrid_alpha: Weight for hybrid search
            fulltext_index_name: Name of the fulltext index
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
        
        # Hybrid search parameters
        self.enable_hybrid_search = enable_hybrid_search
        self.hybrid_alpha = hybrid_alpha
        self.fulltext_index_name = fulltext_index_name
        
        # Initialize engines
        self.vector_engine = VectorRetrievalEngine(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
            embedding_model=embedding_model,
            index_name="educational_problem_embeddings",  # Use the new comprehensive indexes
            text_node_property=vector_text_node_property,
            embedding_node_property=vector_embedding_node_property,
            embedding_dimension=vector_embedding_dimension,
            distance_metric=vector_distance_metric,
            node_label=vector_node_label,
            search_type=vector_search_type
        )
        
        # Initialize Neo4jVector for true hybrid search
        if enable_hybrid_search and embedding_model:
            try:
                self.neo4j_vector = Neo4jVector(
                    embedding=embedding_model,
                    url=neo4j_url,
                    username=neo4j_username,
                    password=neo4j_password,
                    database=neo4j_database,
                    index_name="educational_problem_embeddings",  # Use the new comprehensive indexes
                    node_label=vector_node_label,
                    text_node_property=vector_text_node_property,
                    embedding_node_property=vector_embedding_node_property,
                    search_type="hybrid"  # Enable true hybrid search
                )
                logger.info("Neo4jVector hybrid search initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4jVector hybrid search: {e}")
                self.neo4j_vector = None
        else:
            self.neo4j_vector = None
        
        self.graph_engine = GraphRetrievalEngine(
            neo4j_url=self.neo4j_url,
            neo4j_username=self.neo4j_username,
            neo4j_password=self.neo4j_password,
            neo4j_database=self.neo4j_database
        )
        
        # Initialize document processor for entity extraction
        self.document_processor = DocumentProcessor(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
        )
        
        # Initialize metadata retriever
        self.metadata_retriever = MetadataRetriever(
            neo4j_url=neo4j_url,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
        )
        
        # Initialize Neo4j driver for direct database access
        self.driver = GraphDatabase.driver(
            self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # Get database schema information
        self.db_schema = self.graph_engine.get_db_schema()
        
        # Update the node label for vector search if not provided
        if not self.vector_node_label:
            self._discover_document_node_label()
    
    def setup_hybrid_search_indexes(self):
        """Setup fulltext index for hybrid search."""
        if not self.enable_hybrid_search:
            return
            
        try:
            driver = GraphDatabase.driver(
                self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password)
            )
            
            with driver.session(database=self.neo4j_database) as session:
                # Create fulltext index for hybrid search
                fulltext_query = f"""
                CREATE FULLTEXT INDEX {self.fulltext_index_name} IF NOT EXISTS
                FOR (n:{self.vector_node_label})
                ON EACH [n.{self.vector_text_node_property}]
                """
                
                session.run(fulltext_query)
                logger.info(f"Fulltext index {self.fulltext_index_name} created for hybrid search")
            
            driver.close()
            
        except Exception as e:
            logger.error(f"Error setting up fulltext index: {e}")
    
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
    
    def retrieve_by_learning_objective(
        self,
        query: str,
        learning_objective: Optional[str] = None,
        content_types: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        vector_k: Optional[int] = None
    ) -> GraphRAGResult:
        """Retrieve content filtered by learning objective and educational metadata.
        
        This method allows targeted retrieval of educational content based on specific
        learning objectives and content types, supporting precise educational needs.
        
        Args:
            query: Query text
            learning_objective: Specific learning objective to filter by
            content_types: Types of content to include (e.g., "Example", "Practice", "Explanation")
            difficulty: Difficulty level to filter by (e.g., "Basic", "Intermediate", "Advanced")
            vector_k: Number of vector results to return (overrides default)
            
        Returns:
            GraphRAGResult containing vector and graph results filtered by learning objective
        """
        vector_docs = []
        vector_avg_score = 0.0
        graph_result = GraphResult(entities=[], relationships=[], paths=[])
        graph_score = 0.0
        
        # Build a custom Cypher query for educational content retrieval
        try:
            logger.info(f"Performing learning objective-filtered vector search for query: {query}")
            
            # Connect to Neo4j
            driver = GraphDatabase.driver(
                self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password)
            )
            
            # Build the node filter clause
            node_filters = []
            node_labels = []
            
            # Include common educational content nodes
            common_node_types = ["Content", "Chapter", "Section", "Example", "Exercise", "Problem", "Solution"]
            
            # If specific content types were provided, filter to those
            if content_types:
                node_labels.extend(content_types)
            else:
                node_labels.extend(common_node_types)
            
            # Add the learning objective filter if provided
            if learning_objective:
                node_filters.append(f"n.learning_objective = '{learning_objective}'")
            
            # Add difficulty filter if provided
            if difficulty:
                node_filters.append(f"n.difficulty = '{difficulty}'")
            
            # Combine node labels and filters
            node_labels_str = " OR ".join([f"n:{label}" for label in node_labels])
            node_filters_str = " AND ".join(node_filters) if node_filters else "true"
            
            with driver.session(database=self.neo4j_database) as session:
                # First try to use vector search combined with learning objective filtering
                try:
                    # Use the comprehensive educational indexes instead of single content_embeddings
                    educational_indexes = [
                        "educational_problem_embeddings",
                        "educational_exercise_embeddings", 
                        "educational_solution_embeddings",
                        "educational_example_embeddings",
                        "educational_para_embeddings"
                    ]
                    
                    # Get the embedding for the query
                    embedding = self.embedding_model.embed_query(query)
                    
                    vector_results = []
                    
                    # Query each educational index
                    for index_name in educational_indexes:
                        try:
                            vector_query = f"""
                            CALL db.index.vector.queryNodes(
                                '{index_name}',
                                $top_k_per_index,
                                $embedding
                            ) YIELD node, score
                            WHERE ({node_labels_str}) AND ({node_filters_str})
                            RETURN node, score
                            """
                            
                            # Execute the query for this index
                            result = session.run(
                                vector_query, 
                                {
                                    "embedding": embedding, 
                                    "top_k_per_index": max(1, (vector_k or self.vector_top_k) // len(educational_indexes))
                                }
                            )
                            
                            # Process the results from this index
                            for record in result:
                                node = record["node"]
                                score = record["score"]
                                
                                # Convert Neo4j node to document
                                metadata = dict(node)
                                content = metadata.get(self.vector_text_node_property, "")
                                
                                # If text_content is not available, try other content properties
                                if not content:
                                    for prop in ["content", "text", "problem_text", "solution_text"]:
                                        if prop in metadata:
                                            content = metadata.get(prop)
                                            break
                                
                                if not content and "title" in metadata:
                                    content = metadata.get("title", "")
                                
                                doc = Document(page_content=content, metadata=metadata)
                                vector_results.append((doc, score))
                                
                        except Exception as e:
                            logger.warning(f"Error querying index {index_name}: {e}")
                            continue
                    
                    # If we have results, extract documents and calculate average score
                    if vector_results:
                        vector_docs = [doc for doc, _ in vector_results]
                        vector_avg_score = sum(score for _, score in vector_results) / len(vector_results)
                        logger.info(f"Found {len(vector_docs)} documents via comprehensive vector search with avg score: {vector_avg_score}")
                    else:
                        logger.warning(f"No vector results found for query with the specified filters")
                        
                        # Fallback to text search with the same filters
                        text_query = f"""
                        MATCH (n)
                        WHERE ({node_labels_str}) AND ({node_filters_str})
                        AND (
                            n.{self.vector_text_node_property} CONTAINS $query 
                            OR n.title CONTAINS $query
                            OR n.content CONTAINS $query
                        )
                        RETURN n
                        LIMIT $limit
                        """
                        
                        result = session.run(
                            text_query, 
                            {
                                "query": query, 
                                "limit": vector_k or self.vector_top_k
                            }
                        )
                        
                        # Process text search results
                        for record in result:
                            node = record["n"]
                            
                            # Convert Neo4j node to document
                            metadata = dict(node)
                            content = metadata.get(self.vector_text_node_property, "")
                            
                            # If text_content is not available, try other content properties
                            if not content:
                                for prop in ["content", "text", "problem_text", "solution_text"]:
                                    if prop in metadata:
                                        content = metadata.get(prop)
                                        break
                            
                            if not content and "title" in metadata:
                                content = metadata.get("title", "")
                            
                            doc = Document(page_content=content, metadata=metadata)
                            vector_docs.append(doc)
                        
                        if vector_docs:
                            logger.info(f"Found {len(vector_docs)} documents via filtered text search")
                        else:
                            logger.warning("No text search results found for query with the specified filters")
                
                except Exception as e:
                    logger.error(f"Error in comprehensive vector search: {e}")
                
                # Now retrieve the connected educational structure
                try:
                    logger.info(f"Retrieving related educational content structure")
                    
                    # Get connected educational structure using the vector result IDs
                    if vector_docs:
                        node_ids = []
                        for doc in vector_docs:
                            if "id" in doc.metadata:
                                node_ids.append(doc.metadata["id"])
                        
                        if node_ids:
                            node_ids_str = ", ".join([str(nid) for nid in node_ids])
                            
                            graph_query = f"""
                            MATCH (n)
                            WHERE id(n) IN [{node_ids_str}]
                            
                            // Find related nodes within 2 hops
                            OPTIONAL MATCH path = (n)-[r:CONTAINS|RELATED_TO|PREREQUISITE_FOR|REFERENCES*1..2]-(related)
                            
                            // Collect all unique nodes and relationships
                            WITH collect(DISTINCT n) + collect(DISTINCT related) AS all_nodes,
                                 collect(DISTINCT r) AS all_rels
                            
                            // Format nodes as entities
                            WITH 
                            [node IN all_nodes WHERE node IS NOT NULL |
                                {{
                                    id: id(node),
                                    name: COALESCE(node.name, node.title, ''),
                                    type: labels(node)[0],
                                    properties: properties(node)
                                }}
                            ] AS entities,
                            
                            // Format relationships
                            [rel IN all_rels WHERE rel IS NOT NULL |
                                {{
                                    source: id(startNode(rel)),
                                    target: id(endNode(rel)),
                                    type: type(rel),
                                    properties: properties(rel)
                                }}
                            ] AS relationships
                            
                            // Return everything
                            RETURN entities, relationships
                            """
                            
                            result = session.run(graph_query)
                            
                            if result.peek():
                                record = result.single()
                                graph_result = GraphResult(
                                    entities=record["entities"],
                                    relationships=record["relationships"],
                                    paths=[]
                                )
                                
                                # Set graph score based on number of entities and relationships
                                entity_count = len(graph_result.entities)
                                rel_count = len(graph_result.relationships)
                                
                                if entity_count > 0 or rel_count > 0:
                                    graph_score = (entity_count + rel_count) / 10  # Normalize to approximate 0-1 range
                                    logger.info(f"Found {entity_count} entities and {rel_count} relationships in educational structure")
                            else:
                                logger.warning(f"No related structure found for the retrieved content")
                    else:
                        logger.warning("No vector documents to build educational structure from")
                
                except Exception as e:
                    logger.error(f"Error retrieving educational structure: {e}")
                
                driver.close()
        
        except Exception as e:
            logger.error(f"Learning objective-based retrieval failed: {e}")
        
        # Return combined results
        return GraphRAGResult(
            vector_documents=vector_docs,
            graph_result=graph_result,
            vector_score=vector_avg_score,
            graph_score=graph_score
        )
    
    def get_available_content_types(self) -> List[Dict[str, Any]]:
        """Get all available content types and their counts.
        
        This method queries the knowledge graph to find all content type nodes
        and returns their metadata, helping applications discover available content.
        
        Returns:
            List of dictionaries containing content type metadata
        """
        content_types = []
        
        try:
            logger.info("Retrieving available content types")
            
            # Connect to Neo4j
            driver = GraphDatabase.driver(
                self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password)
            )
            
            with driver.session(database=self.neo4j_database) as session:
                # Count nodes by label to find content types
                query = """
                MATCH (n)
                WITH labels(n)[0] as node_type, count(*) as count
                WHERE count > 0
                RETURN {
                    type: node_type,
                    count: count
                } as content_type
                ORDER BY count DESC
                """
                
                result = session.run(query)
                
                for record in result:
                    content_type = record["content_type"]
                    content_types.append(content_type)
                    
            driver.close()
            logger.info(f"Found {len(content_types)} content types")
            
        except Exception as e:
            logger.error(f"Error retrieving content types: {e}")
            
        return content_types

    def close(self):
        """Close all connections."""
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.close()
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

    def retrieve_with_metadata_filter(
        self,
        query: str,
        filters: Dict[str, Any],
        node_labels: List[str] = None,
        limit: int = 5
    ) -> GraphRAGResult:
        """
        Retrieve documents using metadata filtering before vector search.
        
        Args:
            query: Search query
            filters: Dictionary of property filters {property: value}
            node_labels: Node labels to include in search
            limit: Maximum number of results
            
        Returns:
            GraphRAGResult containing filtered documents
        """
        logger.info(f"Retrieving with metadata filter. Query: {query}, Filters: {filters}")
        
        # Get embedding for the query
        query_embedding = self._get_query_embedding(query)
        
        # Use metadata retriever to get filtered results
        results = self.metadata_retriever.retrieve_with_metadata_filter(
            query=query_embedding,
            filters=filters,
            node_labels=node_labels,
            limit=limit
        )
        
        # Convert to LangChain documents
        documents = []
        for result in results:
            doc = Document(
                page_content=result.get("text_content", ""),
                metadata=result
            )
            documents.append(doc)
            
        return GraphRAGResult(
            vector_documents=documents,
            graph_result=None,
            vector_query=query
        )
    
    def retrieve_with_relationships(
        self,
        query: str,
        entity_type: str,
        relationship_types: List[str] = None,
        max_depth: int = 1,
        limit: int = 5
    ) -> GraphRAGResult:
        """
        Retrieve documents based on both semantic similarity and relationship traversal.
        
        Args:
            query: Search query
            entity_type: Type of entity to start with
            relationship_types: Types of relationships to traverse
            max_depth: Maximum relationship traversal depth
            limit: Maximum number of results
            
        Returns:
            GraphRAGResult containing documents with relationships
        """
        logger.info(f"Retrieving with relationships. Query: {query}, Entity type: {entity_type}")
        
        # Don't convert to embeddings for now since we're using text search as a fallback
        # query_embedding = self._get_query_embedding(query)
        
        # Use metadata retriever to get results with relationships
        results = self.metadata_retriever.retrieve_with_relationships(
            query=query,  # Pass the text query directly
            entity_type=entity_type,
            relationship_types=relationship_types,
            max_depth=max_depth,
            limit=limit
        )
        
        # Convert to LangChain documents
        documents = []
        for result in results:
            doc = Document(
                page_content=result.get("text_content", ""),
                metadata=result
            )
            documents.append(doc)
            
        return GraphRAGResult(
            vector_documents=documents,
            graph_result=None,
            vector_query=query
        )
    
    def retrieve_learning_path(
        self,
        concept: str,
        difficulty_level: Optional[str] = None,
        max_items: int = 10
    ) -> GraphRAGResult:
        """
        Retrieve a learning path for a concept based on prerequisites.
        
        Args:
            concept: The concept to learn
            difficulty_level: Optional difficulty level filter
            max_items: Maximum number of items in the path
            
        Returns:
            GraphRAGResult containing documents in learning path sequence
        """
        logger.info(f"Retrieving learning path for concept: {concept}")
        
        # Use metadata retriever to get learning path
        results = self.metadata_retriever.retrieve_learning_path(
            concept=concept,
            difficulty_level=difficulty_level,
            max_items=max_items
        )
        
        # Convert to LangChain documents
        documents = []
        for result in results:
            doc = Document(
                page_content=result.get("text_content", ""),
                metadata=result
            )
            documents.append(doc)
            
        return GraphRAGResult(
            vector_documents=documents,
            graph_result=None,
            vector_query=f"Learning path for {concept}"
        )
    
    def retrieve_hierarchical_context(
        self,
        entity_id: str,
        hierarchy_relationships: List[str] = None,
        include_siblings: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve the hierarchical context of an entity (e.g., example → section → chapter).
        
        This method provides a direct way to explore the hierarchical relationships
        of educational content, showing the structure and relationships between
        content at different levels.
        
        Args:
            entity_id: The ID of the entity to get context for
            hierarchy_relationships: List of relationships defining hierarchy
            include_siblings: Whether to include sibling nodes
            
        Returns:
            Dictionary with hierarchical context information
        """
        try:
            logger.info(f"Retrieving hierarchical context for entity: {entity_id}")
            
            # Use the graph engine to retrieve hierarchical context
            return self.graph_engine.retrieve_hierarchical_context(
                entity_id=entity_id,
                hierarchy_relationships=hierarchy_relationships,
                include_siblings=include_siblings
            )
        except Exception as e:
            logger.error(f"Error retrieving hierarchical context: {e}")
            return {}
    
    def _get_query_embedding(self, query: str) -> Union[List[float], str]:
        """Get embedding for query text or return query if no embedding model."""
        if hasattr(self.vector_engine, "embedding_model") and self.vector_engine.embedding_model:
            return self.vector_engine.embedding_model.embed_query(query)
        return query

    def _combine_and_deduplicate(self, vector_docs: List[Document], graph_docs: List[Document]) -> List[Document]:
        """Combine and deduplicate documents from vector and graph retrieval."""
        # Combine documents
        combined_docs = vector_docs + graph_docs
        
        # Deduplicate documents
        seen = set()
        deduplicated_docs = []
        for doc in combined_docs:
            if doc.id not in seen:
                seen.add(doc.id)
                deduplicated_docs.append(doc)
        
        return deduplicated_docs

    def retrieve_educational_content(
        self,
        query: str,
        auto_parse: bool = True,
        limit: int = 10
    ) -> GraphRAGResult:
        """Enhanced educational content retrieval with chapter/section filtering.
        
        Now uses the new chapter_number and semantic_keywords metadata for
        precise content filtering and improved relevance.
        
        Args:
            query: Educational query
            auto_parse: Whether to automatically parse educational parameters
            limit: Maximum number of results
            
        Returns:
            GraphRAGResult with properly filtered educational content
        """
        try:
            logger.info(f"🎓 Enhanced educational retrieval for: {query}")
            # Parse the query for educational context
            parsed_query = self.parse_educational_query(query) if auto_parse else {
                'original_query': query,
                'chapter': None,
                'section': None,
                'content_types': ['Problem', 'Exercise', 'Solution', 'Example'],
                'learning_intent': 'understand',
                'keywords': query.lower().split()
            }
            
            # Strategy 1: Chapter-filtered search (if chapter detected)
            chapter_results = []
            if parsed_query.get('chapter'):
                logger.info(f"📚 Detected chapter {parsed_query['chapter']}, searching...")
                chapter_results = self._search_by_chapter(
                    query, parsed_query['chapter'], limit//2
                )
                logger.info(f"📚 Chapter search returned {len(chapter_results)} results")
            
            # Strategy 2: Semantic keyword search
            logger.info(f"🏷️  Searching with keywords: {parsed_query['keywords']}")
            keyword_results = self._search_by_semantic_keywords(
                query, parsed_query['keywords'], limit//2
            )
            logger.info(f"🏷️  Keyword search returned {len(keyword_results)} results")
            
            # Strategy 3: Direct content search (fallback)
            logger.info(f"🔍 Direct search for content types: {parsed_query['content_types']}")
            direct_results = self._search_educational_content_direct(
                query, parsed_query['content_types'], limit//2
            )
            logger.info(f"🔍 Direct search returned {len(direct_results)} results")
            
            # Combine and deduplicate results
            all_results = []
            seen_ids = set()
            
            # Prioritize chapter-specific results
            for result in chapter_results:
                if result.metadata.get('id') not in seen_ids:
                    result.metadata['search_strategy'] = 'chapter_filtered'
                    all_results.append(result)
                    seen_ids.add(result.metadata.get('id'))
            
            # Add keyword results
            for result in keyword_results:
                if result.metadata.get('id') not in seen_ids and len(all_results) < limit:
                    result.metadata['search_strategy'] = 'semantic_keywords'
                    all_results.append(result)
                    seen_ids.add(result.metadata.get('id'))
            
            # Add direct results as fallback
            for result in direct_results:
                if result.metadata.get('id') not in seen_ids and len(all_results) < limit:
                    result.metadata['search_strategy'] = 'direct_search'
                    all_results.append(result)
                    seen_ids.add(result.metadata.get('id'))
            
            # Create graph result (simplified for now)
            graph_result = GraphResult(
                entities=[],
                relationships=[],
                paths=[]
            )
            
            return GraphRAGResult(
                vector_documents=all_results[:limit],
                graph_result=graph_result,
                vector_score=0.9,  # High confidence with new metadata
                graph_score=0.8
            )
            
        except Exception as e:
            logger.error(f"Error in educational content retrieval: {e}")
            # Fallback to basic search
            return self._fallback_search(query, limit)
    
    def _search_by_chapter(self, query: str, chapter_num: int, limit: int = 5) -> List[Document]:
        """Search for content within a specific chapter using vector similarity.
        
        Args:
            query: Search query
            chapter_num: Chapter number to search in
            limit: Maximum number of results to return
            
        Returns:
            List of documents from the specified chapter
        """
        try:
            # Check if embedding model is available
            if not self.embedding_model:
                logger.warning("No embedding model available for chapter search, falling back to text search")
                return self.find_content_by_chapter_section(
                    chapter_num=chapter_num,
                    content_types=['Problem', 'Exercise', 'Solution', 'Example', 'Para'],
                    limit=limit
                )
            
            # Get query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            with self.driver.session(database=self.neo4j_database) as session:
                # Enhanced search for content in the specified chapter with proper metadata
                cypher_query = """
                MATCH (n)
                WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example OR n:Para OR n:Section)
                AND n.chapter_number = $chapter_num
                AND n.text_content IS NOT NULL
                AND n.text_content <> ""
                AND size(n.text_content) > 20
                AND n.embedding IS NOT NULL
                WITH n, 
                     labels(n)[0] as primary_label,
                     CASE 
                        WHEN n.title IS NOT NULL AND n.title <> '' THEN n.title
                        WHEN n.id IS NOT NULL THEN 'Chapter ' + toString($chapter_num) + ' - ' + n.id
                        ELSE 'Chapter ' + toString($chapter_num) + ' Content'
                     END as content_title,
                     CASE labels(n)[0]
                        WHEN 'Para' THEN 1
                        WHEN 'Section' THEN 1
                        ELSE 2
                     END as content_priority,
                     vector.similarity.cosine(n.embedding, $query_embedding) as similarity
                WHERE similarity > $similarity_threshold
                RETURN n.text_content as content,
                       n.id as node_id,
                       primary_label as content_type,
                       content_title as title,
                       n.module_id as module_id,
                       n.chapter_number as chapter_number,
                       null as section_number,  // Most nodes don't have section_number
                       similarity,
                       content_priority,
                       n.semantic_keywords as keywords,
                       properties(n) as all_properties
                ORDER BY content_priority DESC, similarity DESC
                LIMIT $limit
                """
                
                result = session.run(
                    cypher_query,
                    {
                        "query_embedding": query_embedding,
                        "chapter_num": chapter_num,
                        "similarity_threshold": 0.1,  # Lower threshold to ensure we get results
                        "limit": limit
                    }
                )
                
                documents = []
                for record in result:
                    # Get all properties and filter out text_content
                    all_props = record["all_properties"] or {}
                    filtered_props = {k: v for k, v in all_props.items() 
                                    if k != "text_content" and v is not None}
                    
                    # Create comprehensive metadata
                    metadata = {
                        'id': record["node_id"],
                        'type': record["content_type"],
                        'title': record["title"],
                        'module_id': record["module_id"],
                        'chapter_number': record["chapter_number"],
                        'section_number': record["section_number"],
                        'similarity': float(record["similarity"]),
                        'content_priority': record["content_priority"],
                        'semantic_keywords': record.get("keywords"),
                        'search_strategy': 'chapter_semantic',
                        **filtered_props  # Include all other node properties
                    }
                    
                    doc = Document(
                        page_content=record["content"],
                        metadata=metadata
                    )
                    documents.append(doc)
                
                return documents
                
        except Exception as e:
            logger.error(f"Error in _search_by_chapter: {e}")
            return []
    
    def _search_by_semantic_keywords(self, query: str, keywords: List[str], limit: int) -> List[Document]:
        """Search using semantic keywords with vector similarity across all educational content types."""
        try:
            # If we have an embedding model, use vector search with keyword filtering
            if self.embedding_model:
                return self._vector_search_with_keywords(query, keywords, limit)
            else:
                # Fallback to text-based keyword search
                return self._text_search_with_keywords(query, keywords, limit)
                
        except Exception as e:
            logger.error(f"Error in semantic keyword search: {e}")
            return []
    
    def _vector_search_with_keywords(self, query: str, keywords: List[str], limit: int) -> List[Document]:
        """Use comprehensive vector indexes with keyword filtering."""
        try:
            # Get query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Define all educational indexes
            educational_indexes = [
                ("educational_problem_embeddings", "Problem"),
                ("educational_exercise_embeddings", "Exercise"),
                ("educational_solution_embeddings", "Solution"),
                ("educational_example_embeddings", "Example"),
                ("educational_para_embeddings", "Para")
            ]
            
            all_results = []
            
            with self.driver.session(database=self.neo4j_database) as session:
                for index_name, content_type in educational_indexes:
                    try:
                        # Create keyword matching conditions for filtering
                        keyword_conditions = []
                        for keyword in keywords:
                            escaped_keyword = keyword.replace("'", "\\'")
                            keyword_conditions.append(f"ANY(k IN node.semantic_keywords WHERE toLower(k) CONTAINS toLower('{escaped_keyword}'))")
                        
                        # If no keyword conditions, skip keyword filtering
                        keyword_filter = ""
                        if keyword_conditions:
                            keyword_filter = f"AND ({' OR '.join(keyword_conditions)})"
                        
                        # Query this specific index with vector similarity and keyword filtering
                        search_query = f"""
                        CALL db.index.vector.queryNodes(
                            '{index_name}',
                            $top_k_per_type,
                            $embedding
                        ) YIELD node, score
                        WHERE node.text_content IS NOT NULL
                        AND node.text_content <> ''
                        AND size(node.text_content) > 20
                        {keyword_filter}
                        WITH node, score,
                             labels(node)[0] as primary_label,
                             CASE 
                                WHEN node.title IS NOT NULL AND node.title <> '' THEN node.title
                                WHEN node.id IS NOT NULL THEN 'Chapter ' + toString(node.chapter_number) + ' - ' + node.id
                                ELSE 'Chapter ' + toString(node.chapter_number) + ' Content'
                             END as content_title,
                             CASE labels(node)[0]
                                WHEN 'Para' THEN 1
                                WHEN 'Section' THEN 1
                                ELSE 2
                             END as content_priority,
                             size([k IN node.semantic_keywords WHERE ANY(q IN $query_keywords WHERE toLower(k) CONTAINS toLower(q))]) as keyword_matches
                        RETURN node.text_content as content,
                               node.id as node_id,
                               primary_label as content_type,
                               content_title as title,
                               node.module_id as module_id,
                               node.chapter_number as chapter_number,
                               null as section_number,
                               node.semantic_keywords as keywords,
                               score,
                               keyword_matches,
                               content_priority,
                               'vector_semantic_search' as search_strategy,
                               properties(node) as all_properties
                        ORDER BY score DESC, keyword_matches DESC
                        """
                        
                        # Get results for this content type (limit per type to ensure diversity)
                        results_per_type = max(1, limit // len(educational_indexes))
                        safe_keywords = [kw.replace("'", "") for kw in keywords]
                        
                        result = session.run(
                            search_query,
                            embedding=query_embedding,
                            top_k_per_type=results_per_type,
                            query_keywords=safe_keywords
                        )
                        
                        # Process results for this content type
                        for record in result:
                            try:
                                # Get all properties and filter out text_content
                                all_props = record["all_properties"] or {}
                                filtered_props = {k: v for k, v in all_props.items() 
                                                if k != "text_content" and v is not None}
                                
                                # Create comprehensive metadata
                                metadata = {
                                    'id': record["node_id"],
                                    'type': record["content_type"],
                                    'title': record["title"],
                                    'module_id': record["module_id"],
                                    'chapter_number': record["chapter_number"],
                                    'section_number': record["section_number"],
                                    'content_length': len(record["content"]) if record["content"] else 0,
                                    'keyword_matches': record.get("keyword_matches"),
                                    'semantic_keywords': record.get("keywords"),
                                    'content_priority': record.get("content_priority", 2),
                                    'score': float(record.get("score", 0.0)),
                                    'search_strategy': record.get("search_strategy", 'vector_semantic_search'),
                                    'source_index': index_name,
                                    **filtered_props  # Include all other node properties
                                }
                                
                                doc = Document(
                                    page_content=record["content"],
                                    metadata=metadata
                                )
                                all_results.append(doc)
                                
                            except Exception as e:
                                logger.warning(f"Error processing result from {index_name}: {e}")
                                continue
                                
                    except Exception as e:
                        logger.warning(f"Error querying index {index_name}: {e}")
                        continue
            
            # Sort all results by score and return top results
            all_results.sort(key=lambda x: x.metadata.get('score', 0.0), reverse=True)
            final_results = all_results[:limit]
            
            logger.info(f"Vector semantic search returned {len(final_results)} results from {len(set(r.metadata.get('type') for r in final_results))} content types")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in vector search with keywords: {e}")
            return []
    
    def _text_search_with_keywords(self, query: str, keywords: List[str], limit: int) -> List[Document]:
        """Fallback text-based keyword search when no embedding model is available."""
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                # Create keyword matching conditions with proper escaping
                keyword_conditions = []
                for keyword in keywords:
                    escaped_keyword = keyword.replace("'", "\\'")
                    keyword_conditions.append(f"ANY(k IN n.semantic_keywords WHERE toLower(k) CONTAINS toLower('{escaped_keyword}'))")
                
                if not keyword_conditions:
                    return []
                
                cypher_query = f"""
                MATCH (n)
                WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example OR n:Para OR n:Section)
                AND n.semantic_keywords IS NOT NULL
                AND n.text_content IS NOT NULL
                AND n.text_content <> ""
                AND size(n.text_content) > 20
                AND ({' OR '.join(keyword_conditions)})
                WITH n,
                     labels(n)[0] as primary_label,
                     CASE 
                        WHEN n.title IS NOT NULL AND n.title <> '' THEN n.title
                        WHEN n.id IS NOT NULL THEN 'Content ID: ' + n.id
                        ELSE 'Educational Content'
                     END as content_title,
                     CASE labels(n)[0]
                        WHEN 'Para' THEN 1
                        WHEN 'Section' THEN 1
                        ELSE 2
                     END as content_priority,
                     size([k IN n.semantic_keywords WHERE ANY(q IN $query_keywords WHERE toLower(k) CONTAINS toLower(q))]) as keyword_matches
                RETURN n.text_content as content,
                       n.id as node_id,
                       primary_label as content_type,
                       content_title as title,
                       n.module_id as module_id,
                       n.chapter_number as chapter_number,
                       null as section_number,
                       n.semantic_keywords as keywords,
                       keyword_matches,
                       content_priority,
                       properties(n) as all_properties
                ORDER BY keyword_matches DESC, content_priority DESC, size(n.text_content) DESC
                LIMIT $limit
                """
                
                # Use parameterized query for keywords to avoid injection
                safe_keywords = [kw.replace("'", "") for kw in keywords]
                result = session.run(cypher_query, query_keywords=safe_keywords, limit=limit)
                
                documents = []
                record_count = 0
                for record in result:
                    record_count += 1
                    
                    # Get all properties and filter out text_content
                    all_props = record["all_properties"] or {}
                    filtered_props = {k: v for k, v in all_props.items() 
                                    if k != "text_content" and v is not None}
                    
                    # Create comprehensive metadata
                    metadata = {
                        'id': record["node_id"],
                        'type': record["content_type"],
                        'title': record["title"],
                        'module_id': record["module_id"],
                        'chapter_number': record["chapter_number"],
                        'section_number': record["section_number"],
                        'content_length': len(record["content"]) if record["content"] else 0,
                        'keyword_matches': record.get("keyword_matches"),
                        'semantic_keywords': record.get("keywords"),
                        'content_priority': record.get("content_priority", 2),
                        'search_strategy': 'text_semantic_keywords',
                        **filtered_props  # Include all other node properties
                    }
                    
                    doc = Document(
                        page_content=record["content"],
                        metadata=metadata
                    )
                    documents.append(doc)
                
                logger.info(f"Text semantic search found {record_count} documents with keyword matches")
                return documents
                
        except Exception as e:
            logger.error(f"Error in text-based semantic keyword search: {e}")
            return []
    
    def _search_educational_content_direct(self, query: str, content_types: List[str], limit: int) -> List[Document]:
        """Direct search for educational content."""
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                # Escape single quotes in the query for safe Cypher execution
                escaped_query = query.replace("'", "\\'")
                text_filter = f"toLower(n.text_content) CONTAINS toLower('{escaped_query}')"
                
                cypher_query = f"""
                MATCH (n)
                WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example OR n:Para OR n:Section)
                AND n.text_content IS NOT NULL 
                AND n.text_content <> ""
                AND size(n.text_content) > 20
                AND ({text_filter})
                WITH n,
                     labels(n)[0] as primary_label,
                     CASE 
                        WHEN n.title IS NOT NULL AND n.title <> '' THEN n.title
                        WHEN n.id IS NOT NULL THEN 'Content ID: ' + n.id
                        ELSE 'Educational Content'
                     END as content_title,
                     CASE labels(n)[0]
                        WHEN 'Para' THEN 1
                        WHEN 'Section' THEN 1
                        ELSE 2
                     END as content_priority,
                     size(n.text_content) as content_length
                RETURN n.text_content as content,
                       n.id as node_id,
                       primary_label as content_type,
                       content_title as title,
                       n.module_id as module_id,
                       n.chapter_number as chapter_number,
                       null as section_number,  // Most nodes don't have section_number
                       n.semantic_keywords as keywords,
                       content_length,
                       content_priority,
                       properties(n) as all_properties
                ORDER BY content_priority DESC, content_length DESC
                LIMIT $limit
                """
                
                result = session.run(cypher_query, limit=limit)
                
                documents = []
                record_count = 0
                for record in result:
                    record_count += 1
                    
                    # Get all properties and filter out text_content
                    all_props = record["all_properties"] or {}
                    filtered_props = {k: v for k, v in all_props.items() 
                                    if k != "text_content" and v is not None}
                    
                    # Create comprehensive metadata
                    metadata = {
                        'id': record["node_id"],
                        'type': record["content_type"],
                        'title': record["title"],
                        'module_id': record["module_id"],
                        'chapter_number': record["chapter_number"],
                        'section_number': record["section_number"],
                        'content_length': record.get("content_length", 0),
                        'semantic_keywords': record.get("keywords"),
                        'content_priority': record.get("content_priority", 2),
                        'search_strategy': 'direct_search',
                        **filtered_props  # Include all other node properties
                    }
                    
                    doc = Document(
                        page_content=record["content"],
                        metadata=metadata
                    )
                    documents.append(doc)
                
                logger.info(f"Found {record_count} documents through direct search")
                return documents
                
        except Exception as e:
            logger.error(f"Error in direct educational search: {e}")
            return []
    
    def _fallback_search(self, query: str, limit: int) -> GraphRAGResult:
        """Fallback search when enhanced methods fail."""
        try:
            # First try to extract chapter/section information
            chapter_match = re.search(r'chapter\s+(\d+)', query.lower())
            section_match = re.search(r'section\s+(\d+)', query.lower())
            
            if chapter_match:
                chapter_num = int(chapter_match.group(1))
                section_num = int(section_match.group(1)) if section_match else None
                
                # Try to find content by chapter/section
                docs = self.find_content_by_chapter_section(
                    chapter_num=chapter_num,
                    section_num=section_num,
                    limit=limit
                )
                if docs:
                    return GraphRAGResult(
                        vector_documents=docs,
                        graph_result=GraphResult(entities=[], relationships=[], paths=[]),
                        vector_score=0.7,
                        graph_score=0.5,
                        search_strategy='chapter_section'
                    )
            
            # If no chapter/section results or no chapter specified, fall back to vector search
            if hasattr(self, 'vector_engine') and self.vector_engine:
                vector_docs = self.vector_engine.similarity_search(query, top_k=limit)
                
                return GraphRAGResult(
                    vector_documents=vector_docs,
                    graph_result=GraphResult(entities=[], relationships=[], paths=[]),
                    vector_score=0.5,
                    graph_score=0.3,
                    search_strategy='vector_fallback'
                )
            
            # Last resort: empty result
            return GraphRAGResult(
                vector_documents=[],
                graph_result=GraphResult(entities=[], relationships=[], paths=[]),
                vector_score=0.0,
                graph_score=0.0,
                search_strategy='empty_fallback'
            )
                
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return GraphRAGResult(
                vector_documents=[],
                graph_result=GraphResult(entities=[], relationships=[], paths=[]),
                vector_score=0.0,
                graph_score=0.0,
                search_strategy='error_fallback'
            )

    def find_content_by_chapter_section(self, chapter_num: int, section_num: Optional[int] = None, content_types: Optional[List[str]] = None, limit: int = 5) -> List[Document]:
        """Find content by chapter and optionally section number, filtering by content types.
        
        Args:
            chapter_num: Chapter number to search in
            section_num: Optional section number within the chapter
            content_types: Optional list of content types (labels) to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of documents from the specified chapter/section matching content types
        """
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                conditions = [
                    "n.text_content IS NOT NULL", 
                    "n.text_content <> \"\"",
                    "n.chapter_number = $chapter_num"
                ]
                params = {"chapter_num": chapter_num, "limit": limit}
                
                if section_num is not None:
                    conditions.append("n.section_number = $section_num")
                    params["section_num"] = section_num
                
                # Filter by specific node labels for educational content
                # Build the MATCH clause with proper OR conditions for node labels
                node_match_conditions = ["n:Problem", "n:Exercise", "n:Solution", "n:Example", "n:Para", "n:Section"]

                # If specific content_types are requested, filter the match conditions
                if content_types:
                    valid_requested_types = []
                    # Ensure requested types are valid educational labels
                    for ct in content_types:
                         # Basic validation: ensure it's a plausible label format (alphanumeric)
                        if re.match(r"^[A-Za-z0-9_]+$", ct):
                            valid_requested_types.append(ct)
                    
                    if valid_requested_types:
                        # Filter node_match_conditions to only include requested types
                        node_match_conditions = [f"n:{ct}" for ct in valid_requested_types]
                    else:
                        # If content_types were specified but none are valid, return empty
                        return []

                where_clause = " AND ".join(conditions)
                
                # Build the MATCH clause with OR conditions for node labels
                match_clause = f"MATCH (n) WHERE ({' OR '.join(node_match_conditions)}) AND {where_clause}"
                
                # The complete query
                query = f"""
                {match_clause}
                WITH n, 
                     labels(n)[0] as primary_label,
                     CASE 
                        WHEN n.title IS NOT NULL AND n.title <> '' THEN n.title
                        WHEN n.id IS NOT NULL THEN 'Chapter ' + toString(n.chapter_number) + ' - ' + n.id
                        ELSE 'Chapter ' + toString(n.chapter_number) + ' Content'
                     END as content_title,
                     CASE labels(n)[0] // Prioritize based on the primary label
                        WHEN 'Para' THEN 1
                        WHEN 'Section' THEN 1
                        ELSE 2 // Exercises, Problems, Solutions, Examples get higher priority for display
                     END as content_priority
                RETURN n.text_content as content,
                       n.id as node_id,
                       primary_label as content_type,
                       content_title as title,
                       n.module_id as module_id,
                       n.chapter_number as chapter_number,
                       null as section_number,  // Most nodes don't have section_number
                       content_priority,
                       n.semantic_keywords as keywords,
                       properties(n) as all_properties
                ORDER BY content_priority DESC, size(n.text_content) DESC
                LIMIT $limit
                """

                result = session.run(query, params)
                documents = []
                
                for record in result:
                    # Get all properties and filter out text_content
                    all_props = record["all_properties"] or {}
                    filtered_props = {k: v for k, v in all_props.items() 
                                    if k != "text_content" and v is not None}
                    
                    # Create comprehensive metadata
                    metadata = {
                        'id': record["node_id"],
                        'type': record["content_type"],
                        'title': record["title"],
                        'module_id': record["module_id"],
                        'chapter_number': record["chapter_number"],
                        'section_number': record["section_number"],
                        'content_priority': record["content_priority"],
                        'semantic_keywords': record.get("keywords"),
                        'search_strategy': 'chapter_section',
                        **filtered_props  # Include all other node properties
                    }
                    
                    doc = Document(
                        page_content=record["content"],
                        metadata=metadata
                    )
                    documents.append(doc)
                
                return documents

        except Exception as e:
            logger.error(f"Error in find_content_by_chapter_section: {e}")
            return []

    def analyze_content_quality(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze the quality and characteristics of retrieved educational content.
        
        This method provides detailed analysis of content quality, distribution,
        and educational value to help assess retrieval effectiveness.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Dictionary containing quality metrics and analysis
        """
        if not documents:
            return {
                'total_documents': 0,
                'total_content_length': 0,
                'average_length': 0,
                'content_types': {},
                'quality_scores': [],
                'educational_coverage': 'None'
            }
        
        # Basic statistics
        total_docs = len(documents)
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_length = total_chars / total_docs if total_docs > 0 else 0
        
        # Content type distribution
        content_types = {}
        quality_scores = []
        
        for doc in documents:
            content_type = doc.metadata.get('type', 'Unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Calculate quality score for each document
            quality_score = self._calculate_content_quality_score(doc)
            quality_scores.append(quality_score)
        
        # Determine educational coverage
        educational_types = {'Exercise', 'Example', 'Problem', 'Solution', 'Para'}
        found_types = set(content_types.keys())
        coverage_ratio = len(found_types.intersection(educational_types)) / len(educational_types)
        
        if coverage_ratio >= 0.6:
            educational_coverage = 'Comprehensive'
        elif coverage_ratio >= 0.4:
            educational_coverage = 'Good'
        elif coverage_ratio >= 0.2:
            educational_coverage = 'Partial'
        else:
            educational_coverage = 'Limited'
        
        return {
            'total_documents': total_docs,
            'total_content_length': total_chars,
            'average_length': avg_length,
            'content_types': content_types,
            'quality_scores': quality_scores,
            'average_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'educational_coverage': educational_coverage,
            'coverage_ratio': coverage_ratio
        }
    
    def _calculate_content_quality_score(self, document: Document) -> float:
        """Calculate a quality score for a single document.
        
        Args:
            document: Document to score
            
        Returns:
            Quality score (higher is better)
        """
        score = 0.0
        content = document.page_content
        metadata = document.metadata
        
        # Content length factor (normalized)
        content_length = len(content)
        if content_length > 50:
            score += min(content_length / 200, 3.0)  # Max 3 points for length
        
        # Title quality
        title = metadata.get('title', '')
        if title and title != 'Untitled':
            score += 1.0
            if len(title) > 10:  # Descriptive title
                score += 0.5
        
        # Content structure indicators
        if '$' in content:  # Mathematical notation
            score += 1.0
        if any(keyword in content.lower() for keyword in ['example', 'problem', 'solution', 'exercise', 'definition', 'concept']):
            score += 1.0
        if any(keyword in content.lower() for keyword in ['step', 'first', 'then', 'finally', 'therefore', 'because']):
            score += 0.5  # Structured explanation
        
        # Content type bonus - more balanced across types
        content_type = metadata.get('type', 'Unknown')
        if content_type in ['Exercise', 'Problem']:
            score += 1.5  # Practice material
        elif content_type in ['Example', 'Solution']:
            score += 1.5  # Demonstrative content
        elif content_type == 'Para':
            score += 1.5  # Explanatory content
        elif content_type == 'Section':
            score += 1.0  # Structural content
        
        # Semantic richness
        semantic_keywords = metadata.get('semantic_keywords', [])
        if semantic_keywords and isinstance(semantic_keywords, list):
            score += min(len(semantic_keywords) * 0.2, 1.0)  # Up to 1 point for semantic richness
        
        # Chapter context
        if metadata.get('chapter_number') is not None:
            score += 0.5  # Organized in curriculum
        
        return min(score, 10.0)  # Cap at 10
    
    def filter_high_quality_content(
        self, 
        documents: List[Document], 
        min_quality_score: float = 3.0,
        max_results: Optional[int] = None
    ) -> List[Document]:
        """Filter documents to return only high-quality educational content.
        
        Args:
            documents: List of documents to filter
            min_quality_score: Minimum quality score threshold
            max_results: Maximum number of results to return
            
        Returns:
            Filtered list of high-quality documents
        """
        # Calculate quality scores and filter
        scored_docs = []
        for doc in documents:
            quality_score = self._calculate_content_quality_score(doc)
            if quality_score >= min_quality_score:
                scored_docs.append((doc, quality_score))
        
        # Sort by quality score (highest first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract documents and apply limit
        filtered_docs = [doc for doc, _ in scored_docs]
        if max_results:
            filtered_docs = filtered_docs[:max_results]
        
        return filtered_docs
    
    def organize_educational_content(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Organize documents by educational content type in learning progression order.
        
        Args:
            documents: List of documents to organize
            
        Returns:
            Dictionary with content organized by type in educational order
        """
        # Define educational progression order
        educational_order = ['Example', 'Exercise', 'Problem', 'Solution', 'Para', 'Section']
        
        # Group documents by type
        content_by_type = {}
        for doc in documents:
            content_type = doc.metadata.get('type', 'Unknown')
            if content_type not in content_by_type:
                content_by_type[content_type] = []
            content_by_type[content_type].append(doc)
        
        # Sort each type by quality score
        for content_type in content_by_type:
            docs_with_scores = []
            for doc in content_by_type[content_type]:
                quality_score = self._calculate_content_quality_score(doc)
                docs_with_scores.append((doc, quality_score))
            
            # Sort by quality (highest first)
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            content_by_type[content_type] = [doc for doc, _ in docs_with_scores]
        
        # Return in educational progression order
        organized_content = {}
        for content_type in educational_order:
            if content_type in content_by_type:
                organized_content[content_type] = content_by_type[content_type]
        
        # Add any remaining types not in the standard order
        for content_type, docs in content_by_type.items():
            if content_type not in organized_content:
                organized_content[content_type] = docs
        
        return organized_content

    def parse_educational_query(self, query: str) -> Dict[str, Any]:
        """Parse an educational query to extract key parameters.
        
        Args:
            query: The educational query to parse
            
        Returns:
            Dictionary containing parsed parameters
        """
        # Default values
        parsed = {
            'original_query': query,
            'chapter': None,
            'section': None,
            'content_types': ['Problem', 'Exercise', 'Solution', 'Example', 'Para', 'Section'],
            'learning_intent': 'understand',
            'keywords': query.lower().split()
        }
        
        # Extract chapter/section numbers
        chapter_match = re.search(r'chapter\s*(\d+)', query.lower())
        section_match = re.search(r'section\s*(\d+)', query.lower())
        
        if chapter_match:
            parsed['chapter'] = int(chapter_match.group(1))
        if section_match:
            parsed['section'] = int(section_match.group(1))
            
        # Detect learning intent
        if any(word in query.lower() for word in ['explain', 'what is', 'what are', 'define']):
            parsed['learning_intent'] = 'explain'
            parsed['content_types'] = ['Para', 'Section', 'Example']
        elif any(word in query.lower() for word in ['practice', 'exercise', 'problem']):
            parsed['learning_intent'] = 'practice'
            parsed['content_types'] = ['Exercise', 'Problem']
        elif any(word in query.lower() for word in ['example', 'show me']):
            parsed['learning_intent'] = 'example'
            parsed['content_types'] = ['Example', 'Solution']
        elif any(word in query.lower() for word in ['review', 'summary']):
            parsed['learning_intent'] = 'review'
            parsed['content_types'] = ['Para', 'Section']
            
        # Extract semantic keywords (excluding common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        keywords = [word.lower() for word in query.split() if word.lower() not in stop_words]
        parsed['keywords'] = keywords
        
        return parsed

    def hybrid_search(
        self,
        query: str,
        k: int = None,
        alpha: float = None,
        search_type: str = "hybrid"
    ) -> GraphRAGResult:
        """
        Perform true hybrid search combining vector and fulltext search.
        
        This method uses Neo4jVector's built-in hybrid search that combines:
        - Dense vector search (semantic similarity)
        - Fulltext search (keyword matching)
        - Score combination and re-ranking
        - Result deduplication
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for score combination (0=fulltext only, 1=vector only)
            search_type: Type of search ("hybrid", "vector", "fulltext")
            
        Returns:
            GraphRAGResult with hybrid search results
        """
        k = k or self.vector_top_k
        alpha = alpha if alpha is not None else self.hybrid_alpha
        
        if not self.neo4j_vector:
            logger.warning("Hybrid search not available - falling back to standard vector search")
            return self.retrieve(query, vector_k=k)
        
        try:
            logger.info(f"Performing hybrid search with alpha={alpha}, search_type={search_type}")
            
            # Use Neo4jVector's hybrid search
            if search_type == "hybrid":
                # Perform hybrid search with score
                results = self.neo4j_vector.similarity_search_with_score(query, k=k)
            elif search_type == "vector":
                # Pure vector search
                vector_store_copy = Neo4jVector(
                    embedding=self.neo4j_vector.embedding,
                    url=self.neo4j_url,
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                    database=self.neo4j_database,
                    index_name="educational_problem_embeddings",  # Use the new comprehensive indexes
                    node_label=self.vector_node_label,
                    text_node_property=self.vector_text_node_property,
                    embedding_node_property=self.vector_embedding_node_property,
                    search_type="vector"
                )
                results = vector_store_copy.similarity_search_with_score(query, k=k)
            else:  # fulltext
                results = self._fulltext_search_with_score(query, k)
            
            # Extract documents and scores
            if results:
                documents, scores = zip(*results)
                documents = list(documents)
                scores = list(scores)
                
                # Add hybrid search metadata
                for doc in documents:
                    doc.metadata["search_strategy"] = search_type
                    doc.metadata["hybrid_alpha"] = alpha
                
                logger.info(f"Hybrid search returned {len(documents)} results")
                
                return GraphRAGResult(
                    vector_documents=documents,
                    graph_result=GraphResult(entities=[], relationships=[], paths=[]),
                    vector_score=sum(scores) / len(scores) if scores else 0,
                    hybrid_scores={"combined": scores},
                    search_strategy=search_type,
                    vector_query=query
                )
            else:
                logger.warning("No results from hybrid search")
                return GraphRAGResult(
                    vector_documents=[],
                    graph_result=GraphResult(entities=[], relationships=[], paths=[]),
                    search_strategy=search_type,
                    vector_query=query
                )
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to standard retrieval
            return self.retrieve(query, vector_k=k)
    
    def _fulltext_search_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform fulltext search using Neo4j fulltext index."""
        try:
            driver = GraphDatabase.driver(
                self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password)
            )
            
            with driver.session(database=self.neo4j_database) as session:
                fulltext_query = f"""
                CALL db.index.fulltext.queryNodes(
                    '{self.fulltext_index_name}', 
                    $query
                ) YIELD node, score
                RETURN node, score
                ORDER BY score DESC
                LIMIT $k
                """
                
                result = session.run(fulltext_query, query=query, k=k)
                
                results = []
                for record in result:
                    node = record["node"]
                    score = record["score"]
                    
                    text_content = node.get(self.vector_text_node_property, "")
                    if not text_content:
                        continue
                    
                    metadata = {
                        "node_id": node.id,
                        "labels": list(node.labels),
                        "search_type": "fulltext",
                        **{k: v for k, v in node.items() if k != self.vector_text_node_property}
                    }
                    
                    doc = Document(page_content=text_content, metadata=metadata)
                    results.append((doc, score))
                
                driver.close()
                return results
                
        except Exception as e:
            logger.error(f"Error in fulltext search: {e}")
            return []
    
    def retrieve_by_content_type(
        self,
        query: str,
        content_type: str,
        vector_k: Optional[int] = None
    ) -> GraphRAGResult:
        """Retrieve educational content of a specific type.
        
        This method allows for retrieval of specific educational content types
        like examples, practice problems, or explanations.
        
        Args:
            query: Query text
            content_type: Type of content to retrieve (e.g., "Example", "Practice", "Explanation")
            vector_k: Number of vector results to return (overrides default)
            
        Returns:
            GraphRAGResult containing vector and graph results filtered by content type
        """
        return self.retrieve_by_learning_objective(
            query=query,
            content_types=[content_type],
            vector_k=vector_k
        )
    
    def retrieve_for_retrieval_practice(
        self,
        topic: str,
        difficulty: Optional[str] = None,
        vector_k: Optional[int] = None
    ) -> GraphRAGResult:
        """Retrieve content specifically for retrieval practice.
        
        This method supports educational retrieval practice by finding
        content that helps reinforce key concepts through recall.
        
        Args:
            topic: Topic to practice
            difficulty: Difficulty level (e.g., "Basic", "Intermediate", "Advanced")
            vector_k: Number of vector results to return (overrides default)
            
        Returns:
            GraphRAGResult containing content for retrieval practice
        """
        # Focus query on finding practice problems for retrieval
        practice_query = f"practice problems for {topic}"
        
        # Target content types that support retrieval practice
        content_types = ["Exercise", "Problem", "Example", "Practice"]
        
        return self.retrieve_by_learning_objective(
            query=practice_query,
            content_types=content_types,
            difficulty=difficulty,
            vector_k=vector_k
        )
    
    def retrieve_conceptual_relationships(
        self,
        concept: str,
        vector_k: Optional[int] = None
    ) -> GraphRAGResult:
        """Retrieve content showing relationships between concepts.
        
        This method helps build knowledge graphs by finding related concepts
        and their relationships within the educational content.
        
        Args:
            concept: The central concept to explore relationships for
            vector_k: Number of vector results to return (overrides default)
            
        Returns:
            GraphRAGResult containing conceptual relationship information
        """
        vector_docs = []
        vector_avg_score = 0.0
        graph_result = GraphResult(entities=[], relationships=[], paths=[])
        graph_score = 0.0
        
        try:
            logger.info(f"Retrieving conceptual relationships for: {concept}")
            
            # Connect to Neo4j
            driver = GraphDatabase.driver(
                self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password)
            )
            
            with driver.session(database=self.neo4j_database) as session:
                # Use the comprehensive educational indexes for concept search
                educational_indexes = [
                    "educational_problem_embeddings",
                    "educational_exercise_embeddings", 
                    "educational_solution_embeddings",
                    "educational_example_embeddings",
                    "educational_para_embeddings"
                ]
                
                # Get the embedding for the concept
                embedding = self.embedding_model.embed_query(concept)
                
                vector_results = []
                node_ids = []
                
                # Query each educational index
                for index_name in educational_indexes:
                    try:
                        vector_query = f"""
                        CALL db.index.vector.queryNodes(
                            '{index_name}',
                            $top_k_per_index,
                            $embedding
                        ) YIELD node, score
                        RETURN node, score
                        """
                        
                        result = session.run(
                            vector_query, 
                            {
                                "embedding": embedding, 
                                "top_k_per_index": max(1, (vector_k or self.vector_top_k) // len(educational_indexes))
                            }
                        )
                        
                        for record in result:
                            node = record["node"]
                            score = record["score"]
                            node_ids.append(id(node))
                            
                            # Convert Neo4j node to document
                            metadata = dict(node)
                            content = metadata.get(self.vector_text_node_property, "")
                            
                            # If text_content is not available, try other content properties
                            if not content:
                                for prop in ["content", "text", "problem_text", "solution_text"]:
                                    if prop in metadata:
                                        content = metadata.get(prop)
                                        break
                            
                            if not content and "title" in metadata:
                                content = metadata.get("title", "")
                            
                            doc = Document(page_content=content, metadata=metadata)
                            vector_results.append((doc, score))
                            
                    except Exception as e:
                        logger.warning(f"Error querying index {index_name} for concept: {e}")
                        continue
                
                # If we have results, extract documents and calculate average score
                if vector_results:
                    vector_docs = [doc for doc, _ in vector_results]
                    vector_avg_score = sum(score for _, score in vector_results) / len(vector_results)
                    logger.info(f"Found {len(vector_docs)} documents related to concept with avg score: {vector_avg_score}")
                    
                    # Now build a knowledge graph around these concepts
                    if node_ids:
                        node_ids_str = ", ".join([str(nid) for nid in node_ids])
                        
                        graph_query = f"""
                        // Start with nodes found in vector search
                        MATCH (n) 
                        WHERE id(n) IN [{node_ids_str}]
                        
                        // Find related concepts within 2 hops
                        OPTIONAL MATCH path = (n)-[r:RELATED_TO|PREREQUISITE_FOR|REFERENCES|CONTAINS*1..2]-(related)
                        
                        // Collect all unique nodes and relationships
                        WITH collect(DISTINCT n) + collect(DISTINCT related) AS all_nodes,
                             collect(DISTINCT r) AS all_rels
                        
                        // Format nodes as entities
                        WITH 
                        [node IN all_nodes WHERE node IS NOT NULL |
                            {{
                                id: id(node),
                                name: COALESCE(node.name, node.title, ''),
                                type: labels(node)[0],
                                properties: properties(node)
                            }}
                        ] AS entities,
                        
                        // Format relationships
                        [rel IN all_rels WHERE rel IS NOT NULL |
                            {{
                                source: id(startNode(rel)),
                                target: id(endNode(rel)),
                                type: type(rel),
                                properties: properties(rel)
                            }}
                        ] AS relationships
                        
                        // Return everything
                        RETURN entities, relationships
                        """
                        
                        result = session.run(graph_query)
                        
                        if result.peek():
                            record = result.single()
                            graph_result = GraphResult(
                                entities=record["entities"],
                                relationships=record["relationships"],
                                paths=[]
                            )
                            
                            # Set graph score based on number of entities and relationships
                            entity_count = len(graph_result.entities)
                            rel_count = len(graph_result.relationships)
                            
                            if entity_count > 0 or rel_count > 0:
                                graph_score = (entity_count + rel_count) / 10  # Normalize to approximate 0-1 range
                                logger.info(f"Built concept graph with {entity_count} entities and {rel_count} relationships")
                else:
                    logger.warning(f"No vector results found for concept: {concept}")
                
                driver.close()
        
        except Exception as e:
            logger.error(f"Conceptual relationship retrieval failed: {e}")
        
        # Return combined results
        return GraphRAGResult(
            vector_documents=vector_docs,
            graph_result=graph_result,
            vector_score=vector_avg_score,
            graph_score=graph_score
        ) 