"""
Vector Retrieval Engine for GraphRAG.

This module implements semantic search using vector embeddings stored in Neo4j.
It converts document chunks to vector embeddings and retrieves information 
based on semantic similarity.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import re

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
        index_name: str = "educational_problem_embeddings",
        text_node_property: str = "text_content",
        embedding_node_property: str = "fastRP_embedding",
        embedding_dimension: int = 512,
        distance_metric: str = "cosine",
        node_label: str = "Problem",
        search_type: str = "hybrid"
    ):
        """Initialize the Vector Retrieval Engine.
        
        Args:
            url: Neo4j connection URL
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            embedding_model: Embedding model for generating embeddings
            index_name: Name of vector index in Neo4j
            text_node_property: Property containing text
            embedding_node_property: Property containing embeddings
            embedding_dimension: Dimension of embeddings
            distance_metric: Distance metric for similarity
            node_label: Label of document nodes
            search_type: Type of search to perform ('hybrid', 'vector', 'text')
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
        self.search_type = search_type
        self.vector_store = None
        
        # Create Neo4j driver for direct queries
        self.driver = GraphDatabase.driver(url, auth=(username, password))
        
    def get_node_count(self) -> int:
        """Get count of nodes that have embeddings.
        
        Returns:
            Number of nodes with embeddings
        """
        try:
            with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH (n:{self.node_label})
                WHERE n.{self.embedding_node_property} IS NOT NULL
                RETURN COUNT(n) as count
                """
                result = session.run(query)
                return result.single()["count"] if result.peek() else 0
        except Exception as e:
            logger.error(f"Error getting node count: {e}")
            return 0
        
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
        """Initialize the vector store with Neo4j."""
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
                text_node_property=self.text_node_property,
                embedding_node_property=self.embedding_node_property,
                node_label=self.node_label,
                search_type=self.search_type
            )
            logger.info(f"Vector store initialized with search type: {self.search_type}")
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
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        """Perform similarity search for a query across all educational content types.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of relevant documents from all educational content types
        """
        try:
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Define all educational indexes
            educational_indexes = [
                ("educational_problem_embeddings", "Problem"),
                ("educational_exercise_embeddings", "Exercise"),
                ("educational_solution_embeddings", "Solution"),
                ("educational_example_embeddings", "Example"),
                ("educational_para_embeddings", "Para")
            ]
            
            all_results = []
            
            # Query each educational index
            with self.driver.session(database=self.database) as session:
                for index_name, content_type in educational_indexes:
                    try:
                        # Query this specific index
                        search_query = f"""
                        CALL db.index.vector.queryNodes(
                            '{index_name}',
                            $top_k_per_type,
                            $embedding
                        ) YIELD node, score
                        WHERE node.{self.text_node_property} IS NOT NULL
                        AND node.{self.text_node_property} <> ''
                        AND size(node.{self.text_node_property}) > 20
                        WITH node, score,
                             labels(node)[0] as primary_label,
                             CASE 
                                WHEN node.title IS NOT NULL AND node.title <> '' THEN node.title
                                WHEN node.id IS NOT NULL THEN 'Chapter ' + toString(node.chapter_number) + ' - ' + node.id
                                ELSE 'Chapter ' + toString(node.chapter_number) + ' Content'
                             END as content_title
                        RETURN node.{self.text_node_property} as content,
                               node.id as node_id,
                               primary_label as content_type,
                               content_title as title,
                               node.module_id as module_id,
                               node.chapter_number as chapter_number,
                               null as section_number,
                               node.semantic_keywords as keywords,
                               score,
                               'vector_search' as search_strategy,
                               properties(node) as all_properties
                        ORDER BY score DESC
                        """
                        
                        # Get results for this content type (limit per type to ensure diversity)
                        results_per_type = max(1, top_k // len(educational_indexes))
                        
                        result = session.run(
                            search_query,
                            embedding=query_embedding,
                            top_k_per_type=results_per_type
                        )
                        
                        # Process results for this content type
                        for record in result:
                            try:
                                # Extract metadata with proper defaults
                                metadata = {
                                    'node_id': record.get('node_id', 'unknown'),
                                    'type': record.get('content_type', content_type),
                                    'title': record.get('title', f'{content_type} Content'),
                                    'module_id': record.get('module_id', 'Unknown'),
                                    'chapter_number': record.get('chapter_number', 'N/A'),
                                    'section_number': record.get('section_number', 'N/A'),
                                    'keywords': record.get('keywords', []),
                                    'score': float(record.get('score', 0.0)),
                                    'search_strategy': record.get('search_strategy', 'vector_search'),
                                    'source_index': index_name
                                }
                                
                                # Create document
                                doc = Document(
                                    page_content=record.get('content', ''),
                                    metadata=metadata
                                )
                                all_results.append(doc)
                                
                            except Exception as e:
                                logger.warning(f"Error processing result from {index_name}: {e}")
                                continue
                                
                    except Exception as e:
                        logger.warning(f"Error querying index {index_name}: {e}")
                        continue
            
            # Sort all results by score and return top_k
            all_results.sort(key=lambda x: x.metadata.get('score', 0.0), reverse=True)
            final_results = all_results[:top_k]
            
            logger.info(f"Vector search returned {len(final_results)} results from {len(set(r.metadata.get('type') for r in final_results))} content types")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            # Fallback to text search
            return self._custom_text_search(query, top_k)
    
    def _check_vector_dimensions(self) -> Dict[str, Any]:
        """Check the dimensions of vectors in the index and return compatibility info."""
        result = {"compatible": True, "message": "", "vector_dimensions": self.embedding_dimension}
        try:
            with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH (n:{self.node_label})
                WHERE n.{self.embedding_node_property} IS NOT NULL
                WITH n LIMIT 1
                RETURN size(n.{self.embedding_node_property}) AS dimensions
                """
                record = session.run(query).single()
                if record:
                    result["vector_dimensions"] = record["dimensions"]
                    if record["dimensions"] != self.embedding_dimension:
                        result["compatible"] = False
                        result["message"] = f"Expected {self.embedding_dimension}, got {record['dimensions']}"
        except Exception as e:
            result["compatible"] = False
            result["message"] = f"Error checking vector dimensions: {e}"
        return result
            
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for the query text."""
        embeddings = self.embedding_model.embed_query(query)
        return embeddings
    
    def _custom_text_search(self, query: str, top_k: int) -> List[Document]:
        """Perform text-based search when vector search fails.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            # Get content-bearing node labels
            content_labels = self.get_content_bearing_nodes()
            if not content_labels:
                logger.warning("No content-bearing nodes found")
                return []
            
            # Split query into words for better matching
            query_words = query.lower().split()
            
            with self.driver.session(database=self.database) as session:
                # Enhanced text search with proper metadata extraction
                search_query = f"""
                MATCH (n)
                WHERE any(label in labels(n) WHERE label IN ['Problem', 'Exercise', 'Solution', 'Example', 'Para'])
                AND n.{self.text_node_property} IS NOT NULL
                AND n.{self.text_node_property} <> ''
                AND size(n.{self.text_node_property}) > 20
                AND any(word IN split(toLower(n.{self.text_node_property}), ' ') 
                       WHERE any(q IN $query_words WHERE word CONTAINS q))
                WITH n,
                     labels(n)[0] as primary_label,
                     CASE 
                        WHEN n.title IS NOT NULL AND n.title <> '' THEN n.title
                        WHEN n.id IS NOT NULL THEN primary_label + ' - ' + n.id
                        ELSE primary_label + ' Content'
                     END as content_title,
                     COALESCE(n.chapter_number, 'N/A') as chapter_num,
                     COALESCE(n.module_id, 'Unknown') as mod_id,
                     size([word IN split(toLower(n.{self.text_node_property}), ' ') 
                          WHERE any(q IN $query_words WHERE word CONTAINS q)]) as match_count
                RETURN n.{self.text_node_property} as content,
                       n.id as node_id,
                       primary_label as content_type,
                       content_title as title,
                       chapter_num as chapter_number,
                       mod_id as module_id,
                       match_count,
                       n.semantic_keywords as keywords,
                       properties(n) as all_properties
                ORDER BY match_count DESC, size(n.{self.text_node_property}) DESC
                LIMIT $top_k
                """
                
                result = session.run(search_query, query_words=query_words, top_k=top_k)
                
                docs = []
                for record in result:
                    # Get all properties and filter out text_content
                    all_props = record["all_properties"] or {}
                    filtered_props = {k: v for k, v in all_props.items() 
                                    if k != self.text_node_property and v is not None}
                    
                    # Create comprehensive metadata
                    metadata = {
                        "id": record["node_id"],
                        "type": record["content_type"],
                        "title": record["title"],
                        "chapter_number": record["chapter_number"],
                        "module_id": record["module_id"],
                        "match_count": record["match_count"],
                        "semantic_keywords": record.get("keywords"),
                        "search_strategy": "text_search",
                        **filtered_props  # Include all other node properties
                    }
                    
                    # Create document
                    doc = Document(
                        page_content=record["content"],
                        metadata=metadata
                    )
                    docs.append(doc)
                
                return docs
                
        except Exception as e:
            logger.error(f"Error in custom text search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, top_k: int = 4, **kwargs):
        """Perform similarity search and return documents with their scores.
        
        Args:
            query: Query text
            top_k: Number of results to return 
            **kwargs: Additional arguments for the search
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vector_store:
            self.init_vector_store()
            
        # Check if we have documents with embeddings
        if not self.check_document_existence():
            logger.warning("No documents with embeddings found in the database. Vector search returning empty results.")
            return []
        
        try:
            # Get base documents from similarity search
            docs = self.similarity_search(query, top_k=top_k)
            
            # Compute quality scores
            docs_with_scores = []
            for doc in docs:
                quality_score = self._compute_quality_score(doc, query)
                docs_with_scores.append((doc, quality_score))
                
            # Sort by quality score
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            return docs_with_scores
                
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
    
    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """Get relevant documents for a query based on vector similarity.
        
        Args:
            query: Query text to search for
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents
        """
        try:
            logger.info(f"Found {self.get_node_count()} {self.node_label} nodes with embeddings")
            
            # Dimension mismatch handling - check vector dimensions
            dimension_info = self._check_vector_dimensions()
            embedding_dimensions = dimension_info.get("vector_dimensions", 512)
            
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Check for dimension mismatch
            if len(query_embedding) != embedding_dimensions:
                logger.warning(f"Query embedding dimension ({len(query_embedding)}) doesn't match index dimension ({embedding_dimensions})")
                
                # Use custom Cypher query instead of Neo4jVector's similarity_search
                return self._custom_vector_search(query_embedding, top_k, embedding_dimensions)
            
            # If dimensions match, use standard Neo4jVector's similarity_search
            docs = self.vector_store.similarity_search(query, k=top_k)
            return docs
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def _custom_vector_search(self, query_embedding: List[float], top_k: int, index_dimensions: int) -> List[Document]:
        """Perform custom vector search when dimensions don't match.
        
        Args:
            query_embedding: The full-dimensional query embedding
            top_k: Number of top results to return
            index_dimensions: Dimensions of vectors in the index
            
        Returns:
            List of relevant documents
        """
        docs = []
        try:
            # Since we can't directly compare embeddings with different dimensions,
            # we'll use a simpler approach - text search on key properties
            with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH (n:{self.node_label})
                WHERE n.{self.text_node_property} IS NOT NULL
                RETURN n.{self.text_node_property} AS content, 
                       n.title AS title,
                       apoc.meta.id(n) AS id
                LIMIT {top_k}
                """
                
                result = session.run(query)
                for record in result:
                    metadata = {
                        "id": str(record["id"]),
                        "title": record.get("title", "")
                    }
                    content = record["content"]
                    docs.append(Document(page_content=content, metadata=metadata))
                    
        except Exception as e:
            logger.error(f"Error in custom vector search: {e}")
            
        return docs

    def get_content_bearing_nodes(self) -> List[str]:
        """Get list of node labels that contain actual content.
        
        Returns:
            List of node labels that contain content
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Look for nodes that have text content
                result = session.run("""
                    MATCH (n)
                    WHERE n.text_content IS NOT NULL 
                    AND n.text_content <> ''
                    AND NOT n.text_content =~ '.*[0-9]+.*'  // Exclude nodes with just IDs
                    WITH DISTINCT labels(n) AS labels
                    RETURN labels
                """)
                
                content_labels = []
                for record in result:
                    if record["labels"]:
                        content_labels.extend(record["labels"])
                
                return list(set(content_labels))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error getting content-bearing nodes: {e}")
            return []

    def _compute_quality_score(self, doc: Document, query: str) -> float:
        """Compute a meaningful quality score for a document.
        
        The score is based on multiple factors:
        1. Content completeness (0-0.3)
        2. Relevance to query (0-0.4) 
        3. Content type appropriateness (0-0.3)
        
        Args:
            doc: Document to score
            query: Original query string
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # 1. Content completeness (0-0.3)
        content = doc.page_content
        word_count = len(content.split())
        if word_count > 200:
            score += 0.3
        elif word_count > 100:
            score += 0.2
        elif word_count > 50:
            score += 0.1
            
        # 2. Relevance to query (0-0.4)
        # Use metadata score which combines vector and text similarity
        relevance = doc.metadata.get("score", 0)
        score += 0.4 * relevance
        
        # 3. Content type appropriateness (0-0.3)
        content_type = doc.metadata.get("type", "")
        query_lower = query.lower()
        
        # Map query intent to appropriate content types
        if any(term in query_lower for term in ["explain", "what is", "how do", "why"]):
            if content_type in ["Para", "Example"]:
                score += 0.3
            elif content_type == "Solution":
                score += 0.2
        elif any(term in query_lower for term in ["practice", "exercise", "problem"]):
            if content_type in ["Exercise", "Problem"]:
                score += 0.3
            elif content_type == "Example":
                score += 0.2
        elif any(term in query_lower for term in ["solution", "answer", "solve"]):
            if content_type == "Solution":
                score += 0.3
            elif content_type == "Example":
                score += 0.2
        else:
            # Default scoring for other queries
            if content_type in ["Para", "Example"]:
                score += 0.2
            else:
                score += 0.1
                
        return score 