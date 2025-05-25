"""
Metadata Retrieval for GraphRAG.

This module implements enhanced metadata filtering and relationship-aware retrieval
to improve query results by leveraging graph structure.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from neo4j import GraphDatabase
import warnings

logger = logging.getLogger(__name__)

class MetadataRetriever:
    """
    Retriever that enhances vector search with metadata filtering and relationship traversal.
    """
    
    def __init__(
        self,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
    ):
        """
        Initialize the MetadataRetriever.
        
        Args:
            neo4j_url: URL for the Neo4j database
            neo4j_username: Username for Neo4j authentication
            neo4j_password: Password for Neo4j authentication
            neo4j_database: Neo4j database name
        """
        self.driver = GraphDatabase.driver(
            neo4j_url, auth=(neo4j_username, neo4j_password)
        )
        self.database = neo4j_database
        
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        
    def retrieve_with_relationships(
        self,
        query: str,
        entity_type: str,
        relationship_types: List[str] = None,
        max_depth: int = 1,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve content based on both semantic similarity and relationship traversal.
        
        Args:
            query: The search query
            entity_type: Type of entity to start with
            relationship_types: Types of relationships to traverse
            max_depth: Maximum relationship traversal depth
            limit: Maximum number of results to return
            
        Returns:
            List of result documents with their metadata and relationships
        """
        if relationship_types is None:
            relationship_types = ["RELATED", "PREREQUISITE_FOR", "HAS_EXAMPLE"]
            
        # Build the relationship pattern dynamically - fix the syntax
        rel_pattern = "|".join([f"{rel}" for rel in relationship_types])
        
        # For demonstration purposes, use text search instead of vector search
        # to avoid dimension mismatch issues
        cypher_query = f"""
        // Find nodes of the specified type that match the query text
        MATCH (node)
        WHERE '{entity_type}' IN labels(node)
        AND (
            toLower(node.text_content) CONTAINS toLower($query_text)
            OR toLower(node.title) CONTAINS toLower($query_text)
            OR toLower(node.name) CONTAINS toLower($query_text)
        )
        
        // Find related nodes through specified relationship types
        WITH node
        MATCH path = (node)-[r:{rel_pattern}*1..{max_depth}]-(related)
        
        // Return with related nodes
        RETURN node, collect(distinct related) as related_nodes, 1.0 as score
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                cypher_query,
                query_text=query,
                limit=limit
            )
            
            return [self._process_result(record) for record in result]
    
    def retrieve_with_metadata_filter(
        self,
        query: str,
        filters: Dict[str, Any],
        node_labels: List[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve content with metadata filtering before vector similarity.
        
        Args:
            query: The search query
            filters: Dictionary of property filters {property: value}
            node_labels: Node labels to include in search
            limit: Maximum number of results
            
        Returns:
            List of result documents with their metadata
        """
        if node_labels is None:
            node_labels = ["Problem", "Example", "Solution", "Section"]
            
        # Build the WHERE clause for metadata filtering
        where_clauses = []
        params = {"embedding": query, "top_k": limit * 2, "limit": limit}
        
        for idx, (key, value) in enumerate(filters.items()):
            param_name = f"param_{idx}"
            where_clauses.append(f"node.{key} = ${param_name}")
            params[param_name] = value
            
        where_clause = " AND ".join(where_clauses) if where_clauses else "true"
        
        # Create the labels expression with proper syntax (using | not |:)
        if node_labels and len(node_labels) > 0:
            labels_clause = ":" + "|".join([label for label in node_labels])
        else:
            labels_clause = ""
        
        # First apply the metadata filtering, then do the vector search separately
        cypher_query = f"""
        // First get nodes matching metadata filters
        MATCH (node{labels_clause})
        WHERE {where_clause}
        
        // Then perform vector search on filtered nodes
        WITH collect(node) as filtered_nodes
        UNWIND filtered_nodes as node
        
        // Now do the vector search on these filtered nodes
        CALL db.index.vector.queryNodes('content_embeddings', $top_k, $embedding)
        YIELD node as vector_node, score
        
        // Match the filtered nodes with vector search results
        WITH node, vector_node, score
        WHERE id(node) = id(vector_node)
        
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query, **params)
            return [self._process_result(record) for record in result]
    
    def retrieve_learning_path(
        self,
        concept: str,
        difficulty_level: Optional[str] = None,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a learning path for a concept based on prerequisites.
        
        Args:
            concept: The concept to learn
            difficulty_level: Optional difficulty level filter
            max_items: Maximum number of items in the path
            
        Returns:
            List of learning items in recommended sequence
        """
        difficulty_clause = ""
        params = {"concept": concept, "max_items": max_items}
        
        if difficulty_level:
            difficulty_clause = "AND node.difficulty = $difficulty"
            params["difficulty"] = difficulty_level
            
        # Check if the concept exists
        check_query = """
        MATCH (n {name: $concept})
        RETURN count(n) as count
        """
        
        with self.driver.session(database=self.database) as session:
            # Check if the concept exists
            check_result = session.run(check_query, concept=concept)
            record = check_result.single()
            if not record or record["count"] == 0:
                # Concept not found, return empty list
                return []
            
            # Concept exists, try to find learning path
            try:
                # Cypher query to find a learning path based on prerequisites
                cypher_query = f"""
                MATCH (target {{name: $concept}})
                OPTIONAL MATCH path = (start)-[:PREREQUISITE_FOR*]->(target)
                WHERE NOT (start)<-[:PREREQUISITE_FOR]-()
                WITH path, target
                OPTIONAL MATCH learning_path = (start)-[:PREREQUISITE_FOR*]->(target)
                UNWIND nodes(learning_path) as node
                WHERE node.type IN ['Example', 'Problem', 'Solution', 'Section'] {difficulty_clause}
                WITH node, apoc.coll.indexOf(nodes(learning_path), node) as position
                ORDER BY position
                RETURN node, position
                LIMIT $max_items
                """
                
                result = session.run(cypher_query, **params)
                return [
                    {**self._node_to_dict(record["node"]), "position": record["position"]}
                    for record in result
                ]
            except Exception as e:
                # Log the error and return empty list
                print(f"Error retrieving learning path: {e}")
                return []
    
    def _node_to_dict(self, node) -> Dict[str, Any]:
        """Convert a Neo4j node to a dictionary."""
        # Direct implementation without dependency on graph_retriever
        result = dict(node.items())
        result["__id__"] = node.id
        result["__labels__"] = list(node.labels)
        return result
    
    def _process_result(self, record) -> Dict[str, Any]:
        """Process a Neo4j record into a dictionary result."""
        node = record["node"]
        result = self._node_to_dict(node)
        
        # Add score if available
        if "score" in record:
            result["score"] = record["score"]
            
        # Add related nodes if available
        if "related_nodes" in record:
            result["related"] = [
                self._node_to_dict(related) for related in record["related_nodes"]
            ]
            
        return result 