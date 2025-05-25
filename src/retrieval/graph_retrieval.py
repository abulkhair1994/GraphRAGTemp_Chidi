"""
Graph Retrieval Engine for GraphRAG.

This module implements knowledge graph-based retrieval using Neo4j.
It leverages graph traversal to find connected information and provide
context based on entity relationships.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import re

from neo4j import GraphDatabase, Driver, Session

logger = logging.getLogger(__name__)

@dataclass
class GraphResult:
    """A structured result from graph retrieval."""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    paths: List[Dict[str, Any]]
    context: Optional[str] = None
    
    def to_text(self) -> str:
        """Convert the graph result to a textual representation."""
        text_parts = []
        
        if self.entities:
            text_parts.append("Entities:")
            for entity in self.entities:
                props = ', '.join(f"{k}: {v}" for k, v in entity.get('properties', {}).items())
                text_parts.append(f"  - {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')}) {props}")
        
        if self.relationships:
            text_parts.append("\nRelationships:")
            for rel in self.relationships:
                props = ', '.join(f"{k}: {v}" for k, v in rel.get('properties', {}).items())
                text_parts.append(
                    f"  - {rel.get('source', 'Unknown')} --[{rel.get('type', 'Unknown')}]--> "
                    f"{rel.get('target', 'Unknown')} {props}"
                )
                
        if self.context:
            text_parts.append(f"\nContext: {self.context}")
            
        return "\n".join(text_parts)


class GraphRetrievalEngine:
    """Graph-based retrieval engine using Neo4j."""
    
    def __init__(
        self,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
    ):
        """Initialize the graph retrieval engine.
        
        Args:
            neo4j_url: URL of the Neo4j database
            neo4j_username: Username for Neo4j authentication
            neo4j_password: Password for Neo4j authentication
            neo4j_database: Name of the Neo4j database to use
        """
        self.neo4j_url = neo4j_url
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        self.driver = None
        self.db_schema = None
        
    def connect(self) -> Driver:
        """Connect to Neo4j database and return the driver."""
        if not self.driver:
            self.driver = GraphDatabase.driver(
                self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password)
            )
            # Test the connection
            with self.driver.session(database=self.neo4j_database) as session:
                session.run("RETURN 1")
                logger.info("Successfully connected to Neo4j")
                
        return self.driver
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def _get_session(self) -> Session:
        """Get a new Neo4j session."""
        driver = self.connect()
        return driver.session(database=self.neo4j_database)
    
    def get_db_schema(self, force_refresh=False) -> Dict[str, Any]:
        """Get database schema information.
        
        This includes node labels, relationship types, and node properties.
        
        Args:
            force_refresh: Whether to force a refresh of the cached schema
            
        Returns:
            Dict with database schema information
        """
        if self.db_schema is not None and not force_refresh:
            return self.db_schema
            
        schema = {
            "labels": [],
            "relationship_types": [],
            "properties": {},
            "label_counts": {},
            "relationship_counts": {}
        }
        
        with self._get_session() as session:
            # Get node labels
            result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
            labels = [record["label"] for record in result]
            schema["labels"] = labels
            
            # Get counts for each label
            for label in labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result.single()["count"] if result.peek() else 0
                schema["label_counts"][label] = count
                
                # Get properties for this label
                result = session.run(f"""
                    MATCH (n:{label})
                    WITH n LIMIT 1
                    RETURN keys(n) as props
                """)
                props = result.single()["props"] if result.peek() else []
                schema["properties"][label] = props
            
            # Get relationship types
            result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
            rel_types = [record["relationshipType"] for record in result]
            schema["relationship_types"] = rel_types
            
            # Get counts for each relationship type
            for rel_type in rel_types:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                count = result.single()["count"] if result.peek() else 0
                schema["relationship_counts"][rel_type] = count
                
                # Get a sample relationship
                result = session.run(f"""
                    MATCH (a)-[r:{rel_type}]->(b)
                    WITH labels(a)[0] as a_label, labels(b)[0] as b_label, r
                    LIMIT 1
                    RETURN a_label, b_label, keys(r) as props
                """)
                if result.peek():
                    record = result.single()
                    schema["relationship_patterns"] = schema.get("relationship_patterns", [])
                    schema["relationship_patterns"].append({
                        "type": rel_type,
                        "source_label": record["a_label"],
                        "target_label": record["b_label"],
                        "properties": record["props"]
                    })
                    
        self.db_schema = schema
        return schema
    
    def entity_search(
        self, 
        entity_name: str, 
        entity_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for entities by name.
        
        Args:
            entity_name: Name or partial name of the entity
            entity_types: Optional list of entity types to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of entity objects
        """
        with self._get_session() as session:
            query_parts = ["MATCH (e)"]
            params = {"name": f"(?i).*{entity_name}.*"}
            
            # Add filters
            where_clauses = []
            
            # First check if entity has a name property
            result = session.run("""
                MATCH (n) 
                WHERE n.name IS NOT NULL 
                RETURN count(n) > 0 as has_name_property
            """)
            has_name_property = result.single()["has_name_property"] if result.peek() else False
            
            if has_name_property:
                where_clauses.append("e.name =~ $name")
            else:
                # Try to find a property that might contain the entity name
                schema = self.get_db_schema()
                text_props = set()
                
                for label, props in schema["properties"].items():
                    for prop in props:
                        if any(text_term in prop.lower() for text_term in [
                            "name", "title", "label", "text", "content", "description"
                        ]):
                            text_props.add(prop)
                
                if text_props:
                    text_props_str = " OR ".join([f"e.{prop} =~ $name" for prop in text_props])
                    where_clauses.append(f"({text_props_str})")
                else:
                    # Fallback to checking any string property
                    where_clauses.append("ANY(prop IN keys(e) WHERE e[prop] =~ $name AND apoc.meta.type(e[prop]) = 'STRING')")
            
            if entity_types:
                type_clauses = []
                for entity_type in entity_types:
                    type_clauses.append(f"e:{entity_type}")
                where_clauses.append("(" + " OR ".join(type_clauses) + ")")
                
            if where_clauses:
                query_parts.append("WHERE " + " AND ".join(where_clauses))
                
            query_parts.extend([
                "RETURN e.name AS name, labels(e) AS types, properties(e) AS properties",
                f"LIMIT {limit}"
            ])
            
            query = "\n".join(query_parts)
            
            try:
                result = session.run(query, params)
                return [dict(record) for record in result]
            except Exception as e:
                logger.error(f"Error in entity search: {e}")
                
                # Try a simpler fallback query if previous query failed
                try:
                    fallback_query = f"""
                        MATCH (e)
                        WITH e, labels(e) as types, properties(e) as props
                        RETURN e.name AS name, types, props as properties
                        LIMIT {limit}
                    """
                    result = session.run(fallback_query)
                    return [dict(record) for record in result]
                except Exception as fallback_e:
                    logger.error(f"Fallback entity search also failed: {fallback_e}")
                    return []
            
    def relationship_search(
        self, 
        source_entity: str,
        relationship_types: Optional[List[str]] = None,
        target_entity: Optional[str] = None,
        depth: int = 1,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for relationships between entities.
        
        Args:
            source_entity: Name of the source entity
            relationship_types: Optional list of relationship types to filter by
            target_entity: Optional name of the target entity
            depth: Maximum traversal depth for relationships
            limit: Maximum number of results to return
            
        Returns:
            List of relationship objects
        """
        with self._get_session() as session:
            # First check if entities have name properties
            result = session.run("""
                MATCH (n) 
                WHERE n.name IS NOT NULL 
                RETURN count(n) > 0 as has_name_property
            """)
            has_name_property = result.single()["has_name_property"] if result.peek() else False
            
            if not has_name_property:
                # If nodes don't have name properties, try to find a sample node
                logger.warning("Nodes don't have name properties, attempting to find relationship patterns")
                
                # Use DB schema to get relationship patterns
                schema = self.get_db_schema()
                
                if not schema["relationship_types"]:
                    logger.warning("No relationship types found in database")
                    return []
                
                # Find all relationship types
                rel_type = schema["relationship_types"][0]
                
                try:
                    # Just get a sample of relationships
                    query = f"""
                        MATCH (source)-[r:{rel_type}]->(target)
                        RETURN 
                            labels(source)[0] as source_type,
                            labels(target)[0] as target_type,
                            type(r) as type,
                            properties(source) as source_props,
                            properties(target) as target_props,
                            properties(r) as properties
                        LIMIT {limit}
                    """
                    
                    result = session.run(query)
                    relationships = []
                    
                    for record in result:
                        source_props = record["source_props"]
                        target_props = record["target_props"]
                        
                        # Try to find a property that could serve as a name
                        source_name = self._extract_name_from_props(source_props)
                        target_name = self._extract_name_from_props(target_props)
                        
                        relationships.append({
                            "source": source_name,
                            "source_type": record["source_type"],
                            "target": target_name,
                            "target_type": record["target_type"],
                            "type": record["type"],
                            "properties": record["properties"]
                        })
                    
                    return relationships
                except Exception as e:
                    logger.error(f"Error in fallback relationship search: {e}")
                    return []
            
            # Regular relationship search using names
            query_parts = ["MATCH (source)"]
            params = {"source_name": source_entity}
            
            where_clauses = ["source.name = $source_name"]
            
            # Build the relationship pattern
            rel_pattern = "-[r"
            if relationship_types:
                rel_pattern += ":" + "|:".join(relationship_types)
            rel_pattern += f"*1..{depth}]->"
            
            # Build the target pattern
            target_pattern = "(target)"
            if target_entity:
                where_clauses.append("target.name = $target_name")
                params["target_name"] = target_entity
                
            # Complete the query
            query_parts.append(f"MATCH (source){rel_pattern}{target_pattern}")
            
            if where_clauses:
                query_parts.append("WHERE " + " AND ".join(where_clauses))
                
            query_parts.extend([
                "RETURN source.name AS source, type(r) AS type, target.name AS target,",
                "properties(r) AS properties",
                f"LIMIT {limit}"
            ])
            
            query = "\n".join(query_parts)
            
            try:
                result = session.run(query, params)
                return [dict(record) for record in result]
            except Exception as e:
                logger.error(f"Error in relationship search: {e}")
                return []
    
    def _extract_name_from_props(self, properties: Dict[str, Any]) -> str:
        """Extract a name from entity properties."""
        # Try common name properties
        for name_prop in ["name", "title", "label", "id"]:
            if name_prop in properties:
                return str(properties[name_prop])
        
        # If no common name property, use the first string property
        for key, value in properties.items():
            if isinstance(value, str) and len(value) < 50:  # Reasonable length for a name
                return value
        
        # Last resort - return a random property
        if properties:
            first_key = next(iter(properties))
            return f"{first_key}:{properties[first_key]}"
            
        return "Unknown"
        
    def path_search(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Search for paths between two entities.
        
        Args:
            start_entity: Name of the starting entity
            end_entity: Name of the ending entity
            max_depth: Maximum path length
            
        Returns:
            List of path objects with nodes and relationships
        """
        with self._get_session() as session:
            query = """
            MATCH path = (start)-[*1..%d]->(end)
            WHERE start.name = $start_name AND end.name = $end_name
            RETURN path
            LIMIT 5
            """ % max_depth
            
            params = {
                "start_name": start_entity,
                "end_name": end_entity
            }
            
            try:
                result = session.run(query, params)
                paths = []
                
                for record in result:
                    path = record["path"]
                    nodes = []
                    relationships = []
                    
                    # Extract nodes
                    for node in path.nodes:
                        node_data = dict(node)
                        node_data["labels"] = list(node.labels)
                        nodes.append(node_data)
                    
                    # Extract relationships
                    for rel in path.relationships:
                        relationships.append({
                            "type": rel.type,
                            "properties": dict(rel),
                            "start_node": nodes.index(dict(rel.start_node)),
                            "end_node": nodes.index(dict(rel.end_node))
                        })
                    
                    paths.append({
                        "nodes": nodes,
                        "relationships": relationships
                    })
                
                return paths
            except Exception as e:
                logger.error(f"Error in path search: {e}")
                return []
    
    def community_search(
        self, 
        seed_entity: str,
        max_nodes: int = 20,
        max_depth: int = 2
    ) -> GraphResult:
        """Find the community of nodes around a seed entity.
        
        Args:
            seed_entity: Center entity to explore from
            max_nodes: Maximum number of nodes to include
            max_depth: Maximum traversal depth
            
        Returns:
            GraphResult with entities and relationships
        """
        with self._get_session() as session:
            # Check if nodes have a name property
            result = session.run("""
                MATCH (n) 
                WHERE n.name IS NOT NULL 
                RETURN count(n) > 0 as has_name_property
            """)
            has_name_property = result.single()["has_name_property"] if result.peek() else False
            
            if not has_name_property:
                # If no name property exists, return empty result
                logger.warning("Nodes don't have name properties, can't perform community search by name")
                return GraphResult(entities=[], relationships=[], paths=[])
            
            try:
                query = """
                MATCH (seed:node {name: $seed_name})
                CALL apoc.path.subgraphAll(seed, {
                    maxLevel: $max_depth,
                    limit: $max_nodes
                })
                YIELD nodes, relationships
                WITH nodes, relationships
                RETURN 
                    [n IN nodes | {
                        name: n.name, 
                        type: labels(n)[0],
                        properties: properties(n)
                    }] AS entities,
                    [r IN relationships | {
                        source: startNode(r).name, 
                        type: type(r), 
                        target: endNode(r).name,
                        properties: properties(r)
                    }] AS relationships
                """
                
                params = {
                    "seed_name": seed_entity,
                    "max_depth": max_depth,
                    "max_nodes": max_nodes
                }
                
                result = session.run(query, params)
                
                if not result.peek():
                    logger.warning(f"No community found for seed entity {seed_entity}")
                    
                    # Fallback query - find any nodes connected to the seed entity
                    fallback_query = """
                    MATCH (seed {name: $seed_name})-[r]-(other)
                    RETURN 
                        [{name: seed.name, type: labels(seed)[0], properties: properties(seed)}] + 
                        collect({name: other.name, type: labels(other)[0], properties: properties(other)}) AS entities,
                        collect({
                            source: seed.name, 
                            type: type(r), 
                            target: other.name,
                            properties: properties(r)
                        }) AS relationships
                    LIMIT 1
                    """
                    
                    result = session.run(fallback_query, {"seed_name": seed_entity})
                    
                    if not result.peek():
                        return GraphResult(entities=[], relationships=[], paths=[])
                
                record = result.single()
                return GraphResult(
                    entities=record["entities"],
                    relationships=record["relationships"],
                    paths=[]
                )
            except Exception as e:
                logger.error(f"Error in community search: {e}, trying fallback query")
                
                # Simple fallback query
                try:
                    fallback_query = """
                    MATCH (n)-[r]->(m)
                    RETURN 
                        collect(DISTINCT {name: n.name, type: labels(n)[0], properties: properties(n)}) + 
                        collect(DISTINCT {name: m.name, type: labels(m)[0], properties: properties(m)}) AS entities,
                        collect({
                            source: n.name, 
                            type: type(r), 
                            target: m.name,
                            properties: properties(r)
                        }) AS relationships
                    LIMIT 1
                    """
                    
                    result = session.run(fallback_query)
                    
                    if not result.peek():
                        return GraphResult(entities=[], relationships=[], paths=[])
                    
                    record = result.single()
                    return GraphResult(
                        entities=record["entities"],
                        relationships=record["relationships"],
                        paths=[]
                    )
                except Exception as fallback_e:
                    logger.error(f"Fallback query also failed: {fallback_e}")
                    return GraphResult(entities=[], relationships=[], paths=[])
    
    def knowledge_graph_query(
        self, 
        query_text: str,
        entity_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> GraphResult:
        """Perform a complex knowledge graph query based on the actual database structure.
        
        This method adapts to the database schema and tries various approaches to find
        relevant information.
        
        Args:
            query_text: Natural language query
            entity_types: Types of entities to include
            relationship_types: Types of relationships to include
            limit: Maximum number of results
            
        Returns:
            GraphResult object with entities and relationships
        """
        # Get database schema
        schema = self.get_db_schema()
        
        if not schema["labels"]:
            logger.warning("No node labels found in the database")
            return GraphResult(entities=[], relationships=[], paths=[])
        
        # Extract potential entities from the query
        extracted_entities = self._extract_query_entities(query_text)
        
        # Find entities in the graph
        all_entities = []
        all_relationships = []
        
        # If entity types were not specified, use the most populated ones from the schema
        if not entity_types:
            # Get top 3 most populated entity types
            sorted_labels = sorted(schema["label_counts"].items(), key=lambda x: x[1], reverse=True)
            entity_types = [label for label, count in sorted_labels[:3]]
            logger.info(f"Using top entity types from schema: {entity_types}")
            
        # If relationship types were not specified, use the ones from the schema
        if not relationship_types and schema["relationship_types"]:
            # Get top 3 most used relationship types
            sorted_rel_types = sorted(schema["relationship_counts"].items(), key=lambda x: x[1], reverse=True)
            relationship_types = [rel_type for rel_type, count in sorted_rel_types[:3]]
            logger.info(f"Using top relationship types from schema: {relationship_types}")
        
        # Try to find entities matching extracted terms
        for entity_name in extracted_entities:
            # Try entity search with the name
            entities = self.entity_search(
                entity_name=entity_name,
                entity_types=entity_types,
                limit=5
            )
            
            if entities:
                all_entities.extend(entities)
                
                # For each found entity, explore relationships
                for entity in entities[:2]:  # Limit to top 2 matching entities
                    # Try to find related entities through relationships
                    for rel_type in relationship_types if relationship_types else ["RELATED_TO"]:
                        try:
                            rels = self.relationship_search(
                                source_entity=entity["name"],
                                relationship_types=[rel_type],
                                depth=1,
                                limit=5
                            )
                            all_relationships.extend(rels)
                        except Exception as e:
                            logger.error(f"Error finding relationships for {entity['name']}: {e}")
        
        # If we didn't find specific entities, try a more general approach
        if not all_entities:
            logger.info("No specific entities found, trying general graph exploration")
            
            # Try to get a sample of entities from top entity types
            with self._get_session() as session:
                for label in entity_types[:2] if entity_types else schema["labels"][:2]:
                    try:
                        query = f"""
                            MATCH (n:{label})
                            WITH n, properties(n) as props
                            RETURN {{ 
                                name: CASE 
                                    WHEN n.name IS NOT NULL THEN n.name 
                                    ELSE '{label}_' + id(n) 
                                END, 
                                type: '{label}',
                                properties: props
                            }} as entity
                            LIMIT {limit // 2}
                        """
                        
                        result = session.run(query)
                        entities = [record["entity"] for record in result]
                        all_entities.extend(entities)
                        
                        # Get relationships for the first few entities
                        for entity in entities[:2]:
                            entity_name = entity["name"]
                            
                            rel_query = f"""
                                MATCH (n:{label})-[r]->(m)
                                WHERE n.name = $name OR id(n) = $id
                                WITH n, r, m, properties(r) as props
                                RETURN {{
                                    source: CASE 
                                        WHEN n.name IS NOT NULL THEN n.name 
                                        ELSE '{label}_' + id(n) 
                                    END,
                                    type: type(r),
                                    target: CASE 
                                        WHEN m.name IS NOT NULL THEN m.name 
                                        ELSE labels(m)[0] + '_' + id(m) 
                                    END,
                                    properties: props
                                }} as rel
                                LIMIT {limit // 2}
                            """
                            
                            # Try with name, fallback to direct entity access
                            try:
                                rel_result = session.run(rel_query, {"name": entity_name, "id": -1})
                                rels = [record["rel"] for record in rel_result]
                                all_relationships.extend(rels)
                            except Exception as e:
                                logger.error(f"Error finding relationships for {entity_name}: {e}")
                    except Exception as e:
                        logger.error(f"Error getting entities for label {label}: {e}")
        
        # Remove duplicates
        unique_entities = []
        seen_entities = set()
        for entity in all_entities:
            entity_name = entity.get("name")
            if entity_name and entity_name not in seen_entities:
                seen_entities.add(entity_name)
                unique_entities.append(entity)
                
        unique_relationships = []
        seen_relationships = set()
        for rel in all_relationships:
            rel_key = f"{rel.get('source', '')}-{rel.get('type', '')}-{rel.get('target', '')}"
            if rel_key not in seen_relationships and rel.get('source') and rel.get('target'):
                seen_relationships.add(rel_key)
                unique_relationships.append(rel)
        
        # If we still couldn't find anything, try a direct graph query
        if not unique_entities and not unique_relationships:
            logger.warning("No entities or relationships found with standard methods, trying direct graph query")
            
            with self._get_session() as session:
                try:
                    # Just get a sample of the graph
                    query = """
                        MATCH (n)-[r]->(m)
                        WITH n, r, m, labels(n)[0] as n_label, labels(m)[0] as m_label
                        RETURN 
                            COLLECT(DISTINCT {
                                name: CASE WHEN n.name IS NOT NULL THEN n.name ELSE n_label + '_' + id(n) END,
                                type: n_label,
                                properties: properties(n)
                            })[..10] + 
                            COLLECT(DISTINCT {
                                name: CASE WHEN m.name IS NOT NULL THEN m.name ELSE m_label + '_' + id(m) END,
                                type: m_label,
                                properties: properties(m)
                            })[..10] AS entities,
                            COLLECT({
                                source: CASE WHEN n.name IS NOT NULL THEN n.name ELSE n_label + '_' + id(n) END,
                                type: type(r),
                                target: CASE WHEN m.name IS NOT NULL THEN m.name ELSE m_label + '_' + id(m) END,
                                properties: properties(r)
                            })[..20] AS relationships
                    """
                    
                    result = session.run(query)
                    
                    if result.peek():
                        record = result.single()
                        unique_entities = record["entities"]
                        unique_relationships = record["relationships"]
                except Exception as e:
                    logger.error(f"Direct graph query failed: {e}")
        
        return GraphResult(
            entities=unique_entities[:limit],
            relationships=unique_relationships[:limit],
            paths=[]
        )
    
    def _extract_query_entities(self, query_text: str) -> List[str]:
        """Extract potential entity names from a query.
        
        Args:
            query_text: Natural language query
            
        Returns:
            List of potential entity names
        """
        entities = []
        
        # Check for schema information
        schema = self.get_db_schema()
        
        # Extract capitalized words as potential entities
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', query_text)
        entities.extend(capitalized)
        
        # Extract quoted phrases
        quotes = re.findall(r'"([^"]+)"', query_text)
        entities.extend(quotes)
        
        # Extract domain-specific terms based on database schema
        if schema and schema["labels"]:
            for label in schema["labels"]:
                # Convert label to lowercase 
                label_lower = label.lower()
                if label_lower in query_text.lower():
                    entities.append(label)
        
        # If we didn't find any specific entities, extract key words from the query
        if not entities:
            # Remove stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "about", "is", "are"}
            words = [w for w in query_text.lower().split() if w not in stop_words and len(w) > 3]
            entities.extend(words)
            
        return list(set(entities))
    
    def retrieve_hierarchical_context(
        self,
        entity_id: str,
        hierarchy_relationships: List[str] = None,
        include_siblings: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve the hierarchical context of an entity (e.g., example → section → chapter).
        
        Args:
            entity_id: The ID of the entity to get context for
            hierarchy_relationships: List of relationships defining hierarchy
            include_siblings: Whether to include sibling nodes
            
        Returns:
            Dictionary with hierarchical context
        """
        if hierarchy_relationships is None:
            hierarchy_relationships = ["PART_OF", "IN_SECTION", "IN_CHAPTER", "CONTAINS"]
            
        # Create relationship patterns with proper syntax
        hierarchy_pattern = "|".join([f"{rel}" for rel in hierarchy_relationships])
        
        # Cypher query to retrieve hierarchy
        cypher_query = f"""
        MATCH (entity {{id: $entity_id}})
        OPTIONAL MATCH path = (entity)-[r:{hierarchy_pattern}*]->(parent)
        WITH entity, collect(path) as paths
        OPTIONAL MATCH (sibling)-[:{hierarchy_pattern}]->(parent)
        WHERE sibling <> entity AND $include_siblings
        WITH entity, paths, collect(sibling) as siblings
        RETURN entity, paths, siblings
        """
        
        with self.driver.session(database=self.neo4j_database) as session:
            try:
                result = session.run(
                    cypher_query,
                    entity_id=entity_id,
                    include_siblings=include_siblings
                )
                record = result.single()
                if record:
                    return {
                        "entity": self._node_to_dict(record["entity"]),
                        "hierarchy": self._paths_to_hierarchy(record["paths"]),
                        "siblings": [self._node_to_dict(sibling) for sibling in record["siblings"]]
                    }
                return {}
            except Exception as e:
                # Log the error and return empty context
                print(f"Error retrieving hierarchical context: {e}")
                return {}
    
    def _node_to_dict(self, node) -> Dict[str, Any]:
        """Convert a Neo4j node to a dictionary."""
        result = dict(node.items())
        result["__id__"] = node.id
        result["__labels__"] = list(node.labels)
        return result
    
    def _paths_to_hierarchy(self, paths) -> Dict[str, Any]:
        """Convert Neo4j paths to a hierarchical structure."""
        hierarchy = {}
        for path in paths:
            current = hierarchy
            for node in path.nodes:
                node_dict = self._node_to_dict(node)
                node_id = node_dict["__id__"]
                if node_id not in current:
                    current[node_id] = {"data": node_dict, "children": {}}
                current = current[node_id]["children"] 

    def test_connection(self) -> bool:
        """Test the connection to the Neo4j database.
        
        Returns:
            True if connection is successful, otherwise raises an exception
        """
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            self.neo4j_url,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        try:
            with driver.session(database=self.neo4j_database) as session:
                # Run a simple query to check connectivity
                result = session.run("RETURN 1 as success")
                record = result.single()
                return record and record["success"] == 1
        finally:
            driver.close()

    def find_content_by_chapter_section(
        self,
        chapter: Optional[int] = None,
        section: Optional[int] = None,
        content_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find content by chapter and section numbers.
        
        Args:
            chapter: Chapter number to filter by
            section: Section number to filter by 
            content_types: List of content types to include
            limit: Maximum number of results to return
            
        Returns:
            List of content items matching the criteria
        """
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                # Build query conditions
                conditions = []
                params = {}
                
                if chapter is not None:
                    conditions.append("n.chapter = $chapter")
                    params["chapter"] = chapter
                    
                if section is not None:
                    conditions.append("n.section = $section")
                    params["section"] = section
                    
                if content_types:
                    conditions.append("any(label IN labels(n) WHERE label IN $content_types)")
                    params["content_types"] = content_types
                
                # Add text content check
                conditions.append(f"n.{self.text_node_property} IS NOT NULL")
                conditions.append(f"n.{self.text_node_property} <> ''")
                
                # Combine conditions
                where_clause = " AND ".join(conditions) if conditions else "TRUE"
                
                # Build and execute query
                query = f"""
                MATCH (n)
                WHERE {where_clause}
                RETURN n, labels(n) as labels
                ORDER BY n.chapter, n.section
                LIMIT {limit}
                """
                
                result = session.run(query, params)
                
                items = []
                for record in result:
                    node = record["n"]
                    labels = record["labels"]
                    
                    # Extract text and metadata
                    text = node.get(self.text_node_property, "")
                    metadata = {
                        "node_id": node.id,
                        "labels": labels,
                        "chapter": node.get("chapter"),
                        "section": node.get("section"),
                        **{k: v for k, v in node.items() 
                           if k not in [self.text_node_property, "chapter", "section"]}
                    }
                    
                    items.append({
                        "content": text,
                        "metadata": metadata
                    })
                    
                return items
                
        except Exception as e:
            logger.error(f"Error in find_content_by_chapter_section: {e}")
            return [] 