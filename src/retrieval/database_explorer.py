"""
Neo4j Database Explorer for GraphRAG.

This module provides functionality to explore the contents of a Neo4j database
for retrieval purposes, showing actual data without creating any fictional content.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from neo4j import GraphDatabase
from tabulate import tabulate
from src.utils.env_manager import load_env_vars, EnvManager

class DatabaseExplorer:
    """Explore Neo4j database contents for GraphRAG development."""
    
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ):
        """Initialize the Database Explorer.
        
        Args:
            url: Neo4j connection URL
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        self.url = url
        self.username = username
        self.password = password
        self.database = database
        
    def connect(self):
        """Connect to Neo4j database."""
        return GraphDatabase.driver(
            self.url, 
            auth=(self.username, self.password)
        )
    
    def close(self):
        """Close the connection."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def get_schema_overview(self) -> Dict[str, List[str]]:
        """Get database schema information.
        
        Returns:
            Dictionary containing labels, relationship types, and properties
        """
        driver = self.connect()
        schema = {
            "labels": [],
            "relationship_types": [],
            "properties": []
        }
        
        try:
            with driver.session(database=self.database) as session:
                # Get labels (node types)
                labels_result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
                schema["labels"] = [record["label"] for record in labels_result]
                
                # Get relationship types
                rel_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
                schema["relationship_types"] = [record["relationshipType"] for record in rel_result]
                
                # Get properties
                prop_result = session.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey")
                schema["properties"] = [record["propertyKey"] for record in prop_result]
        finally:
            driver.close()
            
        return schema
    
    def get_vector_indexes(self) -> List[Dict[str, Any]]:
        """Get information about vector indexes in the database.
        
        Returns:
            List of vector index information
        """
        driver = self.connect()
        vector_indexes = []
        
        try:
            with driver.session(database=self.database) as session:
                try:
                    vector_index_result = session.run("""
                        SHOW INDEXES 
                        YIELD type, labelsOrTypes, properties, name 
                        WHERE type = 'VECTOR' 
                        RETURN name, labelsOrTypes, properties
                    """)
                    
                    for record in vector_index_result:
                        vector_indexes.append({
                            "name": record["name"],
                            "labels": record["labelsOrTypes"],
                            "properties": record["properties"]
                        })
                except Exception as e:
                    # If VECTOR isn't recognized as a valid index type, it might be an older version
                    print(f"Error querying vector indexes: {e}")
        finally:
            driver.close()
            
        return vector_indexes
    
    def get_sample_nodes(self, label: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get sample nodes for a given label.
        
        Args:
            label: Node label to query
            limit: Maximum number of nodes to return
            
        Returns:
            List of node data
        """
        driver = self.connect()
        nodes = []
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(f"""
                    MATCH (n:{label}) 
                    RETURN n LIMIT $limit
                """, limit=limit)
                
                for record in result:
                    node = record["n"]
                    node_data = dict(node)
                    
                    # Handle embeddings and large values for display
                    for key, value in node_data.items():
                        if isinstance(value, list) and len(value) > 10:
                            # Likely an embedding - truncate it
                            node_data[key] = f"List with {len(value)} elements"
                        elif isinstance(value, str) and len(value) > 100:
                            # Truncate long strings
                            node_data[key] = value[:100] + "..."
                    
                    nodes.append(node_data)
        finally:
            driver.close()
            
        return nodes
    
    def get_sample_relationships(self, rel_type: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get sample relationships for a given type.
        
        Args:
            rel_type: Relationship type to query
            limit: Maximum number of relationships to return
            
        Returns:
            List of relationship data
        """
        driver = self.connect()
        relationships = []
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(f"""
                    MATCH (a)-[r:{rel_type}]->(b)
                    RETURN type(r) as type, 
                           labels(a)[0] as from_label, 
                           a.name as from_name,
                           labels(b)[0] as to_label,
                           b.name as to_name,
                           properties(r) as properties
                    LIMIT $limit
                """, limit=limit)
                
                for record in result:
                    relationships.append({
                        "type": record["type"],
                        "from_label": record["from_label"],
                        "from_name": record["from_name"],
                        "to_label": record["to_label"],
                        "to_name": record["to_name"],
                        "properties": record["properties"]
                    })
        finally:
            driver.close()
            
        return relationships
    
    def analyze_text_properties(self) -> Dict[str, Any]:
        """Analyze text properties across different node types."""
        driver = self.connect()
        analysis = {}
        
        try:
            with driver.session(database=self.database) as session:
                # For each node label, check which text properties are used
                query = """
                MATCH (n)
                WITH labels(n) as labels, n
                UNWIND labels as label
                WITH label, n
                WITH label,
                     count(*) as total_nodes,
                     sum(CASE WHEN n.title IS NOT NULL THEN 1 ELSE 0 END) as has_title,
                     sum(CASE WHEN n.content IS NOT NULL THEN 1 ELSE 0 END) as has_content,
                     sum(CASE WHEN n.text_content IS NOT NULL THEN 1 ELSE 0 END) as has_text_content,
                     sum(CASE WHEN n.problem_text IS NOT NULL THEN 1 ELSE 0 END) as has_problem_text,
                     sum(CASE WHEN n.solution_text IS NOT NULL THEN 1 ELSE 0 END) as has_solution_text
                RETURN label, total_nodes, has_title, has_content, has_text_content, 
                       has_problem_text, has_solution_text
                ORDER BY total_nodes DESC
                """
                result = session.run(query)
                analysis["text_properties"] = [dict(record) for record in result]
        finally:
            driver.close()
            
        return analysis

    def analyze_relationships(self) -> Dict[str, Any]:
        """Analyze relationship patterns in the graph."""
        driver = self.connect()
        analysis = {}
        
        try:
            with driver.session(database=self.database) as session:
                # Get relationship patterns between different node types
                query = """
                MATCH (a)-[r]->(b)
                WITH labels(a)[0] as from_label,
                     type(r) as rel_type,
                     labels(b)[0] as to_label,
                     count(*) as count
                RETURN from_label, rel_type, to_label, count
                ORDER BY count DESC
                """
                result = session.run(query)
                analysis["relationship_patterns"] = [dict(record) for record in result]

                # Get average number of relationships per node type
                query = """
                MATCH (n)
                WITH labels(n)[0] as label,
                     count(*) as total_nodes
                OPTIONAL MATCH (n)-[r]-()
                WITH label, total_nodes, count(r) as total_relationships
                RETURN label,
                       total_nodes,
                       total_relationships,
                       toFloat(total_relationships)/total_nodes as avg_relationships
                ORDER BY avg_relationships DESC
                """
                result = session.run(query)
                analysis["node_connectivity"] = [dict(record) for record in result]
        finally:
            driver.close()
            
        return analysis

    def analyze_hierarchical_structure(self) -> Dict[str, Any]:
        """Analyze the hierarchical structure through CONTAINS relationships."""
        driver = self.connect()
        analysis = {}
        
        try:
            with driver.session(database=self.database) as session:
                # Get hierarchy levels and their node counts
                query = """
                MATCH path = (root)-[:CONTAINS*]->(leaf)
                WHERE NOT ()-[:CONTAINS]->(root)
                WITH root, length(path) as depth, leaf
                RETURN labels(root)[0] as root_type,
                       labels(leaf)[0] as leaf_type,
                       depth,
                       count(*) as count
                ORDER BY depth, count DESC
                """
                result = session.run(query)
                analysis["hierarchy_levels"] = [dict(record) for record in result]

                # Get parent-child relationships
                query = """
                MATCH (parent)-[:CONTAINS]->(child)
                WITH labels(parent)[0] as parent_type,
                     labels(child)[0] as child_type,
                     count(*) as relationship_count
                RETURN parent_type, child_type, relationship_count
                ORDER BY relationship_count DESC
                """
                result = session.run(query)
                analysis["parent_child_patterns"] = [dict(record) for record in result]
        finally:
            driver.close()
            
        return analysis

    def analyze_educational_metadata(self) -> Dict[str, Any]:
        """Analyze educational metadata including courses, chapters, and learning units.
        
        This method extracts information specifically relevant to educational content
        such as course names, subjects, chapters, and their hierarchical organization.
        
        Returns:
            Dictionary containing educational content analysis
        """
        driver = self.connect()
        analysis = {}
        
        try:
            with driver.session(database=self.database) as session:
                # Check if Course nodes exist
                course_count_query = """
                MATCH (c:Course)
                RETURN count(c) as course_count
                """
                result = session.run(course_count_query)
                course_count = result.single()["course_count"] if result.peek() else 0
                analysis["course_count"] = course_count
                
                # Get educational content types and counts
                content_types_query = """
                MATCH (n) 
                WHERE ANY(label IN labels(n) WHERE label IN [
                    'Chapter', 'Section', 'Exercise', 'Problem', 'Solution', 
                    'Example', 'Practice', 'Content', 'Module'
                ])
                WITH labels(n)[0] as content_type, count(n) as count
                RETURN content_type, count
                ORDER BY count DESC
                """
                result = session.run(content_types_query)
                content_types = [{"type": record["content_type"], "count": record["count"]} for record in result]
                analysis["content_types"] = content_types
                
                # Analyze learning objectives
                learning_objectives_query = """
                MATCH (n)
                WHERE n.learning_objective IS NOT NULL
                WITH n.learning_objective as objective, count(n) as count, collect(labels(n)[0])[0] as node_type
                RETURN objective, count, node_type
                ORDER BY count DESC
                LIMIT 20
                """
                result = session.run(learning_objectives_query)
                learning_objectives = [
                    {
                        "objective": record["objective"],
                        "count": record["count"],
                        "node_type": record["node_type"]
                    }
                    for record in result
                ]
                analysis["learning_objectives"] = learning_objectives
                
                # Get topic distribution
                topics_query = """
                MATCH (n)
                WHERE n.topic IS NOT NULL
                WITH n.topic as topic, count(n) as count, collect(labels(n)[0])[0] as node_type
                RETURN topic, count, node_type
                ORDER BY count DESC
                LIMIT 20
                """
                result = session.run(topics_query)
                topics = [
                    {
                        "topic": record["topic"],
                        "count": record["count"],
                        "node_type": record["node_type"]
                    }
                    for record in result
                ]
                analysis["topics"] = topics
                
                # Get difficulty distribution
                difficulty_query = """
                MATCH (n)
                WHERE n.difficulty IS NOT NULL
                WITH n.difficulty as difficulty, count(n) as count, collect(labels(n)[0])[0] as node_type
                RETURN difficulty, count, node_type
                ORDER BY count DESC
                """
                result = session.run(difficulty_query)
                difficulties = [
                    {
                        "difficulty": record["difficulty"],
                        "count": record["count"],
                        "node_type": record["node_type"]
                    }
                    for record in result
                ]
                analysis["difficulties"] = difficulties
                
                if course_count > 0:
                    # Get course metadata properties
                    course_props_query = """
                    MATCH (c:Course)
                    WITH c LIMIT 1
                    RETURN keys(c) as course_properties
                    """
                    result = session.run(course_props_query)
                    course_properties = result.single()["course_properties"] if result.peek() else []
                    analysis["course_properties"] = course_properties
                    
                    # Get all courses with key metadata
                    courses_query = """
                    MATCH (c:Course)
                    RETURN 
                      c.name as name,
                      c.title as title,
                      c.subject as subject,
                      c.description as description,
                      id(c) as id
                    ORDER BY c.name
                    """
                    result = session.run(courses_query)
                    courses = [dict(record) for record in result]
                    analysis["courses"] = courses
                    
                    # Get course structure - chapters per course
                    course_structure_query = """
                    MATCH (c:Course)-[:CONTAINS]->(ch:Chapter)
                    WITH c.name as course_name, count(ch) as chapter_count
                    RETURN course_name, chapter_count
                    ORDER BY chapter_count DESC
                    """
                    result = session.run(course_structure_query)
                    course_structure = [dict(record) for record in result]
                    analysis["course_structure"] = course_structure
                    
                    # Get educational hierarchy
                    hierarchy_query = """
                    MATCH path = (c:Course)-[:CONTAINS*]->(n)
                    WITH c.name as course_name, labels(n)[0] as node_type, count(n) as count
                    RETURN course_name, node_type, count
                    ORDER BY course_name, count DESC
                    """
                    result = session.run(hierarchy_query)
                    hierarchy = [dict(record) for record in result]
                    analysis["educational_hierarchy"] = hierarchy
                    
                    # Check for metadata properties across educational content
                    metadata_query = """
                    MATCH (n)
                    WHERE any(label in labels(n) WHERE label IN ['Course', 'Chapter', 'Section', 'Exercise', 'Problem', 'Solution'])
                    WITH labels(n)[0] as node_type,
                         count(*) as node_count,
                         sum(CASE WHEN n.subject IS NOT NULL THEN 1 ELSE 0 END) as has_subject,
                         sum(CASE WHEN n.course_name IS NOT NULL THEN 1 ELSE 0 END) as has_course_name,
                         sum(CASE WHEN n.difficulty IS NOT NULL THEN 1 ELSE 0 END) as has_difficulty,
                         sum(CASE WHEN n.learning_objective IS NOT NULL THEN 1 ELSE 0 END) as has_learning_objective,
                         sum(CASE WHEN n.topic IS NOT NULL THEN 1 ELSE 0 END) as has_topic
                    RETURN node_type, node_count, has_subject, has_course_name, has_difficulty, has_learning_objective, has_topic
                    ORDER BY node_count DESC
                    """
                    result = session.run(metadata_query)
                    metadata_analysis = [dict(record) for record in result]
                    analysis["educational_metadata"] = metadata_analysis
                    
                    # Get course topics if they exist
                    topics_query = """
                    MATCH (c:Course)
                    WHERE c.topic IS NOT NULL OR c.topics IS NOT NULL OR c.subject IS NOT NULL
                    RETURN 
                      c.name as course_name,
                      c.topic as topic,
                      c.topics as topics,
                      c.subject as subject
                    """
                    result = session.run(topics_query)
                    topics = [dict(record) for record in result]
                    analysis["course_topics"] = topics
                else:
                    # Look for hierarchical structure even without Course nodes
                    hierarchy_query = """
                    MATCH (root)-[:CONTAINS*]->(leaf)
                    WHERE NOT ()-[:CONTAINS]->(root)
                    WITH root, labels(root)[0] as root_type, leaf, labels(leaf)[0] as leaf_type
                    RETURN DISTINCT root_type, leaf_type, 
                           count(*) as path_count,
                           collect(DISTINCT root.name)[0] as sample_root_name
                    ORDER BY path_count DESC
                    """
                    result = session.run(hierarchy_query)
                    hierarchy = []
                    for record in result:
                        hierarchy.append({
                            "root_type": record["root_type"],
                            "leaf_type": record["leaf_type"],
                            "path_count": record["path_count"],
                            "sample_root_name": record["sample_root_name"]
                        })
                    analysis["hierarchy"] = hierarchy
                    
                    # Analyze concept relationships
                    concept_query = """
                    MATCH (a)-[r:RELATED_TO|PREREQUISITE_FOR|REFERENCES]->(b)
                    WITH type(r) as relationship_type, count(r) as count,
                         labels(a)[0] as source_type, labels(b)[0] as target_type
                    RETURN relationship_type, count, source_type, target_type
                    ORDER BY count DESC
                    """
                    result = session.run(concept_query)
                    concept_relationships = []
                    for record in result:
                        concept_relationships.append({
                            "relationship_type": record["relationship_type"],
                            "count": record["count"],
                            "source_type": record["source_type"],
                            "target_type": record["target_type"]
                        })
                    analysis["concept_relationships"] = concept_relationships
                
                # Analyze educational content structure
                structure_query = """
                MATCH (n)
                WHERE ANY(label IN labels(n) WHERE label IN [
                    'Chapter', 'Section', 'Exercise', 'Problem', 'Solution', 
                    'Example', 'Practice', 'Content', 'Module'
                ])
                OPTIONAL MATCH (n)-[:CONTAINS]->(child)
                WITH n, labels(n)[0] as node_type, count(child) as children_count
                RETURN node_type, 
                       count(n) as node_count,
                       sum(children_count) as total_children,
                       avg(children_count) as avg_children_per_node
                ORDER BY node_count DESC
                """
                result = session.run(structure_query)
                structure = []
                for record in result:
                    structure.append({
                        "node_type": record["node_type"],
                        "node_count": record["node_count"],
                        "total_children": record["total_children"],
                        "avg_children_per_node": record["avg_children_per_node"]
                    })
                analysis["structure"] = structure
                
                # Analyze popular reference patterns
                reference_query = """
                MATCH (a)-[r:REFERENCES]->(b)
                WITH labels(a)[0] as source_type, labels(b)[0] as target_type, count(r) as ref_count
                RETURN source_type, target_type, ref_count
                ORDER BY ref_count DESC
                LIMIT 10
                """
                result = session.run(reference_query)
                references = []
                for record in result:
                    references.append({
                        "source_type": record["source_type"],
                        "target_type": record["target_type"],
                        "ref_count": record["ref_count"]
                    })
                analysis["references"] = references
                
        except Exception as e:
            print(f"Error analyzing educational metadata: {e}")
        finally:
            driver.close()
            
        return analysis

    def explore_database(self, print_output: bool = True) -> Dict[str, Any]:
        """Explore the complete database and return findings.
        
        Args:
            print_output: Whether to print formatted output to console
            
        Returns:
            Dictionary containing all exploration results
        """
        results = {}
        
        # Get schema overview
        schema = self.get_schema_overview()
        results["schema"] = schema
        
        if print_output:
            print("\n📋 DATABASE SCHEMA OVERVIEW:")
            print(f"Node Labels ({len(schema['labels'])}): {', '.join(schema['labels'])}")
            print(f"Relationship Types ({len(schema['relationship_types'])}): {', '.join(schema['relationship_types'])}")
            print(f"Properties ({len(schema['properties'])}): {', '.join(schema['properties'])}")
        
        # Get vector indexes
        vector_indexes = self.get_vector_indexes()
        results["vector_indexes"] = vector_indexes
        
        if print_output:
            print("\n🔢 VECTOR INDEXES:")
            if vector_indexes:
                for idx in vector_indexes:
                    print(f"  Name: {idx['name']}")
                    print(f"  Labels: {idx['labels']}")
                    print(f"  Properties: {idx['properties']}")
                    print("")
            else:
                print("  No vector indexes found")
        
        # Sample data for each label
        node_samples = {}
        for label in schema["labels"]:
            node_samples[label] = self.get_sample_nodes(label)
        
        results["node_samples"] = node_samples
        
        if print_output:
            print("\n📊 SAMPLE DATA BY NODE LABEL:")
            for label, nodes in node_samples.items():
                print(f"\n  {label} Nodes:")
                
                if not nodes:
                    print("    No data found")
                    continue
                
                # Create a table for better display
                table_data = []
                headers = ["Property", "Value"]
                
                for i, node in enumerate(nodes):
                    if i > 0:
                        table_data.append(["---", "---"])  # Separator between nodes
                        
                    for key, value in node.items():
                        # Format the value for display
                        if isinstance(value, list):
                            if len(value) > 10:  # Likely an embedding
                                value_display = f"[{value[0]:.6f}, {value[1]:.6f}, ... + {len(value)-2} more]"
                            else:
                                value_display = str(value)
                        elif isinstance(value, dict):
                            value_display = json.dumps(value, indent=2)
                        else:
                            value_display = str(value)
                            
                        table_data.append([key, value_display])
                
                print(tabulate(table_data, headers=headers, tablefmt="simple"))
        
        # Sample relationships
        relationship_samples = {}
        for rel_type in schema["relationship_types"]:
            relationship_samples[rel_type] = self.get_sample_relationships(rel_type)
        
        results["relationship_samples"] = relationship_samples
        
        if print_output:
            if schema["relationship_types"]:
                print("\n🔗 SAMPLE RELATIONSHIPS:")
                for rel_type, relationships in relationship_samples.items():
                    print(f"\n  {rel_type} Relationships:")
                    
                    if not relationships:
                        print("    No data found")
                        continue
                    
                    table_data = []
                    headers = ["From", "Relationship", "To", "Properties"]
                    
                    for rel in relationships:
                        from_node = f"{rel['from_label']}:{rel['from_name'] or 'unknown'}"
                        to_node = f"{rel['to_label']}:{rel['to_name'] or 'unknown'}"
                        props = json.dumps(rel["properties"]) if rel["properties"] else "{}"
                        
                        table_data.append([from_node, rel["type"], to_node, props])
                    
                    print(tabulate(table_data, headers=headers, tablefmt="simple"))
        
        # Analyze text properties
        text_analysis = self.analyze_text_properties()
        results["text_analysis"] = text_analysis
        
        if print_output:
            print("\n📝 TEXT PROPERTIES ANALYSIS:")
            headers = ["Label", "Total Nodes", "Title", "Content", "Text", "Problem", "Solution"]
            table_data = [
                [
                    record["label"],
                    record["total_nodes"],
                    f"{record['has_title']}/{record['total_nodes']}",
                    f"{record['has_content']}/{record['total_nodes']}",
                    f"{record['has_text_content']}/{record['total_nodes']}",
                    f"{record['has_problem_text']}/{record['total_nodes']}",
                    f"{record['has_solution_text']}/{record['total_nodes']}"
                ]
                for record in text_analysis["text_properties"]
            ]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))

        # Analyze relationships
        rel_analysis = self.analyze_relationships()
        results["relationship_analysis"] = rel_analysis
        
        if print_output:
            print("\n🔗 RELATIONSHIP PATTERNS:")
            headers = ["From", "Relationship", "To", "Count"]
            table_data = [
                [record["from_label"], record["rel_type"], record["to_label"], record["count"]]
                for record in rel_analysis["relationship_patterns"]
            ]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))
            
            print("\n📊 NODE CONNECTIVITY:")
            headers = ["Node Type", "Total Nodes", "Total Relationships", "Avg Relationships"]
            table_data = [
                [
                    record["label"],
                    record["total_nodes"],
                    record["total_relationships"],
                    f"{record['avg_relationships']:.2f}"
                ]
                for record in rel_analysis["node_connectivity"]
            ]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))

        # Analyze hierarchical structure
        hierarchy_analysis = self.analyze_hierarchical_structure()
        results["hierarchy_analysis"] = hierarchy_analysis
        
        if print_output:
            print("\n📚 HIERARCHICAL STRUCTURE:")
            headers = ["Root Type", "Leaf Type", "Depth", "Count"]
            table_data = [
                [record["root_type"], record["leaf_type"], record["depth"], record["count"]]
                for record in hierarchy_analysis["hierarchy_levels"]
            ]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))

            print("\n📈 PARENT-CHILD PATTERNS:")
            headers = ["Parent Type", "Child Type", "Count"]
            table_data = [
                [record["parent_type"], record["child_type"], record["relationship_count"]]
                for record in hierarchy_analysis["parent_child_patterns"]
            ]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))

        # Analyze educational metadata
        educational_metadata = self.analyze_educational_metadata()
        results["educational_metadata"] = educational_metadata
        
        if print_output:
            print("\n📚 EDUCATIONAL METADATA:")
            for key, value in educational_metadata.items():
                if isinstance(value, list):
                    print(f"\n{key}:")
                    for item in value:
                        print(f"  - {item}")
                else:
                    print(f"\n{key}: {value}")

        return results

    def check_embedding_properties(self, potential_embedding_props=None):
        """Check which embedding properties exist in the database and analyze them.
        
        Args:
            potential_embedding_props: List of property names to check as potential embeddings
                                      (default: common embedding property names)
                                      
        Returns:
            Dictionary with embedding property information per node label
        """
        if potential_embedding_props is None:
            potential_embedding_props = [
                "embedding", "embeddings", "fastRP_embedding", 
                "vector", "vectors", "text_embedding", "content_embedding"
            ]
            
        driver = self.connect()
        results = {}
        
        try:
            with driver.session(database=self.database) as session:
                # Get all node labels first
                labels_result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
                labels = [record["label"] for record in labels_result]
                
                for label in labels:
                    label_results = {}
                    
                    # Check each potential embedding property
                    for prop in potential_embedding_props:
                        try:
                            # Check if this property exists for this label
                            result = session.run(f"""
                                MATCH (n:{label})
                                WHERE n.{prop} IS NOT NULL
                                RETURN COUNT(n) as count
                            """)
                            
                            count = result.single()["count"] if result.peek() else 0
                            
                            if count > 0:
                                # This property exists, get a sample to check its dimensions
                                result = session.run(f"""
                                    MATCH (n:{label})
                                    WHERE n.{prop} IS NOT NULL
                                    WITH n LIMIT 1
                                    RETURN size(n.{prop}) as dimensions, 
                                           apoc.meta.type(n.{prop}) as type
                                """)
                                
                                if result.peek():
                                    record = result.single()
                                    dimensions = record.get("dimensions", "unknown")
                                    prop_type = record.get("type", "unknown")
                                else:
                                    dimensions = "unknown"
                                    prop_type = "unknown"
                                    
                                # Add to results
                                label_results[prop] = {
                                    "count": count,
                                    "dimensions": dimensions,
                                    "type": prop_type
                                }
                        except Exception as e:
                            # Skip errors from property queries
                            pass
                    
                    if label_results:
                        results[label] = label_results
                    
                # Also check if there's a vector index for any of these properties
                try:
                    vector_indexes = self.get_vector_indexes()
                    if vector_indexes:
                        results["vector_indexes"] = vector_indexes
                except Exception as e:
                    results["vector_indexes_error"] = str(e)
                    
        except Exception as e:
            results["error"] = str(e)
        finally:
            driver.close()
            
        return results

    def print_embedding_properties(self, results=None):
        """Print a formatted report of embedding properties in the database.
        
        Args:
            results: Results from check_embedding_properties (if None, will run the check)
        """
        if results is None:
            results = self.check_embedding_properties()
            
        print("\n📊 EMBEDDING PROPERTIES IN DATABASE:")
        
        if "error" in results:
            print(f"  Error checking embeddings: {results['error']}")
            return
            
        # Print vector indexes first if available
        if "vector_indexes" in results:
            print("\n  🔢 VECTOR INDEXES:")
            for idx in results["vector_indexes"]:
                print(f"    - Name: {idx['name']}")
                print(f"      Labels: {idx['labels']}")
                print(f"      Properties: {idx['properties']}")
        elif "vector_indexes_error" in results:
            print(f"  ⚠️ Error checking vector indexes: {results['vector_indexes_error']}")
            
        # Print embedding properties by node label
        found_embeddings = False
        
        for label, props in results.items():
            if label in ["vector_indexes", "vector_indexes_error", "error"]:
                continue
                
            embedding_props = []
            for prop_name, prop_info in props.items():
                embedding_props.append({
                    "property": prop_name,
                    "count": prop_info["count"],
                    "dimensions": prop_info["dimensions"],
                    "type": prop_info["type"]
                })
                
            if embedding_props:
                found_embeddings = True
                print(f"\n  📑 {label} Nodes:")
                
                for prop in embedding_props:
                    print(f"    - {prop['property']}: {prop['count']} nodes, {prop['dimensions']} dimensions, type: {prop['type']}")
                    
        if not found_embeddings:
            print("  ⚠️ No embedding properties found in any nodes")

    def drop_vector_index(self, index_name: str):
        """Drop a vector index by name.
        
        Args:
            index_name: Name of the vector index to drop
            
        Returns:
            True if successful, False otherwise
        """
        driver = self.connect()
        
        try:
            with driver.session(database=self.database) as session:
                print(f"\n🗑️ DROPPING VECTOR INDEX: {index_name}")
                result = session.run(f"""
                    DROP INDEX {index_name}
                """)
                print(f"  ✅ Vector index '{index_name}' successfully dropped")
                return True
        except Exception as e:
            print(f"  ❌ Error dropping vector index: {e}")
            return False
        finally:
            driver.close()
    
    def create_vector_index(self, index_name: str, node_label: str, property_name: str, dimensions: int = 512, similarity_function: str = "cosine"):
        """Create a vector index for a specified node label and property.
        
        Args:
            index_name: Name of the vector index to create
            node_label: Label of nodes to index
            property_name: Property containing vector embeddings
            dimensions: Dimension of the vector embeddings
            similarity_function: Similarity function to use (cosine, euclidean)
            
        Returns:
            True if successful, False otherwise
        """
        driver = self.connect()
        
        try:
            with driver.session(database=self.database) as session:
                print(f"\n📊 CREATING VECTOR INDEX: {index_name}")
                print(f"  - Node Label: {node_label}")
                print(f"  - Property: {property_name}")
                print(f"  - Dimensions: {dimensions}")
                print(f"  - Similarity: {similarity_function}")
                
                result = session.run(f"""
                    CALL db.index.vector.createNodeIndex(
                        '{index_name}',
                        '{node_label}',
                        '{property_name}',
                        {dimensions},
                        '{similarity_function}'
                    )
                """)
                
                print(f"  ✅ Vector index '{index_name}' successfully created")
                return True
        except Exception as e:
            print(f"  ❌ Error creating vector index: {e}")
            return False
        finally:
            driver.close()

    def get_chapter_information(self) -> Dict[str, Any]:
        """Get detailed information about chapters in the database.
        
        Returns:
            Dictionary containing chapter information including numbers, titles, and content counts
        """
        driver = self.connect()
        chapter_info = {
            "chapters": [],
            "total_chapters": 0,
            "chapter_numbers": [],
            "content_by_chapter": {}
        }
        
        try:
            with driver.session(database=self.database) as session:
                # Get all chapters with their numbers and titles
                result = session.run("""
                MATCH (c:Chapter)
                RETURN c.Number as chapter_number, 
                       c.title as title,
                       c.name as name,
                       c.chapter_number as alt_chapter_number,
                       id(c) as node_id
                ORDER BY COALESCE(c.Number, c.chapter_number, 0)
                """)
                
                chapters = []
                for record in result:
                    chapter_num = record['chapter_number'] or record['alt_chapter_number']
                    title = record['title'] or record['name'] or 'No title'
                    node_id = record['node_id']
                    
                    chapter_data = {
                        "number": chapter_num,
                        "title": title,
                        "node_id": node_id
                    }
                    chapters.append(chapter_data)
                
                chapter_info["chapters"] = chapters
                chapter_info["total_chapters"] = len(chapters)
                chapter_info["chapter_numbers"] = [ch["number"] for ch in chapters if ch["number"] is not None]
                
                # Get content count by chapter
                for chapter in chapters:
                    if chapter["number"] is not None:
                        content_result = session.run("""
                        MATCH (n)
                        WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example OR n:Content)
                        AND n.chapter_number = $chapter_num
                        WITH labels(n)[0] as content_type, count(n) as count
                        RETURN content_type, count
                        ORDER BY count DESC
                        """, chapter_num=chapter["number"])
                        
                        content_counts = {}
                        total_content = 0
                        for content_record in content_result:
                            content_type = content_record["content_type"]
                            count = content_record["count"]
                            content_counts[content_type] = count
                            total_content += count
                        
                        chapter_info["content_by_chapter"][chapter["number"]] = {
                            "title": chapter["title"],
                            "content_types": content_counts,
                            "total_content": total_content
                        }
                
        except Exception as e:
            chapter_info["error"] = str(e)
        finally:
            driver.close()
            
        return chapter_info
    
    def print_chapter_information(self, chapter_info=None):
        """Print formatted chapter information.
        
        Args:
            chapter_info: Results from get_chapter_information (if None, will run the query)
        """
        if chapter_info is None:
            chapter_info = self.get_chapter_information()
            
        print("\n📚 CHAPTER INFORMATION:")
        print("=" * 50)
        
        if "error" in chapter_info:
            print(f"❌ Error retrieving chapter information: {chapter_info['error']}")
            return
            
        if not chapter_info["chapters"]:
            print("⚠️ No chapters found in the database")
            return
            
        # Print basic chapter info
        for chapter in chapter_info["chapters"]:
            chapter_num = chapter["number"]
            title = chapter["title"]
            print(f"Chapter {chapter_num}: {title}")
        
        print(f"\nTotal Chapters Found: {chapter_info['total_chapters']}")
        print(f"Chapter Numbers Available: {chapter_info['chapter_numbers']}")
        
        # Print content distribution by chapter
        if chapter_info["content_by_chapter"]:
            print(f"\n📊 CONTENT BY CHAPTER:")
            print("-" * 30)
            
            for chapter_num in sorted(chapter_info["chapter_numbers"]):
                if chapter_num in chapter_info["content_by_chapter"]:
                    chapter_data = chapter_info["content_by_chapter"][chapter_num]
                    title = chapter_data["title"]
                    total = chapter_data["total_content"]
                    
                    print(f"\nChapter {chapter_num}: {title}")
                    print(f"  Total Content: {total:,} items")
                    
                    if chapter_data["content_types"]:
                        for content_type, count in chapter_data["content_types"].items():
                            print(f"    • {content_type}: {count:,}")

    def check_fastrp_embeddings(self, auto_fix=True):
        """Check for fastRP_embedding properties and automatically fix embedding issues.
        
        Args:
            auto_fix: Whether to automatically apply fixes without prompting
            
        Returns:
            Dictionary with results of the check and any applied fixes
        """
        driver = self.connect()
        results = {}
        
        try:
            with driver.session(database=self.database) as session:
                # Check for fastRP_embedding property in Content nodes
                print("\n📊 CHECKING FASTRP EMBEDDINGS:")
                dimensions = None
                for label in ["Content", "Chapter", "Document", "Equation", "Example", "Exercise"]:
                    try:
                        result = session.run(f"""
                            MATCH (n:{label})
                            WHERE n.fastRP_embedding IS NOT NULL
                            RETURN COUNT(n) as count
                        """)
                        count = result.single()["count"] if result.peek() else 0
                        print(f"  - Found {count} {label} nodes with fastRP_embedding property")
                        
                        # Get a sample node with fastRP_embedding to check its dimensions
                        if count > 0:
                            result = session.run(f"""
                                MATCH (n:{label}) 
                                WHERE n.fastRP_embedding IS NOT NULL
                                RETURN size(n.fastRP_embedding) as dimensions LIMIT 1
                            """)
                            dimensions = result.single()["dimensions"] if result.peek() else 0
                            print(f"    Dimensions: {dimensions}")
                            
                            # Check if these nodes also have embedding property
                            result = session.run(f"""
                                MATCH (n:{label})
                                WHERE n.fastRP_embedding IS NOT NULL AND n.embedding IS NOT NULL
                                RETURN COUNT(n) as count
                            """)
                            embedding_count = result.single()["count"] if result.peek() else 0
                            print(f"    Nodes with both fastRP_embedding and embedding: {embedding_count}")
                    except Exception as e:
                        print(f"  ⚠️ Error checking {label} nodes: {e}")
                
                # Check for vector indexes
                print("\n🔢 CHECKING VECTOR INDEXES:")
                result = session.run("""
                    SHOW INDEXES
                    YIELD name, type, labelsOrTypes, properties  
                    WHERE type = 'VECTOR'
                    RETURN name, labelsOrTypes, properties
                """)
                
                found_indexes = False
                index_name = None
                index_label = None
                index_property = None
                
                for record in result:
                    found_indexes = True
                    name = record["name"]
                    labels = record["labelsOrTypes"]
                    props = record["properties"]
                    
                    # Store information about the index
                    index_name = name
                    if labels and len(labels) > 0:
                        index_label = labels[0]
                    if props and len(props) > 0:
                        index_property = props[0]
                        
                    print(f"  - Index: {name}, Labels: {labels}, Properties: {props}")
                    
                if not found_indexes:
                    print("  ⚠️ No vector indexes found")
                
                # Check if we need to fix the embeddings
                print("\n💡 CHECKING IF EMBEDDINGS NEED FIXING:")
                
                # Count nodes that have fastRP_embedding but not embedding
                result = session.run("""
                    MATCH (n)
                    WHERE n.fastRP_embedding IS NOT NULL AND n.embedding IS NULL
                    RETURN COUNT(n) as count
                """)
                nodes_to_fix = result.single()["count"] if result.peek() else 0
                
                # OPTION 1: Copy fastRP_embedding to embedding 
                if nodes_to_fix > 0 and index_property == "embedding":
                    print(f"  - Found {nodes_to_fix} nodes with fastRP_embedding but no embedding property")
                    print(f"  - Vector index is using 'embedding' property")
                    print(f"  ⚠️ Option 1: Copy fastRP_embedding values to embedding property")
                    
                    apply_fix = auto_fix
                    if not auto_fix:
                        user_input = input("\nApply fix option 1? Copy fastRP_embedding to embedding (y/n): ").strip().lower()
                        apply_fix = user_input == 'y'
                        
                    if apply_fix:
                        print("\n🛠️ APPLYING FIX (OPTION 1):")
                        print("  Copying fastRP_embedding values to embedding property...")
                        
                        result = session.run("""
                            MATCH (n)
                            WHERE n.fastRP_embedding IS NOT NULL AND n.embedding IS NULL
                            SET n.embedding = n.fastRP_embedding
                            RETURN COUNT(n) as updated_nodes
                        """)
                        updated_nodes = result.single()["updated_nodes"] if result.peek() else 0
                        print(f"  ✅ Successfully copied fastRP_embedding to embedding for {updated_nodes} nodes")
                        results["fix_applied"] = "option1"
                        results["updated_nodes"] = updated_nodes
                
                # OPTION 2: Drop existing index and create a new one using fastRP_embedding
                elif index_property != "fastRP_embedding" and found_indexes and dimensions is not None:
                    print(f"  - Found vector index '{index_name}' using property '{index_property}'")
                    print(f"  - Have fastRP_embedding vectors with {dimensions} dimensions")
                    print(f"  ⚠️ Option 2: Recreate vector index to use fastRP_embedding directly")
                    
                    apply_fix = auto_fix
                    if not auto_fix:
                        user_input = input("\nApply fix option 2? Recreate vector index to use fastRP_embedding (y/n): ").strip().lower()
                        apply_fix = user_input == 'y'
                        
                    if apply_fix:
                        print("\n🛠️ APPLYING FIX (OPTION 2):")
                        
                        # Drop the existing index
                        if self.drop_vector_index(index_name):
                            # Create a new index using fastRP_embedding
                            success = self.create_vector_index(
                                index_name=index_name,
                                node_label=index_label,
                                property_name="fastRP_embedding",
                                dimensions=dimensions
                            )
                            if success:
                                results["fix_applied"] = "option2"
                                results["index_name"] = index_name
                                results["index_property"] = "fastRP_embedding"
                
                # No fix needed cases
                elif index_property == "fastRP_embedding":
                    print(f"  ✅ No fix needed: Vector index is already using 'fastRP_embedding' property directly")
                    results["fix_applied"] = "none_needed"
                elif nodes_to_fix == 0:
                    print(f"  ✅ No fix needed: All nodes with fastRP_embedding already have embedding property")
                    results["fix_applied"] = "none_needed"
                else:
                    print(f"  ⚠️ No vector index found or index using unknown property")
                    results["fix_applied"] = "none_possible"
                    
        except Exception as e:
            print(f"  ❌ Error checking fastRP embeddings: {e}")
            results["error"] = str(e)
        finally:
            driver.close()
            
        return results

if __name__ == "__main__":
    # Load environment variables
    load_env_vars()
    
    # Get Neo4j credentials
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    # Create explorer instance
    explorer = DatabaseExplorer(
        url=neo4j_creds["uri"],
        username=neo4j_creds["username"],
        password=neo4j_creds["password"],
        database=neo4j_creds["database"]
    )
    
    print("\n🔍 Exploring your Neo4j database...")
    results = explorer.explore_database(print_output=True)
    
    # Print specific educational metadata analysis in a readable format
    print("\n📚 EDUCATIONAL CONTENT STRUCTURE:")
    edu_metadata = explorer.analyze_educational_metadata()
    
    # Show content types
    if "content_types" in edu_metadata and edu_metadata["content_types"]:
        print(f"\n  Content Types ({len(edu_metadata['content_types'])}):")
        content_table = []
        headers = ["Type", "Count"]
        for content_type in edu_metadata["content_types"]:
            content_table.append([
                content_type.get("type", "N/A"),
                content_type.get("count", 0)
            ])
        print(tabulate(content_table, headers=headers, tablefmt="simple"))
    
    # Show learning objectives
    if "learning_objectives" in edu_metadata and edu_metadata["learning_objectives"]:
        print(f"\n  Learning Objectives (Top {len(edu_metadata['learning_objectives'])}):")
        objectives_table = []
        headers = ["Objective", "Count", "Node Type"]
        for objective in edu_metadata["learning_objectives"]:
            # Truncate long objectives
            obj_text = objective.get("objective", "N/A")
            if len(obj_text) > 50:
                obj_text = obj_text[:47] + "..."
                
            objectives_table.append([
                obj_text,
                objective.get("count", 0),
                objective.get("node_type", "N/A")
            ])
        print(tabulate(objectives_table, headers=headers, tablefmt="simple"))
    
    # Show topics
    if "topics" in edu_metadata and edu_metadata["topics"]:
        print(f"\n  Topics (Top {len(edu_metadata['topics'])}):")
        topics_table = []
        headers = ["Topic", "Count", "Node Type"]
        for topic in edu_metadata["topics"]:
            topics_table.append([
                topic.get("topic", "N/A"),
                topic.get("count", 0),
                topic.get("node_type", "N/A")
            ])
        print(tabulate(topics_table, headers=headers, tablefmt="simple"))
    
    # Show difficulties
    if "difficulties" in edu_metadata and edu_metadata["difficulties"]:
        print(f"\n  Difficulty Levels ({len(edu_metadata['difficulties'])}):")
        difficulty_table = []
        headers = ["Difficulty", "Count", "Node Type"]
        for difficulty in edu_metadata["difficulties"]:
            difficulty_table.append([
                difficulty.get("difficulty", "N/A"),
                difficulty.get("count", 0),
                difficulty.get("node_type", "N/A")
            ])
        print(tabulate(difficulty_table, headers=headers, tablefmt="simple"))
    
    # Show content structure
    if "structure" in edu_metadata and edu_metadata["structure"]:
        print(f"\n  Content Structure ({len(edu_metadata['structure'])}):")
        structure_table = []
        headers = ["Node Type", "Count", "Total Children", "Avg Children/Node"]
        for structure in edu_metadata["structure"]:
            structure_table.append([
                structure.get("node_type", "N/A"),
                structure.get("node_count", 0),
                structure.get("total_children", 0),
                f"{structure.get('avg_children_per_node', 0):.2f}"
            ])
        print(tabulate(structure_table, headers=headers, tablefmt="simple"))
    
    # Show concept relationships
    if "concept_relationships" in edu_metadata and edu_metadata["concept_relationships"]:
        print(f"\n  Concept Relationships (Top {len(edu_metadata['concept_relationships'])}):")
        relationships_table = []
        headers = ["Relationship", "Source", "Target", "Count"]
        for rel in edu_metadata["concept_relationships"]:
            relationships_table.append([
                rel.get("relationship_type", "N/A"),
                rel.get("source_type", "N/A"),
                rel.get("target_type", "N/A"),
                rel.get("count", 0)
            ])
        print(tabulate(relationships_table, headers=headers, tablefmt="simple"))
    
    print("\n🔍 Checking embedding properties...")
    explorer.print_embedding_properties()
    
    print("\n🔍 Checking and fixing fastRP embeddings...")
    explorer.check_fastrp_embeddings(auto_fix=True)
    
    print("\n📚 Getting chapter information...")
    explorer.print_chapter_information()
    
    print("\n✅ Database exploration complete!") 