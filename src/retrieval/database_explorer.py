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
    
    print("\n🔍 Checking embedding properties...")
    explorer.print_embedding_properties()
    
    print("\n🔍 Checking and fixing fastRP embeddings...")
    explorer.check_fastrp_embeddings(auto_fix=True)
    
    print("\n✅ Database exploration complete!") 