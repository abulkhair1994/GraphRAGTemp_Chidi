"""
Neo4j Database Explorer for GraphRAG.

This module provides functionality to explore the contents of a Neo4j database
for retrieval purposes, showing actual data without creating any fictional content.
"""

import json
from typing import Dict, List, Any, Optional

from neo4j import GraphDatabase
from tabulate import tabulate

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
            if schema["labels"]:
                print(f"  Node Labels: {', '.join(schema['labels'])}")
            else:
                print("  No node labels found in database")
                
            if schema["relationship_types"]:
                print(f"  Relationship Types: {', '.join(schema['relationship_types'])}")
            else:
                print("  No relationship types found in database")
                
            if schema["properties"]:
                print(f"  Properties: {', '.join(schema['properties'])}")
            else:
                print("  No properties found in database")
        
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
        
        return results

def explore_neo4j_database(url, username, password, database):
    """
    Connect to Neo4j and explore the actual database contents.
    
    Args:
        url: Neo4j connection URL
        username: Neo4j username
        password: Neo4j password
        database: Neo4j database name
    """
    explorer = DatabaseExplorer(url, username, password, database)
    print("\n🔍 Exploring your Neo4j database...")
    explorer.explore_database(print_output=True) 