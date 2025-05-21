#!/usr/bin/env python3
"""
Add text_content property to Content nodes for text-based searching.

This script adds a text_content property to Content nodes by collecting 
relevant textual properties from the nodes and combining them.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from neo4j import GraphDatabase
from src.utils.env_manager import load_env_vars, EnvManager
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def add_text_content_property():
    """Add text_content property to Content nodes for text-based search."""
    # Load environment variables
    load_env_vars()
    
    # Get Neo4j credentials
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        neo4j_creds["uri"],
        auth=(neo4j_creds["username"], neo4j_creds["password"])
    )
    
    try:
        print(f"{Fore.CYAN}Adding text_content property to Content nodes...{Style.RESET_ALL}")
        
        with driver.session(database=neo4j_creds["database"]) as session:
            # First check available properties on Content nodes
            result = session.run("""
                MATCH (n:Content) 
                WITH keys(n) AS props
                UNWIND props AS prop
                WHERE prop <> 'fastRP_embedding' AND prop <> 'embedding'
                RETURN DISTINCT prop
            """)
            
            available_props = [record["prop"] for record in result]
            
            print(f"Available properties on Content nodes: {', '.join(available_props)}")
            
            # Count Content nodes
            result = session.run("MATCH (n:Content) RETURN COUNT(n) as count")
            content_count = result.single()["count"] if result.peek() else 0
            
            print(f"Found {content_count} Content nodes")
            
            # Create text_content by combining available properties
            update_query = """
            MATCH (n:Content)
            WHERE n.text_content IS NULL
            WITH n, 
                 CASE WHEN n.id IS NOT NULL THEN "ID: " + n.id ELSE "" END + 
                 CASE WHEN n.module_id IS NOT NULL THEN " Module: " + n.module_id ELSE "" END + 
                 CASE WHEN n.label IS NOT NULL THEN " Label: " + n.label ELSE "" END AS textContent
            SET n.text_content = textContent
            RETURN COUNT(n) as updated
            """
            
            result = session.run(update_query)
            updated_count = result.single()["updated"] if result.peek() else 0
            
            print(f"{Fore.GREEN}✓ Added text_content to {updated_count} Content nodes{Style.RESET_ALL}")
            
            # Verify text_content was added
            result = session.run("""
                MATCH (n:Content)
                WHERE n.text_content IS NOT NULL
                RETURN COUNT(n) as count
            """)
            
            verify_count = result.single()["count"] if result.peek() else 0
            print(f"Total Content nodes with text_content: {verify_count}")
            
            # Sample text_content from a few nodes
            result = session.run("""
                MATCH (n:Content)
                WHERE n.text_content IS NOT NULL
                RETURN n.text_content AS content LIMIT 3
            """)
            
            print("\nSample text_content values:")
            for i, record in enumerate(result, 1):
                print(f"  {i}. {record['content']}")
                
            return {
                "total_nodes": content_count,
                "updated_nodes": updated_count,
                "nodes_with_text_content": verify_count
            }
    except Exception as e:
        print(f"{Fore.RED}Error adding text_content: {e}{Style.RESET_ALL}")
        return {"error": str(e)}
    finally:
        driver.close()

def main():
    """Run the script to add text_content property to Content nodes."""
    print(f"{Fore.CYAN}=== Adding text_content Property to Content Nodes ==={Style.RESET_ALL}")
    result = add_text_content_property()
    
    if "error" in result:
        print(f"{Fore.RED}Failed to add text_content property: {result['error']}{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.GREEN}Summary:{Style.RESET_ALL}")
        print(f"Total Content nodes: {result['total_nodes']}")
        print(f"Nodes updated: {result['updated_nodes']}")
        print(f"Nodes with text_content: {result['nodes_with_text_content']}")
        
        if result['nodes_with_text_content'] > 0:
            print(f"\n{Fore.GREEN}✓ Success! Text content added to Content nodes.{Style.RESET_ALL}")
            print(f"You can now run the GraphRAG retrieval test again to check if vector retrieval works better.")
        else:
            print(f"\n{Fore.RED}✗ Failed! No nodes have text_content property.{Style.RESET_ALL}")
    
if __name__ == "__main__":
    main() 