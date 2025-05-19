"""
Example demonstrating the GraphRAG retrieval system.

This script shows how to use the vector and graph-based retrieval engines
together for enhanced context retrieval in RAG applications.
"""

import sys
import os
from pathlib import Path
import textwrap
import time
from typing import Dict, List, Any, Optional
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import environment utilities and retrieval engines
from src.utils.env_manager import load_env_vars, EnvManager
from src.retrieval import GraphRAGRetriever, GraphRAGResult
from src.retrieval.database_explorer import DatabaseExplorer

# Import langchain components
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

def debug_database_schema(url, username, password, database):
    """Check the database schema to better understand the data organization."""
    print(f"\n{Fore.CYAN}🔍 Examining Neo4j database schema...{Style.RESET_ALL}")
    
    driver = GraphDatabase.driver(url, auth=(username, password))
    
    try:
        with driver.session(database=database) as session:
            # Check for labels
            result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
            labels = [record["label"] for record in result]
            print(f"  {Fore.GREEN}📊 Node labels in database:{Style.RESET_ALL}")
            for label in labels:
                print(f"    - {label}")
            
            # Check for relationship types
            result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
            rel_types = [record["relationshipType"] for record in result]
            print(f"  {Fore.GREEN}🔗 Relationship types in database:{Style.RESET_ALL}")
            for rel_type in rel_types:
                print(f"    - {rel_type}")
            
            # Total node count
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            print(f"  {Fore.GREEN}📊 Total node count:{Style.RESET_ALL} {node_count}")
            
            # Sample of a few node types with counts
            for label in labels[:5]:  # Show first 5 labels
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                label_count = result.single()["count"]
                print(f"    - {label}: {label_count} nodes")
                
                # Get properties for this label
                result = session.run(f"""
                    MATCH (n:{label})
                    WITH n LIMIT 1
                    RETURN keys(n) as props
                """)
                props = result.single()["props"] if result.peek() else []
                if props:
                    print(f"      Properties: {', '.join(props)}")
                
    except Exception as e:
        print(f"  {Fore.RED}❌ Error querying Neo4j schema: {e}{Style.RESET_ALL}")
    finally:
        driver.close()
    
    print(f"{Fore.CYAN}🔍 Schema examination complete{Style.RESET_ALL}")

def check_documents_in_neo4j(url, username, password, database):
    """Check if documents are properly stored in Neo4j."""
    print(f"\n{Fore.CYAN}🔎 Checking Neo4j database for documents...{Style.RESET_ALL}")
    
    driver = GraphDatabase.driver(url, auth=(username, password))
    
    try:
        with driver.session(database=database) as session:
            # Try to detect document nodes
            result = session.run("""
                MATCH (n) 
                WHERE n.page_content IS NOT NULL OR n.text IS NOT NULL
                RETURN labels(n)[0] as label, count(n) as count
            """)
            
            doc_labels = []
            for record in result:
                label = record["label"]
                count = record["count"]
                doc_labels.append(label)
                print(f"  {Fore.GREEN}📄 Found potential document nodes:{Style.RESET_ALL} {label} ({count} nodes)")
            
            if not doc_labels:
                print(f"  {Fore.YELLOW}⚠️ No nodes with page_content or text properties found{Style.RESET_ALL}")
                
                # Try to check if there are any nodes with text-like properties
                result = session.run("""
                    MATCH (n)
                    WITH n LIMIT 100
                    UNWIND keys(n) AS prop
                    WITH prop, apoc.meta.type(n[prop]) AS type, count(*) AS cnt
                    WHERE type = 'STRING' AND cnt > 10
                    RETURN prop, cnt ORDER BY cnt DESC LIMIT 5
                """)
                
                if result.peek():
                    print(f"  {Fore.YELLOW}⚠️ Found potential text properties:{Style.RESET_ALL}")
                    for record in result:
                        print(f"    - {record['prop']}: {record['cnt']} occurrences")
            
            # Check for Document labeled nodes anyway (common convention)
            for potential_label in ['Document', 'Chunk', 'TextChunk', 'Content']:
                result = session.run(f"""
                    MATCH (d:{potential_label})
                    RETURN count(d) as count
                """)
                doc_count = result.single()["count"] if result.peek() else 0
                
                if doc_count > 0:
                    print(f"  {Fore.GREEN}📊 {potential_label} nodes found:{Style.RESET_ALL} {doc_count}")
                    
                    # Get sample document contents
                    result = session.run(f"""
                        MATCH (d:{potential_label})
                        RETURN CASE 
                            WHEN d.page_content IS NOT NULL THEN d.page_content
                            WHEN d.text IS NOT NULL THEN d.text
                            WHEN d.content IS NOT NULL THEN d.content
                            ELSE 'No content found'
                        END AS content, 
                        properties(d) as props
                        LIMIT 1
                    """)
                    
                    if result.peek():
                        record = result.single()
                        content = record["content"]
                        props = record["props"]
                        
                        print(f"  {Fore.GREEN}📄 Sample {potential_label} properties:{Style.RESET_ALL}")
                        for key, value in props.items():
                            if isinstance(value, str) and len(value) > 50:
                                print(f"    - {key}: {value[:50]}...")
                            elif key == "embedding":
                                print(f"    - {key}: [Vector with {len(value)} dimensions]")
                            else:
                                print(f"    - {key}: {value}")
            
            # Check for vector index
            result = session.run("""
                SHOW INDEXES 
                YIELD name, type, labelsOrTypes, properties
                WHERE type = 'VECTOR'
                RETURN name, labelsOrTypes, properties
            """)
            
            print(f"  {Fore.GREEN}🔢 Vector indexes:{Style.RESET_ALL}")
            indexes_found = False
            for record in result:
                indexes_found = True
                print(f"    - Name: {record['name']}")
                print(f"      Labels: {record['labelsOrTypes']}")
                print(f"      Properties: {record['properties']}")
            
            if not indexes_found:
                print(f"    {Fore.YELLOW}⚠️ No vector indexes found{Style.RESET_ALL}")
                
            # Check if documents have embeddings
            for label in doc_labels or ['Document', 'Chunk', 'TextChunk', 'Content']:
                # Try with multiple potential embedding property names
                for emb_prop in ['embedding', 'embeddings', 'vector', 'vectors']:
                    result = session.run(f"""
                        MATCH (d:{label})
                        WHERE d.{emb_prop} IS NOT NULL
                        RETURN count(d) as count
                    """)
                    if result.peek():
                        docs_with_embeddings = result.single()["count"]
                        if docs_with_embeddings > 0:
                            print(f"  {Fore.GREEN}📊 {label} nodes with {emb_prop}:{Style.RESET_ALL} {docs_with_embeddings}")
                
    except Exception as e:
        print(f"  {Fore.RED}❌ Error querying Neo4j: {e}{Style.RESET_ALL}")
    finally:
        driver.close()
    
    print(f"{Fore.CYAN}🔎 Database check complete{Style.RESET_ALL}")

def print_separator(width=80):
    """Print a separator line."""
    print(f"{Fore.YELLOW}{'-' * width}{Style.RESET_ALL}")

def print_query_header(query: str, query_num: int):
    """Print a formatted query header."""
    print_separator()
    print(f"{Fore.CYAN}📝 QUERY #{query_num}:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{Style.BRIGHT}\"{query}\"{Style.RESET_ALL}")
    print_separator()

def format_content(content: str, indent=2, width=80) -> str:
    """Format content with proper indentation and wrapping."""
    wrapper = textwrap.TextWrapper(
        width=width, 
        initial_indent=' ' * indent, 
        subsequent_indent=' ' * indent
    )
    return wrapper.fill(content)

def display_results(result: GraphRAGResult):
    """Display formatted results from both vector and graph engines."""
    # Vector results section
    print(f"\n{Fore.MAGENTA}🔍 VECTOR ENGINE RESULTS:{Style.RESET_ALL}")
    if result.vector_documents:
        for j, doc in enumerate(result.vector_documents, 1):
            print(f"  {Fore.GREEN}Document {j}:{Style.RESET_ALL}")
            content = doc.page_content
            wrapped_content = format_content(content, indent=4)
            print(wrapped_content)
            print(f"    {Fore.BLUE}Source:{Style.RESET_ALL} {doc.metadata.get('source', 'Unknown')}")
            # Add a small separator between documents
            if j < len(result.vector_documents):
                print(f"    {'-' * 40}")
    else:
        print(f"  {Fore.YELLOW}No results found using vector search{Style.RESET_ALL}")
    
    # Graph results section
    print(f"\n{Fore.MAGENTA}🕸️ GRAPH ENGINE RESULTS:{Style.RESET_ALL}")
    if result.graph_result and (result.graph_result.entities or result.graph_result.relationships):
        if result.graph_result.entities:
            print(f"  {Fore.GREEN}Entities ({len(result.graph_result.entities)}):{Style.RESET_ALL}")
            for entity in result.graph_result.entities[:10]:  # Show first 10
                entity_name = entity.get('name', 'Unknown')
                entity_type = entity.get('type', 'Unknown')
                properties = entity.get('properties', {})
                
                # Format properties if they exist
                prop_str = ""
                if properties and len(properties) > 0:
                    prop_items = [f"{k}: {v}" for k, v in properties.items() 
                                 if k not in ['name', 'type'] and v is not None 
                                 and not isinstance(v, (list, dict)) and len(str(v)) < 50]
                    if prop_items:
                        prop_str = f" - Properties: {{{', '.join(prop_items)}}}"
                
                print(f"    - {Fore.CYAN}{entity_name}{Style.RESET_ALL} ({Fore.YELLOW}{entity_type}{Style.RESET_ALL}){prop_str}")
            
            if len(result.graph_result.entities) > 10:
                print(f"      {Fore.YELLOW}(+ {len(result.graph_result.entities) - 10} more entities){Style.RESET_ALL}")
        
        if result.graph_result.relationships:
            print(f"  {Fore.GREEN}Relationships ({len(result.graph_result.relationships)}):{Style.RESET_ALL}")
            for rel in result.graph_result.relationships[:10]:  # Show first 10
                source = rel.get('source', 'Unknown')
                target = rel.get('target', 'Unknown')
                rel_type = rel.get('type', 'Unknown')
                print(f"    - {Fore.CYAN}{source}{Style.RESET_ALL} --[{Fore.YELLOW}{rel_type}{Style.RESET_ALL}]--> {Fore.CYAN}{target}{Style.RESET_ALL}")
            
            if len(result.graph_result.relationships) > 10:
                print(f"      {Fore.YELLOW}(+ {len(result.graph_result.relationships) - 10} more relationships){Style.RESET_ALL}")
    else:
        print(f"  {Fore.YELLOW}No results found using graph search{Style.RESET_ALL}")
    
    # Analysis section
    print(f"\n{Fore.MAGENTA}🔄 RETRIEVAL ANALYSIS:{Style.RESET_ALL}")
    if result.vector_documents and (result.graph_result.entities or result.graph_result.relationships):
        print(f"  {Fore.GREEN}✓ Both engines returned results{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Vector documents: {len(result.vector_documents)}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Graph entities: {len(result.graph_result.entities)}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Graph relationships: {len(result.graph_result.relationships)}{Style.RESET_ALL}")
    elif result.vector_documents:
        print(f"  {Fore.YELLOW}⚠ Only the VECTOR engine returned results{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Vector documents: {len(result.vector_documents)}{Style.RESET_ALL}")
    elif result.graph_result.entities or result.graph_result.relationships:
        print(f"  {Fore.YELLOW}⚠ Only the GRAPH engine returned results{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Graph entities: {len(result.graph_result.entities)}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Graph relationships: {len(result.graph_result.relationships)}{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}✗ No results found from either engine{Style.RESET_ALL}")
        print(f"    {Fore.RED}This could indicate an issue with the database or retrieval settings.{Style.RESET_ALL}")
        print(f"    {Fore.RED}Please review the debug output above for more information.{Style.RESET_ALL}")

def main():
    """Run a simple example of GraphRAG retrieval on the existing College Algebra database."""
    # Load environment variables
    load_env_vars()
    
    # Get Neo4j credentials
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    # Check if OpenAI API key is set
    openai_api_key = EnvManager.get_openai_api_key()
    if not openai_api_key:
        print(f"{Fore.RED}OpenAI API key not found. Please set OPENAI_API_KEY environment variable.{Style.RESET_ALL}")
        return
        
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}✨ GraphRAG Retrieval Example - Neo4j Database ✨{Style.RESET_ALL}\n")
    
    # Examine the database schema for better understanding
    debug_database_schema(
        neo4j_creds["uri"],
        neo4j_creds["username"],
        neo4j_creds["password"],
        neo4j_creds["database"]
    )
    
    # Check the existing documents in Neo4j
    check_documents_in_neo4j(
        neo4j_creds["uri"],
        neo4j_creds["username"],
        neo4j_creds["password"],
        neo4j_creds["database"]
    )
    
    # Use database explorer to get more detailed information
    print(f"\n{Fore.CYAN}🔍 Getting additional database insights with DatabaseExplorer...{Style.RESET_ALL}")
    try:
        explorer = DatabaseExplorer(
            neo4j_creds["uri"],
            neo4j_creds["username"],
            neo4j_creds["password"],
            neo4j_creds["database"]
        )
        
        # Get schema overview
        schema = explorer.get_schema_overview()
        print(f"  {Fore.GREEN}📊 Database Overview:{Style.RESET_ALL}")
        for node_type, count in schema["node_counts"].items():
            print(f"    - {node_type}: {count} nodes")
            
        # Get most connected nodes
        print(f"\n  {Fore.GREEN}🔗 Most Connected Nodes:{Style.RESET_ALL}")
        try:
            connected_nodes = explorer.get_most_connected_nodes(limit=5)
            for node in connected_nodes:
                print(f"    - {node['name']} ({node['label']}): {node['connection_count']} connections")
        except Exception as e:
            print(f"    {Fore.YELLOW}⚠️ Could not get most connected nodes: {e}{Style.RESET_ALL}")
            
        # Detect node labels that might contain text
        print(f"\n  {Fore.GREEN}📄 Detecting Text Content Nodes:{Style.RESET_ALL}")
        try:
            text_nodes = explorer.detect_text_content_nodes()
            for label, details in text_nodes.items():
                print(f"    - {label}: {details['count']} nodes containing text in '{details['property']}' property")
        except Exception as e:
            print(f"    {Fore.YELLOW}⚠️ Could not detect text content nodes: {e}{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"  {Fore.RED}❌ Error using DatabaseExplorer: {e}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}Initializing GraphRAG retriever...{Style.RESET_ALL}")
    try:
        # Try to initialize with auto-detection of document node label
        retriever = GraphRAGRetriever(
            neo4j_url=neo4j_creds["uri"],
            neo4j_username=neo4j_creds["username"],
            neo4j_password=neo4j_creds["password"],
            neo4j_database=neo4j_creds["database"],
            embedding_model=embeddings,
        )
        
        # Print detected settings
        print(f"  {Fore.GREEN}✓ Successfully initialized GraphRAG retriever:{Style.RESET_ALL}")
        print(f"    - {Fore.CYAN}Document node label:{Style.RESET_ALL} {retriever.node_label or 'Auto-detecting'}")
        if retriever.db_schema and retriever.db_schema["labels"]:
            print(f"    - {Fore.CYAN}Available entity types:{Style.RESET_ALL} {', '.join(retriever.db_schema['labels'][:5])}...")
        if retriever.db_schema and retriever.db_schema["relationship_types"]:
            print(f"    - {Fore.CYAN}Available relationship types:{Style.RESET_ALL} {', '.join(retriever.db_schema['relationship_types'][:5])}...")
        
        # Create a list of queries focusing on College Algebra
        queries = [
            "How many chapters are in the College Algebra curriculum?",
            "What are the main concepts related to functions in College Algebra?"
        ]
        
        # Run sample queries
        for i, query in enumerate(queries, 1):
            print_query_header(query, i)
            
            print(f"{Fore.BLUE}⏳ Retrieving information for query...{Style.RESET_ALL}")
            start_time = time.time()
            
            # Retrieve with combined approach
            result = retriever.retrieve(query)
            
            # Calculate time
            elapsed_time = time.time() - start_time
            print(f"{Fore.BLUE}⌛ Retrieval completed in {elapsed_time:.2f} seconds{Style.RESET_ALL}")
            
            # Display formatted results
            display_results(result)
        
        # Close the retriever
        retriever.close()
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error initializing or using GraphRAG retriever: {e}{Style.RESET_ALL}")
        print(f"{Fore.RED}Please check your database connection and schema.{Style.RESET_ALL}")
        
    print(f"\n{Fore.GREEN}✅ GraphRAG retrieval example completed!{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 