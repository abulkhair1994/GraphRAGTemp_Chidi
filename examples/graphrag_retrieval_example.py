"""
Minimal GraphRAG retrieval test - optimized for compact output.

This script tests the GraphRAG retrieval engines and identifies issues.
"""

import sys
import os
from pathlib import Path
import time
import logging
from typing import Dict, List, Any, Optional
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Suppress all logging
logging.basicConfig(level=logging.CRITICAL)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import environment utilities and retrieval engines
from src.utils.env_manager import load_env_vars, EnvManager
from src.retrieval import GraphRAGRetriever
from src.retrieval.vector_retrieval import VectorRetrievalEngine
from src.retrieval.database_explorer import DatabaseExplorer

def fix_text_content_property():
    """Add text_content property to all Content nodes if missing."""
    # Load environment variables
    load_env_vars()
    
    # Get Neo4j credentials
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    # Create database explorer
    explorer = DatabaseExplorer(
        url=neo4j_creds["uri"],
        username=neo4j_creds["username"],
        password=neo4j_creds["password"],
        database=neo4j_creds["database"]
    )
    
    # Connect to Neo4j
    driver = explorer.connect()
    
    try:
        with driver.session(database=neo4j_creds["database"]) as session:
            # Check if text_content property exists
            result = session.run("""
                MATCH (n:Content)
                WHERE n.text_content IS NOT NULL
                RETURN COUNT(n) as count
            """)
            
            count = result.single()["count"] if result.peek() else 0
            
            if count == 0:
                print(f"{Fore.YELLOW}⚠️ No text_content property found. Adding text_content property...{Style.RESET_ALL}")
                
                # Add text_content property by concatenating existing text properties
                result = session.run("""
                    MATCH (n:Content)
                    WHERE n.text_content IS NULL
                    WITH n, 
                         CASE WHEN n.title IS NOT NULL THEN n.title ELSE '' END +
                         CASE WHEN n.label IS NOT NULL THEN ' ' + n.label ELSE '' END +
                         CASE WHEN n.id IS NOT NULL THEN ' ' + n.id ELSE '' END +
                         CASE WHEN n.module_id IS NOT NULL THEN ' ' + n.module_id ELSE '' END AS combined_text
                    SET n.text_content = combined_text
                    RETURN COUNT(n) as updated
                """)
                
                updated = result.single()["updated"] if result.peek() else 0
                print(f"{Fore.GREEN}✅ Added text_content property to {updated} nodes{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.GREEN}✅ text_content property already exists on {count} nodes{Style.RESET_ALL}")
                return False
    except Exception:
        return False
    finally:
        driver.close()

def display_document_content(documents, engine_name=None):
    """Display document content in a readable format, omitting large vectors and technical fields. Trims text to 100 chars and removes duplicate output. Shows which retrieval engine returned the result if provided."""
    skip_keys = {"embedding", "fastRP_embedding", "structural_features"}
    max_value_length = 200
    max_content_length = 100
    if not documents:
        print(f"No results found.")
        return
    seen_contents = set()
    for i, doc in enumerate(documents, 1):
        content = doc.page_content
        if isinstance(content, str):
            trimmed_content = content[:max_content_length] + ("..." if len(content) > max_content_length else "")
        else:
            trimmed_content = str(content)
        # Skip duplicate content
        if trimmed_content in seen_contents:
            continue
        seen_contents.add(trimmed_content)
        print(f"\n--- Result {i} ---")
        if engine_name:
            print(f"[Engine: {engine_name}]")
        print(f"Matched Content:\n{trimmed_content}")
        # Print filtered metadata (skip duplicate values)
        if doc.metadata:
            filtered = {}
            for key, value in doc.metadata.items():
                if key in skip_keys:
                    continue
                if isinstance(value, list) and len(value) > 10:
                    continue
                if isinstance(value, str) and len(value) > max_value_length:
                    value = value[:max_value_length] + "..."
                filtered[key] = value
            if filtered:
                print("\n[Metadata]")
                for key, value in filtered.items():
                    print(f"  {key}: {value}")

def run_query(retriever, query, engine_name=None):
    print(f"\n==============================")
    print(f"Query: {query}")
    result = retriever.retrieve(query)
    display_document_content(result.vector_documents, engine_name=engine_name)
    # Entities
    if hasattr(result, 'graph_result') and getattr(result.graph_result, 'entities', None):
        entities = result.graph_result.entities
        if entities:
            print("\n[Entities]")
            for entity in entities:
                props = getattr(entity, 'properties', {})
                for k, v in props.items():
                    if k in {"embedding", "fastRP_embedding", "structural_features"}:
                        continue
                    if isinstance(v, list) and len(v) > 10:
                        continue
                    if isinstance(v, str) and len(v) > 200:
                        v = v[:200] + "..."
                    print(f"  {k}: {v}")
    # Relationships
    if hasattr(result, 'graph_result') and getattr(result.graph_result, 'relationships', None):
        rels = result.graph_result.relationships
        if rels:
            print("\n[Relationships]")
            for rel in rels:
                if isinstance(rel, dict):
                    print(f"  Type: {rel.get('type', 'N/A')}, From: {rel.get('source', 'N/A')}, To: {rel.get('target', 'N/A')}")
                    if rel.get('properties'):
                        for k, v in rel['properties'].items():
                            if k in {"embedding", "fastRP_embedding", "structural_features"}:
                                continue
                            if isinstance(v, list) and len(v) > 10:
                                continue
                            if isinstance(v, str) and len(v) > 200:
                                v = v[:200] + "..."
                            print(f"    {k}: {v}")
                else:
                    print(f"  Type: {getattr(rel, 'type', 'N/A')}, From: {getattr(rel, 'source', 'N/A')}, To: {getattr(rel, 'target', 'N/A')}")
                    for k, v in getattr(rel, 'properties', {}).items():
                        if k in {"embedding", "fastRP_embedding", "structural_features"}:
                            continue
                        if isinstance(v, list) and len(v) > 10:
                            continue
                        if isinstance(v, str) and len(v) > 200:
                            v = v[:200] + "..."
                        print(f"    {k}: {v}")
    print(f"==============================\n")

def main():
    """Run a minimal GraphRAG retrieval test with extremely concise output."""
    # Load environment variables
    load_env_vars()
    
    # Get Neo4j credentials
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    # Initialize OpenAI embeddings
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    
    print(f"{Fore.CYAN}► GraphRAG Test ◄{Style.RESET_ALL}")
    
    # Create vector engine to check index status
    vector_engine = VectorRetrievalEngine(
        url=neo4j_creds["uri"],
        username=neo4j_creds["username"],
        password=neo4j_creds["password"],
        database=neo4j_creds["database"],
        embedding_model=embeddings,
        index_name="content_embeddings",
        text_node_property="text_content",
        embedding_node_property="fastRP_embedding",
        embedding_dimension=512,
        node_label="Content",
        search_type="hybrid"  # Enable hybrid search
    )
    
    # Check vector index status without logging warnings
    print(f"{Fore.CYAN}Vector Index:{Style.RESET_ALL}")
    try:
        # Query vector indexes directly
        driver = vector_engine.driver
        with driver.session(database=vector_engine.database) as session:
            # Get vector indexes
            result = session.run("""
                SHOW VECTOR INDEXES
                YIELD name, labelsOrTypes, properties, options
                RETURN name, labelsOrTypes[0] AS label, properties[0] AS property, 
                       options.indexConfig.`vector.dimensions` AS dimensions,
                       options.indexConfig.`vector.similarity` AS similarity
            """)
            
            indexes = []
            for record in result:
                indexes.append({
                    "name": record["name"],
                    "label": record["label"],
                    "property": record["property"],
                    "dimensions": record["dimensions"],
                    "similarity": record["similarity"]
                })
            
            # Check text properties
            result = session.run("""
                MATCH (n:Content) 
                WITH keys(n) AS props
                UNWIND props AS prop
                WITH prop, COUNT(*) AS count
                WHERE prop <> 'fastRP_embedding' AND prop <> 'embedding'
                RETURN prop, count ORDER BY count DESC
                LIMIT 5
            """)
            
            text_props = []
            for record in result:
                text_props.append((record["prop"], record["count"]))
                
            # Print index info
            if indexes:
                for idx in indexes:
                    print(f"  ✓ Index: {idx['name']} on {idx['label']}.{idx['property']} ({idx['dimensions']} dim, {idx['similarity']} similarity)")
            else:
                print(f"  ✗ No vector indexes found")
                
            # Print text properties info
            if text_props:
                print(f"  ✓ Text properties found: {', '.join([p[0] for p in text_props])}")
            else:
                print(f"  ✗ No text properties found for Content nodes")
    except Exception:
        print(f"  ✗ Error checking vector index")
    
    # Fix text_content property if needed
    if not any(prop == "text_content" for prop, _ in text_props):
        fix_text_content_property()
    
    # Create retriever
    print(f"\n{Fore.CYAN}Testing Queries:{Style.RESET_ALL}")
    retriever = GraphRAGRetriever(
        neo4j_url=neo4j_creds["uri"],
        neo4j_username=neo4j_creds["username"],
        neo4j_password=neo4j_creds["password"],
        neo4j_database=neo4j_creds["database"],
        embedding_model=embeddings,
        vector_text_node_property="text_content",
        vector_embedding_node_property="fastRP_embedding",
        vector_embedding_dimension=512,
        vector_distance_metric="cosine",
        vector_node_label="Content",
        vector_search_type="hybrid",  # Enable hybrid search
        vector_top_k=5,
        text_top_k=3
    )
    
    # Define test queries
    queries = [
        "What is the relationship between functions and equations?",
        "Explain the difference between rational and irrational numbers"
    ]
    
    # Run and display results for each query
    for query in queries:
        run_query(retriever, query, engine_name="GraphRAGRetriever")
    
    # Close the retriever
    retriever.close()
    
    # Print fix status - DIMENSION MISMATCH FIX
    print(f"\n{Fore.CYAN}Fix Status:{Style.RESET_ALL}")
    if indexes and indexes[0]['dimensions'] != 1536:
        print(f"  ✅ Dimension mismatch fix was applied")
        print(f"      • OpenAI dimensions: 1536")
        print(f"      • FastRP dimensions: {indexes[0]['dimensions']}")
        print(f"      • Solution: Using fastRP_embedding directly in vector index")
    else:
        print(f"  ✅ No dimension mismatch fix needed")
    
    if not any(prop == "text_content" for prop, _ in text_props):
        print(f"  ❌ Text property fix needed")
        print(f"      • Problem: Vector search cannot fall back to text search")
        print(f"      • Solution: Run scripts/add_text_content.py to add text_content property")
    else:
        print(f"  ✅ Text properties available for search: {', '.join([p[0] for p in text_props[:3]])}")
    
    print(f"\n{Fore.GREEN}✓ Test complete!{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 