#!/usr/bin/env python3
"""
Test script for GraphRAG Hybrid Search functionality.

This script demonstrates the new hybrid search capabilities that combine:
- Dense vector search (semantic similarity)
- Fulltext search (keyword matching)
- Score combination and re-ranking
- Result deduplication
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.env_manager import load_env_vars, EnvManager
from src.retrieval import GraphRAGRetriever
from colorama import Fore, Style, init
from langchain_openai import OpenAIEmbeddings

# Initialize colorama
init(autoreset=True)

def test_hybrid_search():
    """Test the hybrid search functionality."""
    print(f"{Fore.CYAN}🔬 Testing GraphRAG Hybrid Search{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 50}{Style.RESET_ALL}")
    
    # Load environment variables
    load_env_vars()
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    # Initialize embedding model
    print(f"{Fore.YELLOW}🧠 Initializing embedding model...{Style.RESET_ALL}")
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Initialize retriever with hybrid search enabled
    retriever = GraphRAGRetriever(
        neo4j_url=neo4j_creds["uri"],
        neo4j_username=neo4j_creds["username"],
        neo4j_password=neo4j_creds["password"],
        neo4j_database=neo4j_creds["database"],
        embedding_model=embedding_model,  # Pass the embedding model
        enable_hybrid_search=True,
        hybrid_alpha=0.5,  # Equal weight to vector and fulltext
        vector_text_node_property="text_content"
    )
    
    # Setup hybrid search indexes
    print(f"{Fore.YELLOW}📋 Setting up hybrid search indexes...{Style.RESET_ALL}")
    retriever.setup_hybrid_search_indexes()
    
    # Test queries
    test_queries = [
        "quadratic functions and parabolas",
        "polynomial equations solving",
        "linear functions graphing",
        "chapter 4 exercises"
    ]
    
    for query in test_queries:
        print(f"\n{Fore.GREEN}🔍 Testing Query: '{query}'{Style.RESET_ALL}")
        
        # Test different search types
        search_types = ["hybrid", "vector", "fulltext"]
        
        for search_type in search_types:
            print(f"\n{Fore.CYAN}  📊 {search_type.upper()} Search:{Style.RESET_ALL}")
            
            try:
                result = retriever.hybrid_search(
                    query=query,
                    k=3,
                    search_type=search_type
                )
                
                print(f"    ✅ Found {len(result.vector_documents)} documents")
                print(f"    📈 Average Score: {result.vector_score:.3f}" if result.vector_score else "    📈 No scores available")
                print(f"    🎯 Strategy: {result.search_strategy}")
                
                # Show top result
                if result.vector_documents:
                    top_doc = result.vector_documents[0]
                    content_preview = top_doc.page_content[:100] + "..." if len(top_doc.page_content) > 100 else top_doc.page_content
                    print(f"    📝 Top Result: {content_preview}")
                    print(f"    🏷️  Type: {top_doc.metadata.get('type', 'Unknown')}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        print(f"{Fore.BLUE}{'-' * 50}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}✅ Hybrid Search Test Complete!{Style.RESET_ALL}")

if __name__ == "__main__":
    test_hybrid_search() 