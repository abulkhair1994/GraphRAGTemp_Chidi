#!/usr/bin/env python3
"""
Quick test to verify chapter-specific search functionality
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.env_manager import load_env_vars, EnvManager
from src.retrieval import GraphRAGRetriever

def test_chapter_search():
    """Test chapter-specific search functionality."""
    print("🔍 Testing Chapter-Specific Search Functionality")
    print("=" * 60)
    
    # Setup
    load_env_vars()
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    retriever = GraphRAGRetriever(
        neo4j_url=neo4j_creds["uri"],
        neo4j_username=neo4j_creds["username"],
        neo4j_password=neo4j_creds["password"],
        neo4j_database=neo4j_creds["database"],
        vector_text_node_property="text_content",
        vector_node_label="Problem"
    )
    
    # Test queries with their expected chapter numbers
    test_cases = [
        ("Show me exercises from chapter 4", 4),
        ("Find content from chapter 1", 1),
        ("What's in chapter 5?", 5),
        ("Chapter 3 examples", 3)
    ]
    
    for query, expected_chapter in test_cases:
        print(f"\n📝 Query: {query}")
        print(f"🎯 Expected Chapter: {expected_chapter}")
        print("-" * 40)
        
        # Test the chapter search with the correct chapter number
        result = retriever.find_content_by_chapter_section(
            chapter_num=expected_chapter,
            content_types=['Exercise', 'Problem', 'Example'],
            limit=3
        )
        
        if result:
            print(f"✅ Found {len(result)} results:")
            for i, doc in enumerate(result[:2], 1):
                print(f"  {i}. Type: {doc.metadata.get('type', 'Unknown')}")
                print(f"     Chapter: {doc.metadata.get('chapter_number', 'N/A')}")
                print(f"     Module: {doc.metadata.get('module_id', 'Unknown')}")
                print(f"     Content: {doc.page_content[:100]}...")
        else:
            print("❌ No results found")
            
        # Also test the full educational content retrieval
        print(f"\n🔍 Testing full educational retrieval for: {query}")
        edu_result = retriever.retrieve_educational_content(
            query=query,
            auto_parse=True,
            limit=3
        )
        
        if edu_result.vector_documents:
            print(f"✅ Educational retrieval found {len(edu_result.vector_documents)} results")
            for i, doc in enumerate(edu_result.vector_documents[:1], 1):
                print(f"  {i}. Type: {doc.metadata.get('type', 'Unknown')}")
                print(f"     Chapter: {doc.metadata.get('chapter_number', 'N/A')}")
                print(f"     Strategy: {doc.metadata.get('search_strategy', 'unknown')}")
        else:
            print("❌ Educational retrieval found no results")

if __name__ == "__main__":
    test_chapter_search() 