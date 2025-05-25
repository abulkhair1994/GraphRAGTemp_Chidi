#!/usr/bin/env python3
"""
Direct database query test to verify enhanced retrieval methods.
"""

from src.utils.env_manager import load_env_vars, EnvManager
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct_queries():
    """Test direct database queries to verify content and metadata."""
    load_env_vars()
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    driver = GraphDatabase.driver(
        neo4j_creds['uri'], 
        auth=(neo4j_creds['username'], neo4j_creds['password'])
    )
    
    try:
        with driver.session(database=neo4j_creds['database']) as session:
            
            print("🧪 Test 1: Chapter-based search")
            # Test the exact query from our enhanced retrieval
            result = session.run("""
            MATCH (n)
            WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example)
            AND n.chapter_number = $chapter_number
            AND n.text_content IS NOT NULL
            AND n.text_content <> ""
            AND size(n.text_content) > 20
            RETURN n.text_content as content,
                   n.id as id,
                   labels(n)[0] as type,
                   n.module_id as module_id,
                   n.chapter_number as chapter,
                   n.semantic_keywords as keywords
            ORDER BY size(n.text_content) DESC
            LIMIT 3
            """, chapter_number=4)
            
            count = 0
            for record in result:
                count += 1
                print(f"   ✅ Found {record['type']} from Chapter {record['chapter']}")
                print(f"      Content: {record['content'][:100]}...")
                print(f"      Keywords: {record['keywords']}")
                print()
            
            print(f"   Total found: {count}")
            
            print("\n🧪 Test 2: Semantic keyword search")
            # Test semantic keyword search
            result = session.run("""
            MATCH (n)
            WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example)
            AND n.semantic_keywords IS NOT NULL
            AND n.text_content IS NOT NULL
            AND n.text_content <> ""
            AND ANY(k IN n.semantic_keywords WHERE toLower(k) CONTAINS toLower('quadratic'))
            RETURN n.text_content as content,
                   n.id as id,
                   labels(n)[0] as type,
                   n.chapter_number as chapter,
                   n.semantic_keywords as keywords
            LIMIT 3
            """)
            
            count = 0
            for record in result:
                count += 1
                print(f"   ✅ Found {record['type']} with quadratic keywords")
                print(f"      Content: {record['content'][:100]}...")
                print(f"      Keywords: {record['keywords']}")
                print()
            
            print(f"   Total found: {count}")
            
            print("\n🧪 Test 3: Direct text search")
            # Test direct text search
            result = session.run("""
            MATCH (n)
            WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example)
            AND n.text_content IS NOT NULL 
            AND n.text_content <> ""
            AND size(n.text_content) > 20
            AND toLower(n.text_content) CONTAINS 'quadratic'
            RETURN n.text_content as content,
                   n.id as id,
                   labels(n)[0] as type,
                   n.chapter_number as chapter
            LIMIT 3
            """)
            
            count = 0
            for record in result:
                count += 1
                print(f"   ✅ Found {record['type']} with 'quadratic' in text")
                print(f"      Content: {record['content'][:100]}...")
                print(f"      Chapter: {record['chapter']}")
                print()
            
            print(f"   Total found: {count}")
            
    except Exception as e:
        logger.error(f"❌ Error during testing: {e}")
        raise
    finally:
        driver.close()

if __name__ == "__main__":
    test_direct_queries() 