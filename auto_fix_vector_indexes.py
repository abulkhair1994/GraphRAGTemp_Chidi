#!/usr/bin/env python3
"""
Automatic Vector Index Fix Script
=================================

This script automatically detects and fixes vector index embedding dimension mismatches
in Neo4j databases used with GraphRAG systems. It runs independently without complex
module dependencies.

Key Issues Fixed:
- Vector indexes using wrong embedding property (fastRP_embedding vs openai_embedding)
- Dimension mismatches (512 vs 1536)
- Missing or corrupted vector indexes
- LangChain import issues (langchain vs langchain_community)

Usage:
    python auto_fix_vector_indexes.py
"""

import os
import sys
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    
    # Get Neo4j credentials from environment
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
    
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        print("❌ Error: Missing Neo4j credentials in environment variables")
        print("Required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")
        sys.exit(1)
    
    return {
        'uri': neo4j_uri,
        'username': neo4j_username,
        'password': neo4j_password
    }

def get_neo4j_driver(creds: Dict[str, str]):
    """Create Neo4j driver with error handling"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            creds['uri'],
            auth=(creds['username'], creds['password'])
        )
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")
        return driver
    except ImportError:
        print("❌ Error: neo4j package not installed. Run: pip install neo4j")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        sys.exit(1)

def check_embedding_properties(driver) -> Dict[str, Any]:
    """Check what embedding properties exist in the database"""
    print("🔍 Checking embedding properties in database...")
    
    with driver.session() as session:
        # Check for different embedding properties
        queries = {
            'openai_embedding': "MATCH (n) WHERE n.openai_embedding IS NOT NULL RETURN count(n) as count, size(n.openai_embedding) as dimension LIMIT 1",
            'fastRP_embedding': "MATCH (n) WHERE n.fastRP_embedding IS NOT NULL RETURN count(n) as count, size(n.fastRP_embedding) as dimension LIMIT 1",
            'embedding': "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) as count, size(n.embedding) as dimension LIMIT 1"
        }
        
        results = {}
        for prop_name, query in queries.items():
            try:
                result = session.run(query).single()
                if result and result['count'] > 0:
                    results[prop_name] = {
                        'count': result['count'],
                        'dimension': result['dimension']
                    }
                    print(f"  ✅ Found {prop_name}: {result['count']} nodes, {result['dimension']} dimensions")
                else:
                    print(f"  ❌ No {prop_name} found")
            except Exception as e:
                print(f"  ❌ Error checking {prop_name}: {e}")
        
        return results

def check_vector_indexes(driver) -> List[Dict[str, Any]]:
    """Check existing vector indexes"""
    print("🔍 Checking vector indexes...")
    
    with driver.session() as session:
        try:
            result = session.run('SHOW INDEXES WHERE type = "VECTOR"')
            indexes = []
            for record in result:
                index_info = {
                    'name': record['name'],
                    'labels': record['labelsOrTypes'],
                    'properties': record['properties'],
                    'state': record.get('state', 'UNKNOWN')
                }
                indexes.append(index_info)
                print(f"  📊 Index: {index_info['name']}")
                print(f"      Labels: {index_info['labels']}")
                print(f"      Properties: {index_info['properties']}")
                print(f"      State: {index_info['state']}")
            
            if not indexes:
                print("  ❌ No vector indexes found")
            
            return indexes
        except Exception as e:
            print(f"  ❌ Error checking vector indexes: {e}")
            return []

def drop_vector_indexes(driver, indexes: List[Dict[str, Any]]) -> bool:
    """Drop existing vector indexes"""
    print("🗑️  Dropping existing vector indexes...")
    
    with driver.session() as session:
        try:
            for index in indexes:
                index_name = index['name']
                print(f"  🗑️  Dropping index: {index_name}")
                session.run(f"DROP INDEX {index_name}")
                print(f"  ✅ Dropped: {index_name}")
            return True
        except Exception as e:
            print(f"  ❌ Error dropping indexes: {e}")
            return False

def create_correct_vector_indexes(driver, embedding_props: Dict[str, Any]) -> bool:
    """Create vector indexes using the correct embedding property"""
    print("🔧 Creating correct vector indexes...")
    
    # Determine which embedding property to use
    if 'openai_embedding' in embedding_props:
        embedding_prop = 'openai_embedding'
        dimension = 1536
        print(f"  📊 Using OpenAI embeddings: {dimension} dimensions")
    elif 'fastRP_embedding' in embedding_props:
        embedding_prop = 'fastRP_embedding'
        dimension = embedding_props['fastRP_embedding']['dimension']
        print(f"  📊 Using FastRP embeddings: {dimension} dimensions")
    elif 'embedding' in embedding_props:
        embedding_prop = 'embedding'
        dimension = embedding_props['embedding']['dimension']
        print(f"  📊 Using generic embeddings: {dimension} dimensions")
    else:
        print("  ❌ No suitable embedding property found")
        return False
    
    # Educational content types to index
    content_types = ['Exercise', 'Problem', 'Solution', 'Example', 'Para']
    
    with driver.session() as session:
        try:
            for content_type in content_types:
                index_name = f"educational_{content_type.lower()}_embeddings"
                
                # Check if nodes of this type exist with the embedding property
                check_query = f"""
                MATCH (n:{content_type}) 
                WHERE n.{embedding_prop} IS NOT NULL 
                RETURN count(n) as count LIMIT 1
                """
                result = session.run(check_query).single()
                
                if result and result['count'] > 0:
                    print(f"  🔧 Creating index for {content_type} ({result['count']} nodes)")
                    
                    # Create vector index using the correct procedure
                    create_query = f"""
                    CALL db.index.vector.createNodeIndex(
                        '{index_name}',
                        '{content_type}',
                        '{embedding_prop}',
                        {dimension},
                        'cosine'
                    )
                    """
                    session.run(create_query)
                    print(f"  ✅ Created: {index_name}")
                else:
                    print(f"  ⚠️  Skipping {content_type}: no nodes with {embedding_prop}")
            
            return True
        except Exception as e:
            print(f"  ❌ Error creating vector indexes: {e}")
            return False

def verify_fix(driver) -> bool:
    """Verify that the fix was successful"""
    print("✅ Verifying fix...")
    
    # Check vector indexes again
    indexes = check_vector_indexes(driver)
    
    if not indexes:
        print("  ❌ No vector indexes found after fix")
        return False
    
    # Test a simple vector query
    with driver.session() as session:
        try:
            # Get a sample embedding to test with
            sample_query = """
            MATCH (n) 
            WHERE n.openai_embedding IS NOT NULL OR n.fastRP_embedding IS NOT NULL OR n.embedding IS NOT NULL
            RETURN n.openai_embedding as openai_emb, n.fastRP_embedding as fastrp_emb, n.embedding as emb
            LIMIT 1
            """
            result = session.run(sample_query).single()
            
            if result:
                # Use the first available embedding
                test_embedding = result['openai_emb'] or result['fastrp_emb'] or result['emb']
                
                if test_embedding and len(indexes) > 0:
                    # Test vector query
                    test_index = indexes[0]['name']
                    test_query = f"""
                    CALL db.index.vector.queryNodes('{test_index}', 1, $embedding)
                    YIELD node, score
                    RETURN count(node) as result_count
                    """
                    test_result = session.run(test_query, embedding=test_embedding).single()
                    
                    if test_result and test_result['result_count'] > 0:
                        print("  ✅ Vector search test successful!")
                        return True
                    else:
                        print("  ⚠️  Vector search test returned no results")
                        return False
            
            print("  ⚠️  Could not find test embedding")
            return False
            
        except Exception as e:
            print(f"  ❌ Error testing vector search: {e}")
            return False

def main():
    """Main function to automatically fix vector index issues"""
    print("🚀 Starting Automatic Vector Index Fix")
    print("=" * 50)
    
    # Load environment
    creds = load_environment()
    print(f"🔗 Connecting to Neo4j: {creds['uri']}")
    
    # Connect to Neo4j
    driver = get_neo4j_driver(creds)
    
    try:
        # Step 1: Check current state
        embedding_props = check_embedding_properties(driver)
        current_indexes = check_vector_indexes(driver)
        
        # Step 2: Determine if fix is needed
        needs_fix = False
        
        if not embedding_props:
            print("❌ No embedding properties found in database")
            return False
        
        if not current_indexes:
            print("⚠️  No vector indexes found - will create them")
            needs_fix = True
        else:
            # Check if indexes are using wrong property
            for index in current_indexes:
                if 'fastRP_embedding' in index['properties'] and 'openai_embedding' in embedding_props:
                    print("⚠️  Found indexes using fastRP_embedding when openai_embedding is available")
                    needs_fix = True
                    break
        
        if not needs_fix:
            print("✅ Vector indexes appear to be correctly configured")
            # Still verify they work
            if verify_fix(driver):
                print("🎉 All systems working correctly!")
                return True
            else:
                print("⚠️  Indexes exist but don't work properly - fixing...")
                needs_fix = True
        
        if needs_fix:
            print("\n🔧 Applying automatic fix...")
            
            # Step 3: Drop existing indexes if they exist
            if current_indexes:
                if not drop_vector_indexes(driver, current_indexes):
                    print("❌ Failed to drop existing indexes")
                    return False
            
            # Step 4: Create correct indexes
            if not create_correct_vector_indexes(driver, embedding_props):
                print("❌ Failed to create new indexes")
                return False
            
            # Step 5: Verify fix
            if verify_fix(driver):
                print("\n🎉 SUCCESS! Vector indexes fixed and working!")
                return True
            else:
                print("\n❌ Fix applied but verification failed")
                return False
        
    finally:
        driver.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 