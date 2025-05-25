#!/usr/bin/env python3
"""
Create a comprehensive vector index for all educational content types.

This script creates a unified vector index that includes Problem, Exercise, Solution, 
Example, and Para nodes, enabling robust retrieval across all educational content types.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.env_manager import load_env_vars, EnvManager
from neo4j import GraphDatabase

def create_comprehensive_vector_index():
    """Create a comprehensive vector index for all educational content types."""
    print("🔧 Creating Comprehensive Educational Vector Index")
    print("=" * 60)
    
    # Load environment variables
    load_env_vars()
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        neo4j_creds["uri"],
        auth=(neo4j_creds["username"], neo4j_creds["password"])
    )
    
    try:
        with driver.session(database=neo4j_creds["database"]) as session:
            # First, check what educational content we have
            print("📊 Analyzing educational content in database...")
            
            content_analysis = session.run("""
                MATCH (n)
                WHERE n:Problem OR n:Exercise OR n:Solution OR n:Example OR n:Para
                AND n.fastRP_embedding IS NOT NULL
                AND n.text_content IS NOT NULL
                WITH labels(n)[0] as content_type, count(*) as count
                RETURN content_type, count
                ORDER BY count DESC
            """)
            
            total_nodes = 0
            for record in content_analysis:
                count = record["count"]
                total_nodes += count
                print(f"   • {record['content_type']}: {count:,} nodes with embeddings")
            
            print(f"\n🎯 Total Educational Content: {total_nodes:,} nodes")
            
            if total_nodes == 0:
                print("❌ No educational content with embeddings found!")
                return False
            
            # Drop existing vector index if it exists
            print("\n🗑️  Dropping existing vector index...")
            try:
                session.run("DROP INDEX content_embeddings IF EXISTS")
                print("✅ Existing index dropped")
            except Exception as e:
                print(f"ℹ️  No existing index to drop: {e}")
            
            # Create comprehensive vector index for all educational content
            print("\n🚀 Creating comprehensive educational vector index...")
            
            # Since Neo4j vector indexes work on single labels, we'll create multiple indexes
            # and update our retrieval logic to query across all of them
            educational_labels = ["Problem", "Exercise", "Solution", "Example", "Para"]
            
            for label in educational_labels:
                index_name = f"educational_{label.lower()}_embeddings"
                
                # Check if this label has nodes with embeddings
                check_result = session.run(f"""
                    MATCH (n:{label})
                    WHERE n.fastRP_embedding IS NOT NULL
                    RETURN count(n) as count
                """)
                
                count = check_result.single()["count"]
                if count > 0:
                    print(f"   📚 Creating index for {label} ({count:,} nodes)...")
                    
                    # Create vector index for this label using newer syntax
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (n:{label})
                        ON (n.fastRP_embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: 512,
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    print(f"   ✅ Index '{index_name}' created successfully")
                else:
                    print(f"   ⚠️  Skipping {label} - no nodes with embeddings")
            
            # Verify the indexes were created
            print("\n🔍 Verifying created indexes...")
            indexes = session.run("""
                SHOW INDEXES
                YIELD name, labelsOrTypes, properties, state, type
                WHERE type = 'VECTOR'
                RETURN name, labelsOrTypes, properties, state
            """)
            
            created_indexes = []
            for record in indexes:
                index_name = record["name"]
                labels = record["labelsOrTypes"]
                state = record["state"]
                
                if index_name.startswith("educational_"):
                    created_indexes.append(index_name)
                    print(f"   ✅ {index_name}: {labels} - {state}")
            
            print(f"\n🎉 Successfully created {len(created_indexes)} educational vector indexes!")
            
            # Create a unified query function
            print("\n📝 Creating unified query approach...")
            print("   The retrieval system will now query across all educational indexes")
            print("   for comprehensive content discovery.")
            
            return True
            
    except Exception as e:
        print(f"❌ Error creating comprehensive vector index: {e}")
        return False
    finally:
        driver.close()

def test_comprehensive_search():
    """Test the comprehensive search across all educational content types."""
    print("\n🧪 Testing Comprehensive Educational Search")
    print("=" * 50)
    
    # Load environment variables
    load_env_vars()
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        neo4j_creds["uri"],
        auth=(neo4j_creds["username"], neo4j_creds["password"])
    )
    
    try:
        with driver.session(database=neo4j_creds["database"]) as session:
            # Test query across all educational indexes
            educational_labels = ["Problem", "Exercise", "Solution", "Example", "Para"]
            
            print("🔍 Testing search across all educational content types...")
            
            # Get a sample embedding for testing
            sample_result = session.run("""
                MATCH (n:Problem)
                WHERE n.fastRP_embedding IS NOT NULL
                RETURN n.fastRP_embedding as embedding
                LIMIT 1
            """)
            
            sample_record = sample_result.single()
            if not sample_record:
                print("❌ No sample embedding found for testing")
                return False
            
            sample_embedding = sample_record["embedding"]
            
            # Test each index
            total_results = 0
            for label in educational_labels:
                index_name = f"educational_{label.lower()}_embeddings"
                
                try:
                    result = session.run(f"""
                        CALL db.index.vector.queryNodes(
                            '{index_name}',
                            3,
                            $embedding
                        ) YIELD node, score
                        RETURN count(*) as result_count
                    """, embedding=sample_embedding)
                    
                    count = result.single()["result_count"]
                    total_results += count
                    print(f"   📚 {label}: {count} results")
                    
                except Exception as e:
                    print(f"   ⚠️  {label}: Index not available - {e}")
            
            print(f"\n✅ Total results across all indexes: {total_results}")
            return total_results > 0
            
    except Exception as e:
        print(f"❌ Error testing comprehensive search: {e}")
        return False
    finally:
        driver.close()

if __name__ == "__main__":
    print("🎓 Comprehensive Educational Vector Index Setup")
    print("=" * 60)
    
    # Create the comprehensive vector index
    success = create_comprehensive_vector_index()
    
    if success:
        # Test the search functionality
        test_success = test_comprehensive_search()
        
        if test_success:
            print("\n🎉 SUCCESS: Comprehensive educational vector index is ready!")
            print("\n📋 Next Steps:")
            print("   1. Update your retrieval system to query across all educational indexes")
            print("   2. Test with various educational queries")
            print("   3. Monitor performance and adjust as needed")
        else:
            print("\n⚠️  Index created but testing failed. Check the configuration.")
    else:
        print("\n❌ Failed to create comprehensive vector index.") 