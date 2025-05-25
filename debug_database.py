#!/usr/bin/env python3
"""Debug script to examine database structure and retrieval issues."""

from src.utils.env_manager import load_env_vars, EnvManager
from neo4j import GraphDatabase
import json

def main():
    load_env_vars()
    neo4j_creds = EnvManager.get_neo4j_credentials()

    driver = GraphDatabase.driver(neo4j_creds['uri'], auth=(neo4j_creds['username'], neo4j_creds['password']))

    with driver.session(database=neo4j_creds['database']) as session:
        # Check what relationships actually exist
        print("=== RELATIONSHIP TYPES IN DATABASE ===")
        result = session.run('''
        MATCH ()-[r]->()
        RETURN DISTINCT type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        LIMIT 10
        ''')
        
        for record in result:
            print(f'Relationship: {record["rel_type"]} - Count: {record["count"]}')
        
        # Check Content to Exercise relationships specifically
        print("\n=== CONTENT-EXERCISE RELATIONSHIPS ===")
        result = session.run('''
        MATCH (c:Content)-[r]-(e:Exercise)
        RETURN type(r) as rel_type, count(*) as count
        ''')
        
        total_relationships = 0
        for record in result:
            count = record["count"]
            total_relationships += count
            print(f'Content-Exercise {record["rel_type"]}: {count}')
        
        if total_relationships == 0:
            print("❌ NO RELATIONSHIPS between Content and Exercise nodes!")
        
        # Check if Exercise nodes have actual content
        print("\n=== EXERCISE NODES WITH ACTUAL CONTENT ===")
        result = session.run('''
        MATCH (n:Exercise)
        WHERE n.text_content IS NOT NULL AND n.text_content <> ""
        RETURN count(*) as count_with_content, 
               count(n) as total_count
        ''')
        
        for record in result:
            print(f'Exercises with content: {record["count_with_content"]} / {record["total_count"]}')
        
        # Check Problem nodes with content
        print("\n=== PROBLEM NODES WITH ACTUAL CONTENT ===")
        result = session.run('''
        MATCH (n:Problem)
        WHERE n.text_content IS NOT NULL AND n.text_content <> ""
        RETURN count(*) as count_with_content, 
               count(n) as total_count
        ''')
        
        for record in result:
            print(f'Problems with content: {record["count_with_content"]} / {record["total_count"]}')
        
        # Sample actual educational content
        print("\n=== SAMPLE EDUCATIONAL CONTENT ===")
        result = session.run('''
        MATCH (n)
        WHERE (n:Exercise OR n:Problem OR n:Solution)
        AND n.text_content IS NOT NULL 
        AND n.text_content <> ""
        AND size(n.text_content) > 50
        RETURN labels(n)[0] as type, n.text_content as content, n.id as id
        LIMIT 5
        ''')
        
        for record in result:
            print(f'Type: {record["type"]}')
            print(f'ID: {record["id"]}')
            print(f'Content: {record["content"][:150]}...')
            print('---')
        
        # Check what's in Content nodes that have embeddings
        print("\n=== CONTENT NODES WITH EMBEDDINGS ===")
        result = session.run('''
        MATCH (n:Content)
        WHERE n.fastRP_embedding IS NOT NULL
        RETURN n.text_content as content, n.id as id
        LIMIT 3
        ''')
        
        for record in result:
            print(f'ID: {record["id"]}')
            print(f'Content: {record["content"][:150]}...')
            print('---')
        
        # Check for chapter/section information
        print("\n=== CHAPTER/SECTION METADATA CHECK ===")
        result = session.run('''
        MATCH (n)
        WHERE n.chapter_number IS NOT NULL OR n.section_number IS NOT NULL
        RETURN labels(n) as labels, n.chapter_number as chapter, n.section_number as section, count(n) as count
        ''')
        
        chapter_data = list(result)
        if chapter_data:
            for record in chapter_data:
                print(f'Labels: {record["labels"]}, Chapter: {record["chapter"]}, Section: {record["section"]}, Count: {record["count"]}')
        else:
            print("❌ NO CHAPTER/SECTION METADATA FOUND!")
        
        # Check what properties actually exist on nodes
        print("\n=== NODE PROPERTIES ANALYSIS ===")
        result = session.run('''
        MATCH (n)
        WHERE n:Problem OR n:Exercise OR n:Solution OR n:Example
        RETURN labels(n)[0] as type, keys(n) as properties
        LIMIT 5
        ''')
        
        for record in result:
            print(f'Type: {record["type"]}, Properties: {record["properties"]}')
        
        # Check Module nodes for chapter information
        print("\n=== MODULE NODES ANALYSIS ===")
        result = session.run('''
        MATCH (m:Module)
        RETURN m.id as module_id, keys(m) as properties
        LIMIT 5
        ''')
        
        for record in result:
            print(f'Module: {record["module_id"]}, Properties: {record["properties"]}')
        
        # Check Chapter nodes if they exist
        print("\n=== CHAPTER NODES CHECK ===")
        result = session.run('''
        MATCH (c:Chapter)
        RETURN c.title as title, c.number as number, keys(c) as properties
        LIMIT 10
        ''')
        
        chapter_nodes = list(result)
        if chapter_nodes:
            for record in chapter_nodes:
                print(f'Chapter: {record["title"]}, Number: {record["number"]}, Properties: {record["properties"]}')
        else:
            print("❌ NO CHAPTER NODES FOUND!")

    driver.close()

if __name__ == "__main__":
    main() 