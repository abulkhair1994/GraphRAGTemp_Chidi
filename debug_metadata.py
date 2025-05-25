from src.utils.env_manager import load_env_vars, EnvManager
from neo4j import GraphDatabase
import json

load_env_vars()
creds = EnvManager.get_neo4j_credentials()

driver = GraphDatabase.driver(creds['uri'], auth=(creds['username'], creds['password']))

with driver.session(database=creds['database']) as session:
    # Check what properties are available on nodes
    result = session.run('''
    MATCH (n)
    WHERE n:Exercise OR n:Example OR n:Problem OR n:Solution OR n:Para
    WITH labels(n)[0] as nodeType, keys(n) as props
    RETURN nodeType, props
    LIMIT 5
    ''')
    
    print('Node properties by type:')
    for record in result:
        print(f'{record["nodeType"]}: {record["props"]}')
    
    # Check specific metadata fields
    result = session.run('''
    MATCH (n)
    WHERE n:Exercise OR n:Example OR n:Problem OR n:Solution OR n:Para
    RETURN labels(n)[0] as type, 
           n.id as id,
           n.module_id as module_id,
           n.chapter_number as chapter_number,
           n.section_number as section_number,
           n.title as title,
           size(n.text_content) as content_length
    LIMIT 10
    ''')
    
    print('\nSample metadata:')
    for record in result:
        print(f'Type: {record["type"]}, ID: {record["id"]}, Module: {record["module_id"]}, Chapter: {record["chapter_number"]}, Section: {record["section_number"]}, Title: {record["title"]}, Content Length: {record["content_length"]}')

driver.close() 