"""
Document Processor for GraphRAG.

This module provides functionality to extract entities and relationships from documents
and add them to the Neo4j graph database to support graph-based retrieval.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set

from langchain.schema import Document
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents to extract entities and relationships for graph-based retrieval."""
    
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ):
        """Initialize the Document Processor.
        
        Args:
            url: Neo4j connection URL
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        self.url = url
        self.username = username
        self.password = password
        self.database = database
        
    def connect(self):
        """Connect to Neo4j database."""
        return GraphDatabase.driver(
            self.url, 
            auth=(self.username, self.password)
        )
    
    def close(self):
        """Close the connection."""
        pass
    
    def process_documents(self, documents: List[Document]) -> List[str]:
        """Process documents to extract entities and relationships.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of document IDs
        """
        document_ids = []
        driver = self.connect()
        
        try:
            for doc in documents:
                # Extract content and metadata
                content = doc.page_content
                metadata = doc.metadata
                source = metadata.get("source", "unknown")
                
                # Process the document
                doc_id = self._process_single_document(driver, content, source, metadata)
                document_ids.append(doc_id)
                
        finally:
            driver.close()
            
        return document_ids
    
    def _process_single_document(
        self, 
        driver, 
        content: str, 
        source: str, 
        metadata: Dict[str, Any]
    ) -> str:
        """Process a single document.
        
        Args:
            driver: Neo4j driver
            content: Document content
            source: Document source
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        # Extract entities from the document
        entities = self._extract_entities(content)
        
        # Extract relationships between entities
        relationships = self._extract_relationships(entities, content)
        
        # Create document node and entity nodes in Neo4j
        doc_id = self._create_document_graph(driver, content, source, metadata, entities, relationships)
        
        return doc_id
    
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities from document content.
        
        Args:
            content: Document content
            
        Returns:
            Dictionary mapping entity types to lists of entity names
        """
        # Simple entity extraction using patterns
        entities = {
            "Chapter": [],
            "Course": [],
            "Concept": []
        }
        
        # Extract Course entities
        course_pattern = r'(?:course|class|program)(?:\s+called|\s+titled|\s+named)?\s+[\'"](.+?)[\'"]'
        course_matches = re.findall(course_pattern, content, re.IGNORECASE)
        
        # Also match courses with specific patterns
        alt_course_pattern = r'(?:The|A)\s+[\'"](.+?)[\'"](?:\s+course|class)'
        alt_course_matches = re.findall(alt_course_pattern, content, re.IGNORECASE)
        
        entities["Course"] = list(set(course_matches + alt_course_matches))
        
        # Extract Chapter entities
        chapter_pattern = r'Chapter\s+(\d+)'
        chapter_matches = re.findall(chapter_pattern, content)
        entities["Chapter"] = [f"Chapter {num}" for num in chapter_matches]
        
        # Extract Concept entities
        concept_patterns = [
            r'(?:concept|topic)s?\s+of\s+[\'"]?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)[\'"]?',
            r'[\'"]?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)[\'"]?\s+(?:concept|topic)s?'
        ]
        
        # Additional specific concepts we want to capture
        specific_concepts = ["Function", "Equation", "Variable", "Algorithm", "Model"]
        
        concept_matches = []
        for pattern in concept_patterns:
            matches = re.findall(pattern, content)
            concept_matches.extend(matches)
        
        # Add specific concepts if they appear in the content
        for concept in specific_concepts:
            if re.search(r'\b' + concept + r's?\b', content):
                concept_matches.append(concept)
                
        entities["Concept"] = list(set(concept_matches))
        
        return entities
    
    def _extract_relationships(
        self, 
        entities: Dict[str, List[str]], 
        content: str
    ) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities.
        
        Args:
            entities: Dictionary of entity types and names
            content: Document content
            
        Returns:
            List of (entity1, relationship_type, entity2) tuples
        """
        relationships = []
        
        # Flatten entities dict for easier processing
        all_entities = []
        for entity_type, entity_names in entities.items():
            for name in entity_names:
                all_entities.append((entity_type, name))
        
        # Extract Course-Chapter relationships
        for course_type, course_name in [(t, n) for t, n in all_entities if t == "Course"]:
            for chapter_type, chapter_name in [(t, n) for t, n in all_entities if t == "Chapter"]:
                # Check if course and chapter appear close to each other in the text
                course_pos = content.lower().find(course_name.lower())
                chapter_pos = content.lower().find(chapter_name.lower())
                
                if course_pos >= 0 and chapter_pos >= 0:
                    # If they appear within reasonable proximity, create a relationship
                    if abs(course_pos - chapter_pos) < 200:  # Arbitrary proximity threshold
                        relationships.append((course_name, "CONTAINS", chapter_name))
        
        # Extract Chapter-Concept relationships
        for chapter_type, chapter_name in [(t, n) for t, n in all_entities if t == "Chapter"]:
            for concept_type, concept_name in [(t, n) for t, n in all_entities if t == "Concept"]:
                chapter_pos = content.lower().find(chapter_name.lower())
                concept_pos = content.lower().find(concept_name.lower())
                
                if chapter_pos >= 0 and concept_pos >= 0:
                    if abs(chapter_pos - concept_pos) < 150:  # Smaller threshold for better precision
                        relationships.append((chapter_name, "COVERS", concept_name))
        
        # Extract Concept-Concept relationships
        concept_entities = [(t, n) for t, n in all_entities if t == "Concept"]
        for i, (type1, name1) in enumerate(concept_entities):
            for type2, name2 in concept_entities[i+1:]:
                # Check for relationship patterns
                pattern = f"{name1}\\s+(?:and|or)\\s+{name2}"
                if re.search(pattern, content, re.IGNORECASE):
                    relationships.append((name1, "RELATED_TO", name2))
                
                # Check for specific relationship between Function and Equation
                if (name1 == "Function" and name2 == "Equation") or (name1 == "Equation" and name2 == "Function"):
                    if re.search(r"functions?.+equations?|equations?.+functions?", content, re.IGNORECASE):
                        relationships.append((name1, "RELATED_TO", name2))
        
        return relationships
    
    def _create_document_graph(
        self,
        driver,
        content: str,
        source: str,
        metadata: Dict[str, Any],
        entities: Dict[str, List[str]],
        relationships: List[Tuple[str, str, str]]
    ) -> str:
        """Create document and entity nodes in Neo4j.
        
        Args:
            driver: Neo4j driver
            content: Document content
            source: Document source
            metadata: Document metadata
            entities: Dictionary of entity types and names
            relationships: List of (entity1, relationship_type, entity2) tuples
            
        Returns:
            Document ID
        """
        # Generate a simple document ID
        import hashlib
        doc_id = hashlib.md5(content.encode()).hexdigest()
        
        with driver.session(database=self.database) as session:
            # Create document node
            session.run(
                """
                CREATE (d:DocumentNode {id: $id, content: $content, source: $source})
                RETURN d
                """,
                id=doc_id, content=content, source=source
            )
            
            # Create entity nodes and relationships to document
            for entity_type, entity_names in entities.items():
                for entity_name in entity_names:
                    session.run(
                        f"""
                        MERGE (e:{entity_type} {{name: $name}})
                        WITH e
                        MATCH (d:DocumentNode {{id: $doc_id}})
                        MERGE (d)-[:MENTIONS]->(e)
                        RETURN e, d
                        """,
                        name=entity_name, doc_id=doc_id
                    )
            
            # Create relationships between entities
            for entity1, rel_type, entity2 in relationships:
                # Note: This is simplified and assumes entity types are known
                # In a production system, you would need to handle entity types more carefully
                session.run(
                    f"""
                    MATCH (e1) WHERE e1.name = $name1
                    MATCH (e2) WHERE e2.name = $name2
                    MERGE (e1)-[:{rel_type}]->(e2)
                    RETURN e1, e2
                    """,
                    name1=entity1, name2=entity2
                )
        
        return doc_id 