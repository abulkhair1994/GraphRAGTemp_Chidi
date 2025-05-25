#!/usr/bin/env python3
"""
SAFE Database Structure Fix Script

This script ONLY ADDS missing chapter/section metadata to existing nodes.
It NEVER removes or deletes anything from the database.

Based on Neo4j graph refactoring best practices:
https://neo4j.com/docs/getting-started/current/data-modeling/modeling-tips/
"""

from src.utils.env_manager import load_env_vars, EnvManager
from neo4j import GraphDatabase
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to safely add chapter/section metadata."""
    logger.info("🔧 Starting SAFE database structure enhancement...")
    logger.info("⚠️  This script ONLY ADDS properties - NEVER removes anything!")
    
    load_env_vars()
    neo4j_creds = EnvManager.get_neo4j_credentials()
    
    driver = GraphDatabase.driver(
        neo4j_creds['uri'], 
        auth=(neo4j_creds['username'], neo4j_creds['password'])
    )
    
    try:
        with driver.session(database=neo4j_creds['database']) as session:
            
            # Step 1: Add chapter numbers to Chapter nodes (SAFE - only adds missing property)
            logger.info("📚 Step 1: Adding chapter numbers to Chapter nodes...")
            add_chapter_numbers(session)
            
            # Step 2: Create chapter-to-content mapping (SAFE - only adds properties)
            logger.info("🔗 Step 2: Adding chapter metadata to educational content...")
            add_chapter_metadata_to_content(session)
            
            # Step 3: Add semantic keyword mappings (SAFE - only adds properties)
            logger.info("🏷️  Step 3: Adding semantic keyword tags...")
            add_semantic_keywords(session)
            
            # Step 4: Verify the changes (READ-ONLY)
            logger.info("✅ Step 4: Verifying changes...")
            verify_changes(session)
            
    except Exception as e:
        logger.error(f"❌ Error during database enhancement: {e}")
        raise
    finally:
        driver.close()
        logger.info("🔒 Database connection closed safely")

def add_chapter_numbers(session):
    """SAFELY add chapter numbers to Chapter nodes - ONLY ADDS missing property."""
    
    # OpenStax College Algebra chapter mapping
    chapter_mapping = {
        "Prerequisites": 0,
        "Equations and Inequalities": 1, 
        "Functions": 2,
        "Linear Functions": 3,
        "Polynomial and Rational Functions": 4,
        "Exponential and Logarithmic Functions": 5,
        "Systems of Equations and Inequalities": 6,
        "Analytic Geometry": 7,
        "Sequences, Probability, and Counting Theory": 8
    }
    
    for title, number in chapter_mapping.items():
        query = """
        MATCH (c:Chapter {title: $title})
        WHERE c.chapter_number IS NULL
        SET c.chapter_number = $number
        RETURN count(c) as updated_count
        """
        
        result = session.run(query, title=title, number=number)
        count = result.single()["updated_count"]
        
        if count > 0:
            logger.info(f"  ✅ Added chapter_number {number} to '{title}' ({count} nodes)")
        else:
            logger.info(f"  ℹ️  Chapter '{title}' already has chapter_number or doesn't exist")

def add_chapter_metadata_to_content(session):
    """SAFELY add chapter metadata to educational content - ONLY ADDS missing properties."""
    
    # Map modules to chapters based on module IDs (OpenStax structure)
    module_chapter_mapping = {
        # Chapter 0: Prerequisites  
        "m86505": 0, "m51239": 0, "m51240": 0, "m51241": 0,
        
        # Chapter 1: Equations and Inequalities
        "m51242": 1, "m51243": 1, "m51244": 1, "m51245": 1,
        
        # Chapter 2: Functions
        "m51246": 2, "m51247": 2, "m51248": 2, "m51249": 2,
        
        # Chapter 3: Linear Functions  
        "m51250": 3, "m51251": 3, "m51252": 3, "m51253": 3,
        
        # Chapter 4: Polynomial and Rational Functions
        "m51254": 4, "m51255": 4, "m51256": 4, "m51257": 4,
        "m51258": 4, "m51259": 4, "m51260": 4, "m51261": 4,
        "m51262": 4, "m51263": 4, "m51264": 4, "m51265": 4,
        "m51266": 4, "m51267": 4, "m51268": 4, "m51269": 4,
        "m51270": 4, "m51271": 4, "m51272": 4, "m51273": 4,
        "m51274": 4, "m51275": 4, "m51276": 4, "m51277": 4,
        "m51278": 4, "m51279": 4, "m51280": 4,
        
        # Chapter 5: Exponential and Logarithmic Functions
        "m49361": 5, "m49362": 5, "m49363": 5, "m49364": 5,
        "m49365": 5, "m49366": 5, "m49367": 5,
        
        # Chapter 6: Systems of Equations and Inequalities
        "m49418": 6, "m49419": 6, "m49420": 6, "m49421": 6,
        "m49422": 6, "m49423": 6, "m49424": 6,
        
        # Chapter 7: Analytic Geometry
        "m49425": 7, "m49426": 7, "m49427": 7, "m49428": 7,
        "m49429": 7, "m49430": 7, "m49431": 7, "m49432": 7,
        "m49433": 7, "m49434": 7, "m49435": 7, "m49436": 7,
        "m49437": 7,
        
        # Chapter 8: Sequences, Probability, and Counting Theory
        "m49438": 8, "m49439": 8, "m49440": 8, "m49441": 8,
        "m49442": 8, "m49443": 8, "m49444": 8, "m49445": 8,
        "m49446": 8
    }
    
    total_updated = 0
    
    for module_id, chapter_num in module_chapter_mapping.items():
        # Update educational content nodes that belong to this module
        query = """
        MATCH (n)
        WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example OR n:Para)
        AND n.module_id = $module_id
        AND n.chapter_number IS NULL
        SET n.chapter_number = $chapter_number
        RETURN count(n) as updated_count
        """
        
        result = session.run(query, module_id=module_id, chapter_number=chapter_num)
        count = result.single()["updated_count"]
        total_updated += count
        
        if count > 0:
            logger.info(f"  ✅ Added chapter_number {chapter_num} to {count} nodes in module {module_id}")
    
    logger.info(f"📊 Total educational content nodes updated: {total_updated}")

def add_semantic_keywords(session):
    """SAFELY add semantic keyword tags - ONLY ADDS missing properties."""
    
    # Keyword to chapter mapping for better semantic search
    keyword_mappings = {
        # Chapter 4: Polynomial and Rational Functions
        4: ["quadratic", "polynomial", "parabola", "factoring", "vertex", "axis of symmetry", 
            "rational function", "asymptote", "degree", "leading coefficient"],
            
        # Chapter 3: Linear Functions
        3: ["linear", "slope", "y-intercept", "point-slope", "slope-intercept", 
            "parallel", "perpendicular", "rate of change"],
            
        # Chapter 5: Exponential and Logarithmic Functions  
        5: ["exponential", "logarithm", "log", "natural log", "e", "growth", "decay"],
        
        # Chapter 2: Functions
        2: ["function", "domain", "range", "input", "output", "f(x)", "composition"],
        
        # Chapter 1: Equations and Inequalities
        1: ["equation", "inequality", "solve", "solution", "variable", "absolute value"],
        
        # Chapter 0: Prerequisites
        0: ["real numbers", "rational", "irrational", "integer", "fraction", "decimal"]
    }
    
    for chapter_num, keywords in keyword_mappings.items():
        # Add semantic tags to content in this chapter
        query = """
        MATCH (n)
        WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example)
        AND n.chapter_number = $chapter_number
        AND n.semantic_keywords IS NULL
        SET n.semantic_keywords = $keywords
        RETURN count(n) as updated_count
        """
        
        result = session.run(query, chapter_number=chapter_num, keywords=keywords)
        count = result.single()["updated_count"]
        
        if count > 0:
            logger.info(f"  ✅ Added semantic keywords to {count} nodes in Chapter {chapter_num}")

def verify_changes(session):
    """Verify the changes made (READ-ONLY verification)."""
    
    # Check Chapter nodes with numbers
    result = session.run("""
    MATCH (c:Chapter)
    WHERE c.chapter_number IS NOT NULL
    RETURN count(c) as chapters_with_numbers
    """)
    chapters_count = result.single()["chapters_with_numbers"]
    logger.info(f"📊 Chapters with numbers: {chapters_count}")
    
    # Check educational content with chapter metadata
    result = session.run("""
    MATCH (n)
    WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example)
    AND n.chapter_number IS NOT NULL
    RETURN count(n) as content_with_chapters
    """)
    content_count = result.single()["content_with_chapters"]
    logger.info(f"📊 Educational content with chapter metadata: {content_count}")
    
    # Check semantic keywords
    result = session.run("""
    MATCH (n)
    WHERE (n:Problem OR n:Exercise OR n:Solution OR n:Example)
    AND n.semantic_keywords IS NOT NULL
    RETURN count(n) as content_with_keywords
    """)
    keywords_count = result.single()["content_with_keywords"]
    logger.info(f"📊 Educational content with semantic keywords: {keywords_count}")
    
    logger.info("✅ Database structure enhancement completed successfully!")
    logger.info("🎯 Your GraphRAG system now has proper chapter/section organization!")

if __name__ == "__main__":
    main() 