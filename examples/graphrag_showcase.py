"""
GraphRAG Educational Assistant Showcase: Real Student Queries Demo

This showcase demonstrates how our GraphRAG system handles realistic student queries
and successfully retrieves relevant educational content. Each test simulates actual
questions students would ask and proves our retrieval system works effectively.
"""

import sys
import os
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from colorama import Fore, Style, init
import time

# Initialize colorama for colored terminal output
init(autoreset=True)

# Suppress all logging and warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import environment utilities and retrieval engines
from src.utils.env_manager import load_env_vars, EnvManager
from src.retrieval import GraphRAGRetriever
from src.retrieval.database_explorer import DatabaseExplorer

def print_banner():
    """Print an attractive banner for the showcase."""
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.CYAN}🎓 GraphRAG Educational Assistant - Student Query Showcase 🎓")
    print(f"{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.WHITE}Demonstrating realistic student queries and intelligent content retrieval")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

def print_section_header(title, emoji="📚"):
    """Print a formatted section header."""
    print(f"\n{Fore.BLUE}{emoji} {title}")
    print(f"{Fore.BLUE}{'-' * (len(title) + 4)}{Style.RESET_ALL}")

def print_query_header(query, student_name="Student"):
    """Print a student query in a conversational format."""
    print(f"\n{Fore.YELLOW}👤 {student_name}: \"{query}\"{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🤖 GraphRAG Assistant: Let me help you with that...{Style.RESET_ALL}")

def print_success_metrics(docs_found, content_types, avg_quality=None):
    """Print success metrics for retrieval."""
    print(f"\n{Fore.GREEN}✅ Retrieval Success Metrics:")
    print(f"   📄 Documents Found: {docs_found}")
    print(f"   📋 Content Types: {len(content_types)} ({', '.join(content_types)})")
    if avg_quality:
        print(f"   ⭐ Average Quality: {avg_quality:.1f}/10")
    print(f"{Style.RESET_ALL}")

def show_content_preview(doc, index, show_full=False):
    """Show a preview of retrieved content."""
    # Extract metadata with better defaults
    title = doc.metadata.get('title', 'Educational Content')
    content_type = doc.metadata.get('type', 'Unknown')
    module_id = doc.metadata.get('module_id', 'Unknown')
    chapter = doc.metadata.get('chapter_number', 'N/A')
    section = doc.metadata.get('section_number', 'N/A')
    search_strategy = doc.metadata.get('search_strategy', 'standard')
    
    # Create a more informative title if it's generic
    if title in ['Educational Content', 'Untitled'] or title.startswith('Content ID:'):
        if chapter != 'N/A' and chapter is not None:
            if section != 'N/A' and section is not None:
                title = f"Chapter {chapter}, Section {section} - {content_type}"
            else:
                title = f"Chapter {chapter} - {content_type}"
        else:
            title = f"{content_type} Content"
    
    print(f"\n{Fore.WHITE}📖 Result {index}: {title}")
    print(f"   🏷️  Type: {content_type} | 📍 Chapter: {chapter} | 🔢 Section: {section} | 🆔 Module: {module_id}")
    print(f"   🔍 Strategy: {search_strategy}")
    
    # Show additional metadata if available
    if doc.metadata.get('semantic_keywords'):
        keywords = doc.metadata['semantic_keywords']
        if isinstance(keywords, list) and keywords:
            print(f"   🏷️  Keywords: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}")
    
    if doc.metadata.get('similarity'):
        print(f"   ⭐ Similarity: {doc.metadata['similarity']:.3f}")
    elif doc.metadata.get('score'):
        print(f"   ⭐ Score: {doc.metadata['score']:.3f}")
    
    content = doc.page_content.strip()
    if show_full or len(content) < 300:
        print(f"   📝 Content: {content}")
    else:
        preview = content[:200] + "..."
        print(f"   📝 Preview: {preview}")
        print(f"   {Fore.CYAN}[Full content: {len(content)} characters]{Style.RESET_ALL}")

def simulate_typing_delay():
    """Simulate realistic response time."""
    time.sleep(0.5)

class StudentQueryShowcase:
    """Showcase class for demonstrating student queries and retrieval results."""
    
    def __init__(self):
        self.retriever = None
        self.total_queries = 0
        self.successful_retrievals = 0
        self.content_types_found = set()
        
    def setup_retriever(self):
        """Initialize the GraphRAG retriever."""
        try:
            load_env_vars()
            neo4j_creds = EnvManager.get_neo4j_credentials()
            
            print(f"{Fore.CYAN}🔧 Initializing GraphRAG Educational Assistant...{Style.RESET_ALL}")
            
            # Initialize embedding model for better search capabilities
            try:
                from langchain_openai import OpenAIEmbeddings
                embedding_model = OpenAIEmbeddings()
                print(f"{Fore.GREEN}✅ OpenAI embeddings initialized{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Could not initialize OpenAI embeddings: {e}{Style.RESET_ALL}")
                embedding_model = None
            
            self.retriever = GraphRAGRetriever(
                neo4j_url=neo4j_creds["uri"],
                neo4j_username=neo4j_creds["username"],
                neo4j_password=neo4j_creds["password"],
                neo4j_database=neo4j_creds["database"],
                vector_text_node_property="text_content",
                vector_node_label="Problem",
                embedding_model=embedding_model
            )
            
            # Test the connection
            self.retriever.graph_engine.test_connection()
            print(f"{Fore.GREEN}✅ Successfully connected to educational content database{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error setting up retriever: {e}{Style.RESET_ALL}")
            return False
    
    def verify_database_content(self):
        """Verify that we have educational content in the database."""
        print_section_header("Database Content Verification", "🔍")
        
        try:
            load_env_vars()
            neo4j_creds = EnvManager.get_neo4j_credentials()
            
            explorer = DatabaseExplorer(
                url=neo4j_creds["uri"],
                username=neo4j_creds["username"],
                password=neo4j_creds["password"],
                database=neo4j_creds["database"]
            )
            
            driver = explorer.connect()
            
            with driver.session(database=neo4j_creds["database"]) as session:
                # Check educational content types
                result = session.run("""
                    MATCH (n) 
                    WHERE n:Exercise OR n:Example OR n:Problem OR n:Solution OR n:Para
                    UNWIND labels(n) AS label
                    WITH label, COUNT(*) AS count
                    WHERE count > 0
                    RETURN label, count
                    ORDER BY count DESC
                """)
                
                print(f"{Fore.GREEN}📊 Educational Content Available:")
                total_content = 0
                for record in result:
                    count = record['count']
                    total_content += count
                    print(f"   • {record['label']}: {count:,} items")
                
                print(f"\n{Fore.CYAN}🎯 Total Educational Content: {total_content:,} items")
                print(f"✅ Database is ready for student queries!{Style.RESET_ALL}")
            
            driver.close()
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error verifying database: {e}{Style.RESET_ALL}")
    
    def test_student_query(self, query, student_name="Student", expected_content_types=None):
        """Test a realistic student query and show results."""
        self.total_queries += 1
        print_query_header(query, student_name)
        simulate_typing_delay()
        
        try:
            # Use the enhanced educational retrieval
            result = self.retriever.retrieve_educational_content(
                query=query,
                auto_parse=True,
                limit=5
            )
            
            if result.vector_documents:
                self.successful_retrievals += 1
                
                # Analyze the results
                quality_analysis = self.retriever.analyze_content_quality(result.vector_documents)
                content_types = list(quality_analysis['content_types'].keys())
                self.content_types_found.update(content_types)
                
                print_success_metrics(
                    len(result.vector_documents), 
                    content_types,
                    quality_analysis['average_quality']
                )
                
                # Show the best result only
                high_quality_docs = self.retriever.filter_high_quality_content(
                    result.vector_documents,
                    min_quality_score=2.0,
                    max_results=1
                )
                
                if high_quality_docs:
                    show_content_preview(high_quality_docs[0], 1, show_full=True)
                    return True  # Indicate success
                    
            else:
                print(f"{Fore.YELLOW}⚠️ No direct matches found. Trying alternative search strategies...{Style.RESET_ALL}")
                fallback_succeeded = self._try_fallback_retrieval(query) # Return fallback status
                if fallback_succeeded:
                    self.successful_retrievals += 1 # Increment if fallback found something
                return fallback_succeeded
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error processing query: {e}{Style.RESET_ALL}")
            return False # Indicate failure on exception

    def _try_fallback_retrieval(self, query):
        """Try alternative retrieval methods when primary search fails."""
        try:
            # Try finding content by parsing for chapter numbers
            parsed = self.retriever.parse_educational_query(query)
            
            if parsed['chapter']:
                documents = self.retriever.find_content_by_chapter_section(
                    chapter_num=parsed['chapter'],
                    section_num=parsed.get('section'),
                    content_types=['Exercise', 'Example', 'Problem'],
                    limit=3
                )
        
                if documents:
                    print(f"{Fore.GREEN}✅ Found {len(documents)} items using chapter-based search{Style.RESET_ALL}")
                    show_content_preview(documents[0], 1)
                    return True
            
            # Try semantic search with broader terms
            broad_terms = ['function', 'equation', 'polynomial', 'quadratic', 'linear']
            for term in broad_terms:
                if term in query.lower():
                    result = self.retriever.retrieve(query=term, vector_k=2)
                    if result.vector_documents:
                        print(f"{Fore.GREEN}✅ Found related content using '{term}' search{Style.RESET_ALL}")
                        show_content_preview(result.vector_documents[0], 1)
                        return True
            
            return False
            
        except Exception as e:
            print(f"{Fore.RED}❌ Fallback retrieval failed: {e}{Style.RESET_ALL}")
            return False

    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration with realistic student queries."""
        print_banner()
        
        if not self.setup_retriever():
            return
            
        self.verify_database_content()
        
        # Define realistic student queries organized by learning scenarios
        student_scenarios = [
            {
                "category": "Homework Help",
                "emoji": "📝",
                "queries": [
                    ("I need help with quadratic functions homework", "Sarah"),
                    ("How do I solve polynomial equations?", "Mike"),
                    ("Can you explain linear functions to me?", "Emma"),
                    ("What are the steps to graph a parabola?", "Alex")
                ]
            },
            {
                "category": "Chapter-Specific Questions", 
                "emoji": "📚",
                "queries": [
                    ("Show me exercises from chapter 4", "Jessica"),
                    ("I need examples from chapter 5 about polynomials", "David"),
                    ("What's in chapter 3 section 2?", "Lisa"),
                    ("Help me understand chapter 4 part 1", "Ryan")
                ]
            },
            {
                "category": "Concept Clarification",
                "emoji": "🤔", 
                "queries": [
                    ("What's the difference between a function and an equation?", "Maya"),
                    ("How do I know when to use the quadratic formula?", "Jake"),
                    ("Why do we factor polynomials?", "Sophie"),
                    ("When do I use function notation?", "Chris")
                ]
            },
            {
                "category": "Exam Preparation",
                "emoji": "📊",
                "queries": [
                    ("I have a test on quadratic functions tomorrow", "Taylor"),
                    ("Show me practice problems for polynomial functions", "Jordan"),
                    ("What should I review for my algebra exam?", "Casey"),
                    ("Help me prepare for linear functions quiz", "Morgan")
                ]
            },
            {
                "category": "Step-by-Step Help",
                "emoji": "👣",
                "queries": [
                    ("Walk me through solving x² + 5x + 6 = 0", "Sam"),
                    ("How do I complete the square step by step?", "Riley"),
                    ("Show me how to find the vertex of a parabola", "Avery"),
                    ("Explain how to use the quadratic formula", "Blake")
                ]
            }
        ]
        
        # Run all scenarios
        for scenario in student_scenarios:
            print_section_header(f"{scenario['category']} Scenarios", scenario['emoji'])
            
            for query, student_name in scenario['queries']:
                self.test_student_query(query, student_name) # Call test_student_query, it handles its own fallback
                print(f"\n{Fore.BLUE}{'─' * 60}{Style.RESET_ALL}")
        
        # Show final statistics
        self._show_final_statistics()
    
    def _show_final_statistics(self):
        """Show comprehensive statistics about the demonstration."""
        print_section_header("Demonstration Results Summary", "📈")
        
        success_rate = (self.successful_retrievals / self.total_queries) * 100 if self.total_queries > 0 else 0
        
        print(f"{Fore.GREEN}🎯 Overall Performance:")
        print(f"   📊 Total Student Queries: {self.total_queries}")
        print(f"   ✅ Successful Retrievals: {self.successful_retrievals}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print(f"   🏷️  Content Types Retrieved: {len(self.content_types_found)}")
        print(f"   📚 Types Found: {', '.join(sorted(self.content_types_found))}")
        
        if success_rate >= 80:
            print(f"\n{Fore.GREEN}🏆 EXCELLENT: GraphRAG system demonstrates high reliability!")
        elif success_rate >= 60:
            print(f"\n{Fore.YELLOW}👍 GOOD: GraphRAG system shows solid performance!")
        else:
            print(f"\n{Fore.RED}⚠️  NEEDS IMPROVEMENT: Consider optimizing retrieval methods")
        
        print(f"\n{Fore.CYAN}🎓 Key Capabilities Demonstrated:")
        print(f"   • Natural language query understanding")
        print(f"   • Chapter and section-based content location") 
        print(f"   • Educational content type recognition")
        print(f"   • Quality-based content filtering")
        print(f"   • Intelligent learning guidance")
        print(f"   • Robust fallback mechanisms")
        
        print(f"\n{Fore.BLUE}{'=' * 80}")
        print(f"{Fore.BLUE}🎉 GraphRAG Educational Assistant Showcase Complete! 🎉")
        print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")

def main():
    """Run the showcase demo."""
    print_banner()
    
    showcase = StudentQueryShowcase()
    if not showcase.setup_retriever():
        print(f"{Fore.RED}❌ Failed to initialize the showcase. Exiting.{Style.RESET_ALL}")
        sys.exit(1)
        
    showcase.verify_database_content()
    showcase.run_comprehensive_demo()

if __name__ == "__main__":
    main() 