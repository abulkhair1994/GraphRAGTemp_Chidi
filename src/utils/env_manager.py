"""
Environment variable management for the GraphRAG project.
Handles loading and validating environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define default model names
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_ANTHROPIC_MODEL = "claude-3-sonnet-20240229"  # Default to Sonnet 3.5
DEFAULT_NEO4J_DATABASE = "neo4j"  # Default database in Neo4j

def load_env_vars():
    """Load environment variables from .env file."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env'
    
    # Load environment variables from .env file
    load_dotenv(dotenv_path=env_path)
    
    # Log success
    logger.info(f"Environment variables loaded from {env_path}")

class EnvManager:
    """Class to manage environment variables."""
    
    @staticmethod
    def get_openai_api_key():
        """Get the OpenAI API key."""
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API key not found in environment variables")
        return api_key
    
    @staticmethod
    def get_anthropic_api_key():
        """Get the Anthropic API key."""
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("Anthropic API key not found in environment variables")
        return api_key
    
    @staticmethod
    def get_anthropic_model():
        """Get the Anthropic model name."""
        model = os.environ.get('ANTHROPIC_MODEL', DEFAULT_ANTHROPIC_MODEL)
        
        # Validate if it's a supported Sonnet model
        if "claude-3-sonnet" not in model and "claude-3-opus" not in model:
            logger.warning(f"Using potentially unsupported Anthropic model: {model}")
            logger.warning(f"Defaulting to {DEFAULT_ANTHROPIC_MODEL}")
            return DEFAULT_ANTHROPIC_MODEL
        
        return model
    
    @staticmethod
    def get_openai_model():
        """Get the OpenAI model name."""
        return os.environ.get('OPENAI_MODEL', DEFAULT_OPENAI_MODEL)
    
    @staticmethod
    def get_neo4j_credentials():
        """Get Neo4j database credentials."""
        required = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
        missing = [var for var in required if not os.environ.get(var)]
        
        if missing:
            logger.warning(f"Missing Neo4j credentials: {', '.join(missing)}")
        
        return {
            'uri': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            'username': os.environ.get('NEO4J_USERNAME', 'neo4j'),
            'password': os.environ.get('NEO4J_PASSWORD', ''),
            'database': os.environ.get('NEO4J_DATABASE', DEFAULT_NEO4J_DATABASE),
        }
    
    @staticmethod
    def get_langsmith_credentials():
        """Get LangSmith credentials."""
        return {
            'api_key': os.environ.get('LANGCHAIN_API_KEY', ''),
            'project': os.environ.get('LANGCHAIN_PROJECT', 'graphrag_project'),
            'tracing': os.environ.get('LANGCHAIN_TRACING_V2', 'false').lower() == 'true',
        }
    
    @staticmethod
    def validate_env_vars():
        """Validate that all required environment variables are set."""
        required_vars = {
            'ANTHROPIC_API_KEY': EnvManager.get_anthropic_api_key(),
            'NEO4J_PASSWORD': EnvManager.get_neo4j_credentials().get('password'),
        }
        
        missing = [key for key, value in required_vars.items() if not value]
        
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            return False
        
        logger.info("All required environment variables are set")
        return True


# Example usage
if __name__ == "__main__":
    load_env_vars()
    
    # Print model names (masking API keys for security)
    print(f"Using Anthropic model: {EnvManager.get_anthropic_model()}")
    print(f"Using OpenAI model: {EnvManager.get_openai_model()}")
    
    # Validate environment variables
    EnvManager.validate_env_vars() 