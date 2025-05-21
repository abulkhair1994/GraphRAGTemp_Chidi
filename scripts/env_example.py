"""Example showing how to use the environment manager."""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our utils
from src.utils.env_manager import load_env_vars, EnvManager

def main():
    """Example of using the environment manager."""
    # Load environment variables
    load_env_vars()
    
    # Validate environment variables
    if not EnvManager.validate_env_vars():
        print("Warning: Some required environment variables are missing.")
    
    # Get the Anthropic model and API key
    model = EnvManager.get_anthropic_model()
    api_key = EnvManager.get_anthropic_api_key()
    
    # Print the model name
    print(f"\nAnthropic Configuration:")
    print(f"  Model: {model}")
    print(f"  API Key (masked): {'*' * 8}{api_key[-4:] if api_key else 'Not found'}")
    
    # Get Neo4j credentials
    neo4j_creds = EnvManager.get_neo4j_credentials()
    print("\nNeo4j Configuration:")
    print(f"  URI: {neo4j_creds['uri']}")
    print(f"  Username: {neo4j_creds['username']}")
    print(f"  Password: {'*' * 8 if neo4j_creds['password'] else 'Not set'}")
    print(f"  Database: {neo4j_creds['database']}")
    
    # Get LangSmith credentials
    langsmith = EnvManager.get_langsmith_credentials()
    print("\nLangSmith Configuration:")
    print(f"  Project: {langsmith['project']}")
    print(f"  Tracing Enabled: {langsmith['tracing']}")
    print(f"  API Key: {'Set' if langsmith['api_key'] else 'Not set'}")

if __name__ == "__main__":
    main() 