# GraphRAGTemp_Chidi

A repository for Graph-based Retrieval Augmented Generation (GraphRAG) development.

## Description

This repository contains code for implementing and testing Graph-based Retrieval Augmented Generation using LangChain and Neo4j. The project implements a knowledge graph-based retrieval system to enhance context for large language models.

## Overview

GraphRAG is an extension of traditional RAG (Retrieval Augmented Generation) that leverages knowledge graphs to enhance the context and relationships between pieces of information. By using Neo4j as a graph database and LangChain for orchestration, this project aims to create more context-aware and accurate responses from Large Language Models.

## Key Components

- **Neo4j**: Graph database for storing and querying complex entity relationships
- **LangChain**: Framework for developing LLM applications
- **FastRP Embeddings**: For graph-based node embeddings
- **Vector Retrieval**: For semantic search capabilities
- **LLM Integration**: Connecting with advanced language models (Claude, GPT) for generation

## Workflow

1. Document ingestion and processing
2. Entity extraction and relationship mapping
3. Knowledge graph construction in Neo4j
4. Contextual retrieval through graph queries
5. Enhanced LLM prompting with graph context
6. Response generation and refinement

## Project Structure

```
GraphRAGTemp_Chidi/
│
├─ config/                 # Configuration settings
│   ├─ prompt_templates.yaml  # LLM prompt templates
│   ├─ model_config.yaml      # LLM model configurations
│   └─ __init__.py            # Module initialization
│
├─ src/                    # All production code
│   ├─ embeddings/         # Embedding generation and management
│   │   ├─ __init__.py             # Module exports
│   │   ├─ create_embeddings.py    # FastRP embedding creation
│   │   ├─ create_openai_embeddings.py # OpenAI embedding creation
│   │   └─ fix_embedding_mismatch.py  # Fix for dimension mismatch
│   ├─ retrieval/          # Graph and vector retrieval components
│   │   ├─ __init__.py           # Module exports
│   │   ├─ graphrag_retriever.py # Combined vector and graph retrieval
│   │   ├─ vector_retrieval.py   # Vector-based retrieval
│   │   ├─ graph_retrieval.py    # Graph-based retrieval
│   │   ├─ database_explorer.py  # Database schema exploration
│   │   └─ document_processor.py # Document processing utilities
│   ├─ utils/              # Shared utilities
│   │   ├─ __init__.py       # Module exports
│   │   ├─ env_manager.py    # Environment variable management
│   │   ├─ neo4j_utils.py    # Neo4j utility functions
│   │   └─ setup_neo4j.py    # Neo4j database setup
│   ├─ llm/                # Language-model clients (placeholder for future implementation)
│   ├─ prompt_engineering/ # Prompt assembly helpers (placeholder for future implementation)
│   └─ handlers/           # Error or event handlers (placeholder for future implementation)
│
├─ scripts/                # Executable entry points for tasks
│   ├─ add_text_content.py # Add text properties to nodes
│   ├─ env_example.py      # Environment setup example
│   ├─ explore_database.py # Database exploration utilities
│   ├─ fix_embeddings.py   # Fix embedding dimension mismatch
│   ├─ neo4j_connection.py # Test Neo4j connectivity
│   └─ run_neo4j_setup.py  # Setup Neo4j database
│
├─ examples/               # Example implementations
│   ├─ graphrag_retrieval_example.py # Example of GraphRAG retrieval
│   └─ env_example.py              # Environment setup example
│
├─ data/                   # Persisted artefacts
│   ├─ embeddings/         # Vector embeddings storage
│   ├─ outputs/            # Generated outputs
│   ├─ cache/              # Cache storage
│   └─ prompts/            # Prompt templates
│
└─ requirements.txt        # Project dependencies
```

## Configuration

The project uses YAML files in the `config/` directory to store configuration settings:

### Prompt Templates (`prompt_templates.yaml`)

Contains various prompt templates used by the system:
- System prompts for different assistant personas
- Task-specific prompts for entity extraction, relationship creation, and Cypher query generation

Example usage:
```python
import yaml

# Load prompt templates
with open("config/prompt_templates.yaml", "r") as f:
    prompt_templates = yaml.safe_load(f)

# Use a specific template
system_prompt = prompt_templates["system"]["graph_expert"]
entity_extraction_prompt = prompt_templates["graph_rag"]["extract_entities"]
```

### Model Configuration (`model_config.yaml`)

Contains settings for LLM providers:
- API configurations for OpenAI and Anthropic
- Model selection and parameters
- Rate limiting settings
- Caching configuration

Example usage:
```python
import yaml

# Load model configuration
with open("config/model_config.yaml", "r") as f:
    model_config = yaml.safe_load(f)

# Access configuration values
default_model = model_config["anthropic"]["models"]["default"]
temperature = model_config["anthropic"]["temperature"]
```

## Getting Started

### Prerequisites

- Python 3.9+
- Neo4j database (local installation or cloud instance)
- API keys for LLM services (Anthropic, OpenAI)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GraphRAGTemp_Chidi.git
   cd GraphRAGTemp_Chidi
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following variables:
   ```
   # Neo4j credentials
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   NEO4J_DATABASE=neo4j
   
   # API keys
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Test Neo4j connectivity:
   ```bash
   python scripts/neo4j_connection.py
   ```

## Usage Examples

- **Testing Neo4j Connection**: 
  ```bash
  python scripts/neo4j_connection.py
  ```
  This script tests your connection to Neo4j using credentials from your `.env` file.

- **Fix Embedding Dimension Mismatch**:
  ```bash
  python scripts/fix_embeddings.py
  ```
  This script resolves dimension mismatches between OpenAI embeddings (1536) and fastRP embeddings (512).

- **Add Text Content to Nodes**:
  ```bash
  python scripts/add_text_content.py
  ```
  This script adds a text_content property to nodes for text-based retrieval fallback.

- **Explore Database Structure**:
  ```bash
  python scripts/explore_database.py
  ```
  This script explores the Neo4j database schema to understand available node labels and relationship types.

- **Run GraphRAG Retrieval Example**:
  ```bash
  python examples/graphrag_retrieval_example.py
  ```
  This example demonstrates the GraphRAG retrieval functionality combining vector and graph-based approaches.

## Core Components

### GraphRAG Retriever

The core of this project is the `GraphRAGRetriever` class which combines:

1. **Vector Retrieval**: Using embeddings to find semantically similar content
2. **Graph Retrieval**: Using graph relationships to find contextually relevant information
3. **Hybrid Approach**: Combining both approaches for improved context and accuracy

Example usage:
```python
from src.retrieval import GraphRAGRetriever
from src.utils.env_manager import load_env_vars, EnvManager
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_env_vars()
neo4j_creds = EnvManager.get_neo4j_credentials()

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create GraphRAG retriever
retriever = GraphRAGRetriever(
    neo4j_url=neo4j_creds["uri"],
    neo4j_username=neo4j_creds["username"],
    neo4j_password=neo4j_creds["password"],
    neo4j_database=neo4j_creds["database"],
    embedding_model=embeddings
)

# Retrieve information
result = retriever.retrieve("What is GraphRAG?")
```

## Dependencies

- Python 3.9+
- Neo4j (v5.14.0+)
- LangChain (v0.0.267+)
- python-dotenv (v1.0.0+)
- LangChain OpenAI/Anthropic integrations
- Other dependencies listed in requirements.txt

## Future Improvements

- Implementation of LLM client modules in the `src/llm/` directory
- Development of prompt engineering utilities in the `src/prompt_engineering/` directory
- Addition of error and event handlers in the `src/handlers/` directory
- Integration with LangGraph for more complex, stateful AI workflows

## License

This project is licensed under the MIT License - see the LICENSE file for details. 