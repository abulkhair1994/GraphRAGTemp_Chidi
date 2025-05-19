# GraphRAGTemp_Chidi

A repository for Graph-based Retrieval Augmented Generation (GraphRAG) development.

## Description

This repository contains code for implementing and testing Graph-based Retrieval Augmented Generation using LangGraph and Neo4j. The project follows the workflow described in the [Neo4j GraphRAG tutorial](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/).

## Overview

GraphRAG is an extension of traditional RAG (Retrieval Augmented Generation) that leverages knowledge graphs to enhance the context and relationships between pieces of information. By using Neo4j as a graph database and LangGraph for orchestration, this project aims to create more context-aware and accurate responses from Large Language Models.

## Key Components

- **LangGraph**: For creating complex, stateful AI workflows
- **Neo4j**: Graph database for storing and querying complex entity relationships
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
│   └─ model_config.yaml      # LLM model configurations
│
├─ src/                    # All production code
│   ├─ llm/                # Language-model clients
│   ├─ prompt_engineering/ # Prompt assembly helpers
│   ├─ utils/              # Shared utilities
│   │   └─ env_manager.py  # Environment variable management
│   └─ handlers/           # Error or event handlers
│
├─ data/                   # Persisted artefacts
│   ├─ embeddings/         # Vector embeddings storage
│   ├─ outputs/            # Generated outputs
│   ├─ cache/              # Cache storage
│   └─ prompts/            # Prompt templates
│
├─ examples/               # Short runnable demos
│   ├─ neo4j_connection.py # Test Neo4j connectivity
│   └─ env_example.py      # Environment setup example
│
└─ notebooks/              # Experiments & analysis
```

## Configuration

The project uses YAML files in the `config/` directory to store configuration settings:

### Prompt Templates (`prompt_templates.yaml`)

Contains various prompt templates used by the system:
- System prompts for different assistant personas
- Task-specific prompts for entity extraction, relationship creation, and Cypher query generation

Example usage:
```python
from src.utils.config_loader import load_config

# Load prompt templates
prompt_templates = load_config("config/prompt_templates.yaml")

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
from src.utils.config_loader import load_config

# Load model configuration
model_config = load_config("config/model_config.yaml")

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
   python examples/neo4j_connection.py
   ```

## Usage Examples

- **Testing Neo4j Connection**: 
  ```bash
  python examples/neo4j_connection.py
  ```
  This example tests your connection to Neo4j using credentials from your `.env` file.

- **Environment Variables Example**:
  ```bash
  python examples/env_example.py
  ```
  This shows how to properly access environment variables in your code.

## Dependencies

- Python 3.9+
- LangGraph (v0.0.15+)
- Neo4j (v5.14.0+)
- LangChain (v0.0.267+)
- python-dotenv (v1.0.0+)
- LangChain OpenAI/Anthropic integrations
- Other dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details. 