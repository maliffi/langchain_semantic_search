# LangChain Semantic Search

This project has been built staring from langchain guide available at [link](https://python.langchain.com/docs/tutorials/retrievers/) and familiarizes with LangChain's document loader, embedding, and vector store abstractions.
These abstractions are designed to support retrieval of data-- from (vector) databases and other sources-- for integration with LLM workflows. They are important for applications that fetch data to be reasoned over as part of model inference, as in the case of retrieval-augmented generation, or RAG (see our RAG tutorial here).
A powerful semantic search implementation built with LangChain that enables efficient document retrieval based on meaning rather than exact keyword matching.

This project implements a complete semantic search pipeline using LangChain's abstractions:

1. **Document Loading**: Ingests documents from PDF files
2. **Text Chunking**: Splits documents into manageable chunks for more effective retrieval
3. **Embedding Generation**: Converts text chunks into vector embeddings using Ollama models
4. **Vector Storage**: Stores embeddings in Qdrant vector database for efficient similarity search
5. **Semantic Retrieval**: Enables querying documents based on semantic meaning rather than keyword matching

This implementation serves as an excellent foundation for building Retrieval Augmented Generation (RAG) applications, where retrieved context can be fed to LLMs to provide more accurate and contextually relevant responses.

## Project Structure

```
langchain_semantic_search/
├── data/               # Directory for source documents (PDFs)
├── docker-compose.yml  # Docker configuration for Qdrant
├── logs/               # Log files directory
├── src/                # Source code
│   ├── __init__.py
│   ├── main.py         # Application entry point
│   ├── config.py       # Configuration settings
│   └── utils/          # Utility modules
│       ├── __init__.py
│       └── logger.py   # Logging configuration
├── .env                # Environment variables
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore file
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## Dependencies

This project relies on the following key dependencies:

### Core
- **langchain**: Base framework for building language model applications
- **langchain-community**: Community integrations for LangChain
- **langchain-ollama**: Integration with Ollama for local embeddings
- **langchain-qdrant**: Integration with Qdrant vector database
- **pypdf**: PDF document processing

### Infrastructure
- **python-dotenv**: Environment variable management
- **loguru**: Advanced logging

### Development
- **pytest**: Testing framework
- **black**: Code formatter
- **isort**: Import formatter
- **pre-commit**: Git hook manager

## Setup and Running Locally

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for running Qdrant)
- Ollama installed locally (for embeddings)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/maliffi/langchain_semantic_search.git
   cd langchain_semantic_search
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables by creating/modifying the `.env` file (see `.env.example` for a template):
   ```
   # Environment (development/production)
   ENV=development

   # Vector DB Connection
   DB_HOST=localhost
   DB_PORT=6333

   # Application Settings
   USE_SAMPLE_DOCS=true  # Set to false to use your own documents

   # Embedding Model (requires Ollama to be running)
   # Recommended model for best performance
   EMBEDDING_MODEL=mxbai-embed-large:v1
   ```

### Running the Vector Database

The project uses Qdrant as its vector database. Start it using Docker:

```bash
docker-compose up -d
```

This will start a Qdrant instance accessible at http://localhost:6333.

### Start Ollama (for embeddings)

Make sure Ollama is running with the embedding model specified in your `.env` file:

```bash
ollama pull nomic-embed-text  # Pull the model first if you don't have it
ollama serve  # Start the Ollama server
```

### Adding Your Documents

Place your PDF documents in the `data/` directory to process them, or set `USE_SAMPLE_DOCS=true` in your `.env` file to use the built-in sample documents.

### Running the Application

```bash
python -m src.main
```

Command-line options:
- `--use-sample-docs`: Use sample documents instead of loading from PDF files

The application will:
1. Load documents (either samples or from PDF files)
2. Process and chunk the documents
3. Generate embeddings
4. Store them in the Qdrant vector database
5. Perform a sample query and display the results

## Development

This project uses several development tools to maintain code quality:

- **Black**: Code formatter that enforces a consistent style
- **isort**: Import statement formatter
- **pre-commit**: Git hook manager that runs checks before commits

To set up the pre-commit hooks:
```bash
pre-commit install
```

## Testing

Run the tests using pytest:
```bash
pytest
```

## Logging

The application uses Loguru for logging. Logs are stored in the `logs/` directory.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
