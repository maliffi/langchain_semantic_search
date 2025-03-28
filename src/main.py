"""
Main application module.
"""
import argparse
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, chain
from langchain_core.vectorstores import VectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.config import Config
from src.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = get_logger("main")

global_vector_store: VectorStore = None


def get_vector_store(embeddings: OllamaEmbeddings, collection_name: str) -> VectorStore:
    """
    Initialize and return a vector store client.

    Returns:
        VectorStore: Configured vector store client.
    """
    client = QdrantClient(host=Config.DB_HOST, port=Config.DB_PORT)

    # Check if collection exists, if not create it
    try:
        client.get_collection(collection_name=collection_name)
        logger.info(f"Collection {collection_name} exists")
    except Exception:
        # Collection doesn't exist, create it
        logger.info(f"Collection {collection_name} doesn't exist. Creating it...")
        # Get the embedding dimension from the embeddings model
        vectors_config = VectorParams(
            size=Config.VECTOR_DIMENSION, distance=Distance.COSINE
        )
        client.create_collection(
            collection_name=collection_name, vectors_config=vectors_config
        )
        logger.info(f"Created collection {collection_name}")

    vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embeddings
    )
    set_vector_store(vector_store)
    return vector_store


def load_documents() -> list[Document]:
    """
    Load documents from the PDF files in the directory specified by Config.INPUT_DOC_FOLDER.
    Returns:
        list[Document]: List of documents loaded from the PDF files.
    """
    file_path = Config.INPUT_DOC_FOLDER
    if not os.path.exists(file_path):
        logger.error(f"Document path {file_path} does not exist")
        return []

    documents = []
    try:
        for file in os.listdir(file_path):
            for file_type in Config.DOC_FILE_TYPES:
                if file.endswith(file_type):
                    full_path = os.path.join(file_path, file)
                    loader = PyPDFLoader(full_path)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {full_path}")

        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []


def get_sample_documents() -> list[Document]:
    """
    Returns a list of sample documents.
    Returns:
        list[Document]: List of sample documents.
    """
    # LangChain implements a Document abstraction, which is intended to represent a unit of text and associated metadata.
    # It has three attributes:
    # - page_content: a string representing the content
    # - metadata: a dict containing arbitrary metadata
    # - id: (optional) a string identifier for the document
    # The metadata attribute can capture information about the source of the document, its relationship to other documents, and other information.
    # Note that an individual Document object often represents a chunk of a larger document.
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]
    logger.info(f"Loaded {len(documents)} sample documents")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Splits documents into smaller chunks.
    """
    # For both information retrieval and downstream question-answering purposes, a page may be too coarse a representation.
    # So it's necessary to split each document into smaller chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        add_start_index=True,
    )

    all_chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
    return all_chunks


def add_docs_to_vector_store(vector_store: VectorStore, documents: list[Document]):
    """
    Adds documents to the vector store.

    Args:
        vector_store: The vector store to add documents to.
        documents: The list of documents to add.
    """
    # Split documents into smaller chunks
    chunks = split_documents(documents)

    # Index documents into vector store
    result = vector_store.add_documents(documents=chunks)
    logger.info(f"Added {len(result)} entries to vector store")
    # Alternatively is possible to generate embeddings and add them to vector store
    # embeddings = []
    # for chunk in chunks:
    #     # Generate embeddings for each chunk
    #     embedding = embeddings.embed_query(chunk)
    #     logger.debug(f"Generated embedding for chunk: {chunk}")
    #     embeddings.append(embedding)
    # vector_store.add_embeddings(embeddings)

    collection_name = getattr(vector_store, "collection_name", None)
    if collection_name is None:
        # Fall back to other ways to get collection name or handle the case
        pass
    logger.info(f"Saved vector store collection: {collection_name}")


def query_vector_store(
    vector_store: VectorStore, query: str
) -> list[tuple[Document, float]]:
    """
    Query the vector store with the given question.

    Args:
        vector_store: The vector store to query.
        query: The query string.

    Returns:
        list[tuple[Document, float]]: List of documents retrieved from the vector store that are similar to the provided query.
    """
    # Query the vector store
    # Put k=1 to retrieve only the first doc found by similarity search
    result = vector_store.similarity_search_with_score(query, k=1)
    logger.info(f"Retrieved {len(result)} documents from vector store")
    return result


def set_vector_store(v_store: VectorStore) -> None:
    global global_vector_store
    global_vector_store = v_store


# LangChain VectorStore objects do not subclass Runnable. LangChain Retrievers are Runnables, so they implement a standard set of methods
# (e.g., synchronous and asynchronous invoke and batch operations).
# Although we can construct retrievers from vector stores, retrievers can interface with non-vector store sources of data, as well (such as external APIs).
@chain
def retriever(query: str) -> list[Document]:
    """
    Retrieve documents from the vector store that are similar to the provided query.

    Args:
        query: Query to search the vector store.

    Returns:
        list[Document]: List of documents retrieved from the vector store that are similar to the provided query.
    """
    results = query_vector_store(global_vector_store, query)
    # Print the retrieved documents
    for doc, score in results:
        logger.info(f"Score: {score}, Document ID: {doc.metadata['_id']}")
    return [doc for doc, _ in results]


def main():
    """
    Main entry point for the application.
    """
    logger.info(f"Application started in {Config.APP_MODE} mode")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Semantic search application")
    parser.add_argument(
        "--use-sample-docs",
        action="store_true",
        help="Use sample documents instead of loading from PDF",
    )
    args = parser.parse_args()

    # Check if we should use sample docs from either env var or command line arg
    use_sample_docs = Config.USE_SAMPLE_DOCS or args.use_sample_docs
    documents = []
    if use_sample_docs:
        logger.info("Using sample documents")
        documents = get_sample_documents()
    else:
        logger.info("Loading documents from PDF")
        documents = load_documents()

    # Initialize embeddings model available in Ollama
    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
    vector_store = get_vector_store(embeddings, Config.VECTOR_STORE_DOC_COLLECTION_NAME)

    add_docs_to_vector_store(vector_store, documents)

    # --Query the vector store
    # You can query directly in a synchronous way the vector store...
    # query = "When was Nike incorporated?"
    # # retrieved_docs = retriever.invoke(query)

    # ...otherwise, you can use batch (with properly structured inputs)
    queries = [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ]
    batch_results = retriever.batch(queries)

    # In case of usage of vector store (instead of for example, external API) we can construct retrievers directly using API exposed by vector stores
    # retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    # Print the results
    for i, docs in enumerate(batch_results):
        logger.info(f"Query: {queries[i]}")
        logger.info(f"Batch result {i+1}: Retrieved {len(docs)} documents")
        for doc in docs:
            logger.info(f"Document: {doc.page_content}")


if __name__ == "__main__":
    main()
