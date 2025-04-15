"""
ChromaDB Vector Store Implementation for SQL RAG.

This module provides a ChromaDB-based implementation of the IVectorstore interface
for storing and retrieving SQL-related documents (schema, questions, and SQL queries).
"""

import chromadb
import uuid
import pandas as pd
import json
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
from .ivectorstore import IVectorstore
from chromadb.utils.embedding_functions import EmbeddingFunction


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Sentence Transformer embedding function that follows ChromaDB's expected interface.
    """

    def __init__(self, model_name: str):
        """
        Initialize with a sentence transformer model.

        Args:
            model_name: The name of the sentence transformer model.
        """
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts.

        Args:
            input: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()


class ChromaVectorStore(IVectorstore):
    """
    A ChromaDB-based vector store implementation for SQL RAG.

    This class implements the IVectorstore interface and provides methods for storing
    and retrieving SQL-related documents using ChromaDB with UAE-Large-V1 embeddings.
    """

    def __init__(self, persist_directory: str = "./vectorstore"):
        """
        Initialize the ChromaDB vector store.

        Args:
            persist_directory: Directory where ChromaDB will persist the data.
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create embedding function with proper interface
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            "WhereIsAI/UAE-Large-V1"
        )

        # Create only the question_sql collection
        self.question_sql_collection = self.client.get_or_create_collection(
            name="question_sql",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def _generate_id(self) -> str:
        """
        Generate a unique ID for a document.

        Returns:
            A unique ID string.
        """
        return str(uuid.uuid4())

    def retrieve_relevant_question_sql(
        self, question: str, k: int = 5, **kwargs
    ) -> list:
        """
        Retrieve relevant question-SQL pairs based on a question.

        Args:
            question: The natural language question to find relevant items for.
            k: Number of results to return (default: 5).
            **kwargs: Additional arguments including threshold for filtering.

        Returns:
            List of relevant question-SQL documents.
        """
        threshold = kwargs.get("threshold", 0.3)

        results = self.question_sql_collection.query(
            query_texts=[question], n_results=k
        )

        relevant_items = []
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            score = (
                results.get("distances", [[]])[0][i] if results.get("distances") else 0
            )

            # Convert distance to similarity score (1 - distance)
            similarity = 1 - score if score else 0
            if similarity >= threshold:
                item = {
                    "id": doc_id,
                    "schema": metadata.get("schema", ""),
                    "question": metadata.get("question", ""),
                    "sql": metadata.get("sql", ""),
                    "similarity": similarity,
                }
            relevant_items.append(item)
            # print(f"Query: {question}")
            # print("Distances:", results.get("distances", [[]])[0])
            # print("Retrieved Metadata Sample:", results["metadatas"][0][0])

        return relevant_items

    def index_question_sql(
        self, question: str, sql: str, schema: str = "", **kwargs
    ) -> str:
        """
        Add a question-SQL pair to the vector store.

        Args:
            question: The natural language question.
            sql: The corresponding SQL query.
            schema: The database schema (in inline format).
            **kwargs: Additional metadata.

        Returns:
            ID of the added document.
        """
        doc_id = kwargs.get("id", self._generate_id())

        # Create a JSON-like structure for storage
        metadata = {"schema": schema, "question": question, "sql": sql}

        # Add additional metadata if provided
        for key, value in kwargs.items():
            if key != "id":
                metadata[key] = value

        # Store the document in the collection
        self.question_sql_collection.add(
            ids=[doc_id],
            documents=[question],  # We embed the question for retrieval
            metadatas=[metadata],
        )

        return doc_id

    def fetch_all_vectorstore_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch all data from the vector store.

        Args:
            **kwargs: Optional filters.

        Returns:
            DataFrame containing all the stored documents.
        """
        all_data = []

        question_sql_data = self.question_sql_collection.get()
        for i, doc_id in enumerate(question_sql_data["ids"]):
            metadata = question_sql_data["metadatas"][i]  # ✅ Fixed variable name
            all_data.append(
                {
                    "id": doc_id,
                    "collection": "question_sql",
                    "schema": metadata.get("schema", ""),
                    "question": metadata.get("question", ""),
                    "sql": metadata.get("sql", ""),
                }
            )

        return pd.DataFrame(all_data)

    def delete_vectorstore_data(self, item_id: str, **kwargs) -> bool:
        """
        Delete an item from the vector store.

        Args:
            item_id: ID of the item to delete.
            **kwargs: Optional parameters.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            self.question_sql_collection.delete(ids=[item_id])
            return True
        except Exception as e:
            print(f"Error deleting item {item_id}: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize the vector store
    vector_store = ChromaVectorStore(persist_directory="./vectorstore")

    # # Example schema
    # schema = """
    # CREATE TABLE employees (
    #     id INT PRIMARY KEY,
    #     name TEXT,
    #     department TEXT,
    #     salary INT
    # );
    # """

    # # Example question and SQL
    # question = "List all employees in the IT department."
    # sql = "SELECT * FROM employees WHERE department = 'IT';"

    # # Add the question-SQL pair to the vector store
    # doc_id = vector_store.index_question_sql(question=question, sql=sql, schema=schema)
    # print(f"Added document with ID: {doc_id}")

    # Retrieve relevant question-SQL pairs for a new question
    new_question = "How many singers do we have?"
    results = vector_store.retrieve_relevant_question_sql(new_question)

    for result in results:
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Question: {result['question']}")
        print(f"SQL: {result['sql']}")
        print(f"Schema: {result['schema']}")
        print("---")
