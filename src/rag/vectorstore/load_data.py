"""
Script to load SQL examples from CSV file into ChromaDB vector store.

This script reads SQL examples (schema, question, SQL query) from a CSV file
and loads them into the ChromaDB vector store for retrieval-augmented generation.
"""

import csv
import os
import pandas as pd
from pathlib import Path

from chroma import ChromaVectorStore


def parse_schema_question_sql(row):
    """
    Parse schema, question and SQL from an input-output pair.

    Args:
        row: Dictionary with 'input' and 'output' keys.

    Returns:
        Tuple of (schema, question, sql).
    """
    input_text = row["input"]
    output_text = row["output"]

    # Split by lines to better handle the format
    lines = input_text.split("\n")

    # The schema typically starts after "-- Database schema" and ends before the question
    schema = ""
    question = ""
    capture_schema = False

    schema, question = input_text.split("\n-- -- ", 1)

    # Remove any residual markers in schema
    schema = schema.replace("-- Database schema", "").strip()

    # Fallback: If we still couldn't parse properly
    if not schema or not question:
        print("Fallback parsing method...")
        # Just split the input in half if we couldn't parse properly
        parts = input_text.split("--")
        if len(parts) > 1:
            schema = parts[0].replace("-- Database schema", "").strip()
            question = parts[-1].replace(" -- -- ").strip()

    # Output is SQL
    sql = output_text.strip()

    print(f"DEBUG - Schema extracted: {schema[:50]}...")
    print(f"DEBUG - Question extracted: {question}")

    return schema, question, sql


def load_data_to_vectorstore(csv_path, vector_store):
    """
    Load SQL examples from CSV file to vector store.

    Args:
        csv_path: Path to the CSV file.
        vector_store: ChromaVectorStore instance.

    Returns:
        Number of examples loaded.
    """
    # Read CSV file using pandas
    df = pd.read_csv(csv_path)
    count = 0

    print(f"Loading {len(df)} examples from {csv_path}...")

    for _, row in df.iterrows():
        try:
            # Parse schema, question and SQL
            schema, question, sql = parse_schema_question_sql(row)

            # Store in vector store - now we only use the index_question_sql method
            doc_id = vector_store.index_question_sql(
                question=question, sql=sql, schema=schema
            )
            count += 1

            if count % 100 == 0:
                print(f"Loaded {count} examples...")

        except Exception as e:
            print(f"Error loading row: {e}")

    print(f"Successfully loaded {count} examples to vector store.")
    return count


def main():
    """Main function to run the data loading process."""
    # Path to CSV file
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "rag" / "processed"
    csv_path = data_dir / "test.csv"

    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return

    # Create vector store directory if it doesn't exist
    vector_store_path = Path(__file__).parent
    vector_store_path.mkdir(exist_ok=True, parents=True)

    # Initialize vector store
    vector_store = ChromaVectorStore(persist_directory=str(vector_store_path))

    # Load data to vector store
    count = load_data_to_vectorstore(csv_path, vector_store)

    print(f"Total examples loaded: {count}")

    # Test a retrieval using only question_sql collection
    test_question = "How many singers are in the database?"
    results = vector_store.retrieve_relevant_question_sql(test_question, k=3)

    print("\nTest retrieval:")
    print(f"Query: {test_question}")
    print(f"Found {len(results)} relevant examples.")

    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Similarity: {result['similarity']:.4f}):")
        print(f"Question: {result['question']}")
        print(f"SQL: {result['sql']}")
        print(f"Schema: {result['schema'][:100]}...")  # Show first 100 chars of schema


if __name__ == "__main__":
    main()
