"""
Script to fetch a specific document by ID from a ChromaDB collection.

This script connects to the existing ChromaDB database and retrieves a specific document
from a given collection using its ID.
"""

import sys
import os
from pathlib import Path
import chromadb
import json

def fetch_document_by_id(collection_name, document_id, persist_directory="./"):
    """
    Fetch a specific document by ID from a ChromaDB collection.
    
    Args:
        collection_name: Name of the collection to query.
        document_id: ID of the document to retrieve.
        persist_directory: Directory where ChromaDB data is stored.
        
    Returns:
        Dictionary with document data if found, None otherwise.
    """
    print(f"Connecting to ChromaDB at: {persist_directory}")
    # Initialize the client with the existing data
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # Get the collection
        collection = client.get_collection(collection_name)
        print(f"Connected to collection: {collection_name}")
        
        # Query for the specific document ID
        result = collection.get(ids=[document_id])
        
        if result and len(result["ids"]) > 0:
            print(f"Document found with ID: {document_id}")
            
            # Format and return the results
            document_data = {
                "id": result["ids"][0],
                "document": result["documents"][0] if "documents" in result and result["documents"] else None,
                "embedding": result["embeddings"][0] if "embeddings" in result and result["embeddings"] else None,
                "metadata": result["metadatas"][0] if "metadatas" in result and result["metadatas"] else None
            }
            
            return document_data
        else:
            print(f"No document found with ID: {document_id}")
            return None
            
    except Exception as e:
        print(f"Error fetching document: {str(e)}")
        return None

def main():
    """Main function to fetch and display a specific document."""
    # Path to the vector store
    vector_store_path = Path(__file__).parent
    
    # Collection name and document ID to fetch
    collection_name = "question_sql"
    document_id = "812f6ace-1a5f-40e7-b7e8-8392ad462418"
    
    # Fetch the document
    document = fetch_document_by_id(collection_name, document_id, str(vector_store_path))
    
    if document:
        print("\nDocument Details:")
        print("-----------------")
        
        # Print the document content
        if document["document"]:
            print(f"\nContent:")
            print(document["document"])
        
        # Print metadata if available
        if document["metadata"]:
            print(f"\nMetadata:")
            try:
                if isinstance(document["metadata"], dict):
                    for key, value in document["metadata"].items():
                        print(f"  {key}: {value}")
                else:
                    print(document["metadata"])
            except Exception as e:
                print(f"Error displaying metadata: {str(e)}")
        
        # Print embedding dimensions (not the full embedding as it would be too verbose)
        if document["embedding"]:
            embedding_length = len(document["embedding"])
            print(f"\nEmbedding: {embedding_length} dimensions")
            # Print first few values as sample
            print(f"Sample values: {document['embedding'][:5]}...")

if __name__ == "__main__":
    main()