"""
Script to list all collections in the ChromaDB database.

This script connects to the existing ChromaDB database and lists all available collections.
"""

import sys
import os
from pathlib import Path
import chromadb

def list_collections(persist_directory="./vectorstore"):
    """
    List all collections in the ChromaDB database.
    
    Args:
        persist_directory: Directory where ChromaDB data is stored.
        
    Returns:
        List of collection names.
    """
    print(f"Connecting to ChromaDB at: {persist_directory}")
    # Initialize the client with the existing data
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Get all collections
    collection_names = client.list_collections()
    
    print(f"Found {len(collection_names)} collections:")
    for i, collection in enumerate(collection_names):
        # Safely handle the case where metadata might be None
        collection_type = "unknown"
        if hasattr(collection, 'metadata') and collection.metadata is not None:
            collection_type = collection.metadata.get('hnsw:space', 'unknown')
        
        print(f"{i+1}. {collection.name} (Collection type: {collection_type})")
        
        # Try to get the count of items in each collection
        try:
            collection_obj = client.get_collection(collection.name)
            count = collection_obj.count()
            print(f"   - Contains {count} documents")
        except Exception as e:
            print(f"   - Error getting count: {str(e)}")
    
    return collection_names

def get_collection_info(collection_name, persist_directory="./vectorstore"):
    """
    Get detailed information about a specific collection.
    
    Args:
        collection_name: Name of the collection to get info for.
        persist_directory: Directory where ChromaDB data is stored.
        
    Returns:
        Dictionary with collection information.
    """
    # Initialize the client with the existing data
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        collection = client.get_collection(collection_name)
        
        # Get all items in the collection (limited to first 5)
        result = collection.get(limit=5)
        
        print(f"\nCollection '{collection_name}' details:")
        print(f"Total documents: {collection.count()}")
        
        if result and result.get("ids"):
            print(f"\nSample document IDs (first 5):")
            for id in result["ids"]:
                print(f"- {id}")
        
        return result
    except Exception as e:
        print(f"Error getting collection info: {str(e)}")
        return None

def main():
    """Main function to run the collection listing process."""
    # Path to the vector store
    vector_store_path = Path(__file__).parent
    
    # List all collections
    collections = list_collections(str(vector_store_path))
    
    if collections:
        # Ask if user wants details on a specific collection
        print("\nEnter a collection number for more details (or press Enter to exit):")
        choice = input("> ")
        
        if choice.isdigit() and 1 <= int(choice) <= len(collections):
            idx = int(choice) - 1
            get_collection_info(collections[idx].name, str(vector_store_path))

if __name__ == "__main__":
    main()