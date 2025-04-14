from vectorstore.chroma import ChromaVectorStore

from pathlib import Path

def chroma_client():
    vector_store_path = Path(__file__).parent / "vectorstore"
    vector_store_path.mkdir(exist_ok=True, parents=True)
    vector_store = ChromaVectorStore(persist_directory=str(vector_store_path))
    return vector_store

def retreieve_results(vector_store, query):
    results = vector_store.retrieve_relevant_question_sql(query, k=3)
    return results