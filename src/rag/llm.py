import requests
from get_context import chroma_client, retreieve_results

def get_sql_query(query: str, vector_store):

    schema = retreieve_results(vector_store, query)
    url = "http://localhost:8080/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    data = {
        "model": "Llama-3.2-3B-Instruct-F16.gguf",  # your local model name; can be anything
        "messages": [
            {
                "role": "system",
                "content": "Given the following database schema and natural language question, write the SQL query that answers the question.",
            },
            {
                "role": "user",
                "content": f"Schema: {schema} -- -- {query}",
            },
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }
    response = requests.post(url, headers=headers, json=data)

    return response.json()
