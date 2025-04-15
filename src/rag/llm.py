import requests
from get_context import chroma_client, retreieve_results
import json


def recognise_intent(query: str):
    # schema = retreieve_results(vector_store, query)
    url = "http://localhost:8080/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    data = {
        "model": "Llama-3.2-3B-Instruct-F16.gguf",  # your local model name; can be anything
        "messages": [
            {
                "role": "system",
                "content": """You are an AI assistant that classifies user queries as either "specific" or "general".

                A "specific" query refers to something that can be answered with a factual, data-driven, or concrete response — such as those involving dates, counts, or database-style queries.

                A "general" query refers to open-ended, opinion-based, or abstract questions that don't rely on structured data.

                Return a JSON object in the format: {"intent": "specific"} or {"intent": "general"}

                Examples:
                Query: How many users signed up in 2023?
                {"intent": "specific"}

                Query: What is the capital of France?
                {"intent": "general"}

                Query: What makes a good concert experience?
                {"intent": "general"}

                Query: List all concerts held in 2022.
                {"intent": "specific"}

                Query: How many people were there in concert between 2024 and 2025?
                {"intent": "specific"}


                """,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 50,
    }
    response = requests.post(url, headers=headers, json=data)

    try:
        intent = json.loads(response.json()["choices"][0]["message"]["content"])["intent"]
    except:
        intent = "general"
    return intent


def get_sql_query(query: str, vector_store):

    schema = retreieve_results(vector_store, query)
    url = "http://localhost:8080/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    data = {
        "model": "Llama-3.2-3B-Instruct-F16.gguf",  # your local model name; can be anything
        "messages": [
            {
                "role": "system",
                "content": " You are a helpful chatbot. Given the following database schema and natural language question, write the SQL query that answers the question and explain the query.",
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


def get_other_query(query: str):

    url = "http://localhost:8080/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    data = {
        "model": "Llama-3.2-3B-Instruct-F16.gguf",  # your local model name; can be anything
        "messages": [
            {
                "role": "system",
                "content": " You are a helpful, concise, and knowledgeable assistant. Answer the user’s questions clearly and accurately.",
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }
    response = requests.post(url, headers=headers, json=data)

    return response.json()
