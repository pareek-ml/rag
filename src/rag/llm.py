import requests
from get_context import chroma_client, retreieve_results
from prompts import (
    SQL_PROMPT,
    INTENT_PROMPT,
    GENERIC_PROMPT,
)  # Import your prompts from the appropriate module
import json


def recognise_intent(query: str):
    # schema = retreieve_results(vector_store, query)
    url = "http://localhost:8080/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    data = {
        "model": "Llama-3.2-3B-Instruct-F16.gguf",
        "messages": [
            {
                "role": "system",
                "content": INTENT_PROMPT,
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
        intent = json.loads(response.json()["choices"][0]["message"]["content"])[
            "intent"
        ]
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
                "content": SQL_PROMPT,
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
                "content": GENERIC_PROMPT,
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
