from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel
from typing import Optional
from llm import get_sql_query, recognise_intent, get_other_query
from get_context import chroma_client
import uvicorn

vector_store = chroma_client()

# Create FastAPI app
app = FastAPI(
    title="SQL Generation API",
    description="API for generating SQL queries from natural language questions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://lemon-coast-0bbfc8a03.6.azurestaticapps.net",
    ],  # or ["*"] during dev
    allow_credentials=True,
    allow_methods=["*"],  # needed to allow OPTIONS, POST, etc.
    allow_headers=["*"],  # allow all headers from browser
)


# Define request model
class SQLGenerationRequest(BaseModel):
    question: str
    # schema: Optional[str] = None  # Optional schema override


# Define response model
class SQLGenerationResponse(BaseModel):
    question: str
    sql_query: str
    model_used: str


@app.get("/")
async def root():
    return {
        "message": "SQL Generation API is running! Use /generate-sql endpoint to generate SQL queries."
    }


@app.post("/generate-sql", response_model=SQLGenerationResponse)
async def generate_sql(request: SQLGenerationRequest):
    try:
        # Get SQL query using the LLM
        intent = recognise_intent(request.question)
        if intent and intent == "specific":
            response = get_sql_query(request.question, vector_store)
        else:
            response = get_other_query(request.question)

        # Extract SQL from response
        if "choices" in response and len(response["choices"]) > 0:
            sql = response["choices"][0]["message"]["content"]
            # if not sql.endswith(";"):
            #     sql += ";"
            model_name = response.get("model", "Unknown")

            return SQLGenerationResponse(
                question=request.question, sql_query=sql, model_used=model_name
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate SQL query")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")


# For direct execution
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
