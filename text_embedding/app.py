from pydantic import BaseModel, Field
from typing import List, Optional, Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import json

# Load embedding model
with open("model_path.json", "r") as f:
    model_path = json.load(f)["model_path"]
model = SentenceTransformer(model_path)


app = FastAPI(
    title="Text Embeddings API compatible with OpenAI's API",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request and response models
class CreateEmbeddingRequest(BaseModel):
    model: Optional[str] = Field(
        description="The model to use for generating embeddings.", default=None)
    input: Union[str, List[str]] = Field(description="The input to embed.")
    user: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "The food was delicious and the waiter...",
                }
            ]
        }
    }


class Embedding(BaseModel):
    object: str
    embedding: List[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class CreateEmbeddingResponse(BaseModel):
    object: str
    data: List[Embedding]
    model: str
    usage: Usage


@app.post("/v1/embeddings", response_model=CreateEmbeddingResponse)
async def create_embedding(request: CreateEmbeddingRequest):
    """Create embeddings for text.

    Args:
        request (CreateEmbeddingRequest): The request object.

    Returns:
        CreateEmbeddingResponse: The response object.
    """
    if isinstance(request.input, str):
        request.input = [request.input]

    embeddings = model.encode(request.input)

    return CreateEmbeddingResponse(
        object="list",
        data=[
            Embedding(
                object="embedding",
                embedding=embedding.tolist(),
                index=index
            )
            for index, embedding in enumerate(embeddings)
        ],
        model=model_path,
        usage=Usage(
            prompt_tokens=len(model.tokenize(request.input)),
            total_tokens=len(model.tokenize(request.input))
        )
    )
