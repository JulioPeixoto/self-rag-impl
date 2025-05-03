from pydantic import BaseModel, AnyUrl
from typing import Optional

class QueryRequest(BaseModel):
    query: str
    webpage_url: Optional[AnyUrl] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    error: Optional[str] = None
    retrieved_documents: Optional[int] = None
