from fastapi import HTTPException, Depends, APIRouter
from self_rag.rag import SelfRAGService
from models.query import QueryRequest, QueryResponse

router = APIRouter()

def get_rag_service():
    try:
        return SelfRAGService.get_instance()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def read_root():
    return {"message": "Self-RAG API, go to /docs to see the API documentation/Swagger UI"}

@router.post("/query", response_model=QueryResponse)
async def create_query(
    query_request: QueryRequest,
    rag_service: SelfRAGService = Depends(get_rag_service)
):
    """
    Process a query using the Self-RAG system.
    
    Optionally ingest a webpage before processing the query.
    """
    try:
        result = rag_service.process_query(
            query=query_request.query,
            webpage_url=query_request.webpage_url
        )
        
        if "error" in result:
            return QueryResponse(
                query=query_request.query,
                answer="",
                error=result["error"]
            )
        
        return QueryResponse(
            query=query_request.query,
            answer=result["answer"],
            retrieved_documents=result.get("retrieved_documents")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
