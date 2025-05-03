import fastapi
import uvicorn
from routes.api import router as api_router


app = fastapi.FastAPI()

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("service.main:app", host="0.0.0.0", port=8080, reload=True)