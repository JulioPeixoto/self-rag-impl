import fastapi
import uvicorn

app = fastapi.FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run("main:app", port=8080, reload=True)