from contextlib import asynccontextmanager
from fastapi import FastAPI, status

from models.loader import load_artifacts, cleanup
from api.endpoints.predict import router as predict_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load_artifacts(app)
    artifacts = load_artifacts(app)
    next(artifacts)
    yield
    cleanup(app)

app = FastAPI(
    lifespan=lifespan, 
    title="Maize Disease Detection API", description="API for detecting diseases in maize leaves using a pre-trained ResNet model.", 
    version="1.0.0")

@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return "Maize Disease Detection"

app.include_router(predict_router)