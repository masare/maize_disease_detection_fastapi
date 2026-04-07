from fastapi import APIRouter, File, UploadFile, Request, status
from services.inference_service import run_inference

router = APIRouter()


@router.get("/model", status_code=status.HTTP_200_OK)
def model_imfo(request: Request):
    return f"Model loaded in {request.app.state.model_loading_time}"


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    return run_inference(request.app, file)