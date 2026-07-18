from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.predict import predict_image

app = FastAPI(title="NeuralLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    results = predict_image(image_bytes)

    predictions = [
        {
            "rank": i + 1,
            "label": label.replace("_", " ").title(),
            "confidence": round(float(confidence), 4),
        }
        for i, (class_id, label, confidence) in enumerate(results)
    ]

    return {"predictions": predictions}