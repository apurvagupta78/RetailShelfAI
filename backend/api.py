from fastapi import FastAPI, UploadFile, File
import shutil
import os

from models.detect import ShelfDetector

app = FastAPI(
    title="RetailShelfAI",
    version="0.1"
)

detector = ShelfDetector("weights/best.pt")


@app.get("/")
def home():
    return {
        "message": "RetailShelfAI API Running"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    image_path = f"uploads/{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = detector.detect(image_path)

    return result
