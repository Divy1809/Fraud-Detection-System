from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid

from image_check import check_duplicate, detect_ai_image

app = FastAPI()

UPLOAD_FOLDER = "uploaded_images"

@app.get("/")
def home():
    return {"message": "Refund Fraud Detection System Running"}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):

    unique_name = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # duplicate check
    if check_duplicate(file_path):
        os.remove(file_path)
        return {"status": "rejected", "reason": "duplicate image"}

    # AI image detection
    prediction = detect_ai_image(file_path)

    if prediction == 1:
        os.remove(file_path)
        return {"status": "rejected", "reason": "AI generated image"}

    return {"status": "approved"}