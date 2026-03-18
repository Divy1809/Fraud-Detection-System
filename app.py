from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

from image_check import check_duplicate, detect_ai_image

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploaded_images"

# ✅ ensure folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.get("/")
def home():
    return {"message": "Refund Fraud Detection System Running"}


@app.post("/upload-image/")
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):

    # ✅ only jpg/png
    if file.content_type not in ["image/jpeg", "image/png"]:
        return {"status": "rejected", "reason": "only jpg png allowed"}

    # ✅ save file
    unique_name = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ duplicate check
    try:
        if check_duplicate(file_path):
            os.remove(file_path)
            return {"status": "rejected", "reason": "duplicate image"}
    except Exception as e:
        print("Duplicate check error:", e)
        os.remove(file_path)
        return {"status": "rejected", "reason": "processing failed"}

    # ✅ AI detection
    try:
        prediction = detect_ai_image(file_path)
    except Exception as e:
        print("AI detection error:", e)
        os.remove(file_path)
        return {"status": "rejected", "reason": "processing failed"}

    # ✅ result
    if prediction == 1:
        os.remove(file_path)
        return {"status": "rejected", "reason": "ai generated"}

    return {"status": "approved"}