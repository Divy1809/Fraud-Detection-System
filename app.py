from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from llm_agent import fraud_decision
import shutil
import os
import uuid

from image_check import check_duplicate, detect_ai_image, register_image_hash

app = FastAPI()


def _parse_llm_output(text):
    decision = "Genuine"
    reason = text.strip() if text else ""

    if not text:
        return decision, reason

    for line in text.splitlines():
        line = line.strip()
        lower_line = line.lower()
        if lower_line.startswith("decision:"):
            decision = line.split(":", 1)[1].strip() or decision
        elif lower_line.startswith("reason:"):
            reason = line.split(":", 1)[1].strip() or reason

    return decision, reason

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


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):

    # ✅ Step 1: file type check
    if file.content_type not in ["image/jpeg", "image/png"]:
        return {"status": "rejected", "reason": "only jpg/png allowed"}

    # ✅ Step 2: save file
    unique_name = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ Step 3: duplicate check
    duplicate_flag = False
    try:
        duplicate_flag = check_duplicate(file_path)
    except Exception as e:
        print("Duplicate check error:", e)

    # ✅ Step 4: AI detection
    ai_flag = False
    try:
        prediction = detect_ai_image(file_path)
        ai_flag = prediction == 1
    except Exception as e:
        print("AI detection error:", e)

    # ✅ Step 5: LLM decision
    context = {
        "duplicate": duplicate_flag,
        "ai_generated": ai_flag
    }

    # ✅ Step 6: final decision
    llm_output = fraud_decision(context)
    print("LLM Output:", llm_output)
    decision, reason = _parse_llm_output(llm_output)
    if decision.lower() == "fraud":
        os.remove(file_path)
        return {"status": "rejected", "decision": decision, "reason": reason}

    register_image_hash(file_path)
    return {"status": "approved", "decision": decision, "reason": reason}

