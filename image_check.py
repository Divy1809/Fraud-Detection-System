from PIL import Image
import os
import imagehash
import torch
from transformers import CLIPProcessor, CLIPModel

# ==============================
# 🔁 DUPLICATE DETECTION
# ==============================

HASH_DB = "hashes.txt"

def load_hashes():
    if not os.path.exists(HASH_DB):
        return set()
    with open(HASH_DB, "r") as f:
        return {line.strip() for line in f.read().splitlines() if line.strip()}

def save_hash(hash_value):
    with open(HASH_DB, "a") as f:
        f.write(hash_value + "\n")


def _normalized_hash(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    return imagehash.phash(img)

def check_duplicate(image_path):
    try:
        # 🔥 Normalize image (VERY IMPORTANT)
        img_hash = _normalized_hash(image_path)

        existing_hashes = load_hashes()

        print("Current hash:", img_hash)

        # 🔥 Compare with existing hashes
        for stored_hash in existing_hashes:
            try:
                stored_hash_obj = imagehash.hex_to_hash(stored_hash)
            except Exception:
                # Skip malformed/legacy entries instead of failing whole check.
                continue

            diff = stored_hash_obj - img_hash
            print("Hash diff:", diff)

            # ✅ Exact-match duplicate only
            if diff == 0:
                print("Duplicate detected")
                return True

        return False

    except Exception as e:
        print("Duplicate check error:", e)
        return False


def register_image_hash(image_path):
    try:
        img_hash = str(_normalized_hash(image_path))
        existing_hashes = load_hashes()
        if img_hash not in existing_hashes:
            save_hash(img_hash)
    except Exception as e:
        print("Hash register error:", e)


# ==============================
# 🤖 CLIP AI IMAGE DETECTION
# ==============================

# Load once (IMPORTANT)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Conservative thresholds to avoid rejecting genuine photos.
# CLIP is weak at AI detection, so we make it extremely strict and let Gemini decide.
AI_CONFIDENCE_MIN = 0.99
AI_MARGIN_MIN = 0.50

def detect_ai_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")

        inputs = clip_processor(
            text=["real image", "AI generated image"],
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        real_prob = probs[0][0].item()
        ai_prob = probs[0][1].item()

        margin = ai_prob - real_prob
        print(f"CLIP → Real: {real_prob:.2f}, AI: {ai_prob:.2f}, Margin: {margin:.2f}")

        # Mark as AI only when confidence is very high and clearly separated.
        if ai_prob >= AI_CONFIDENCE_MIN and margin >= AI_MARGIN_MIN:
            return 1  # AI generated
        return 0  # Real image

    except Exception as e:
        print("CLIP error:", e)
        return 0