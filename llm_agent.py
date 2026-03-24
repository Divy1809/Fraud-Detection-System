import os
import requests

API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or "AIzaSyDhhOHSCsZXUc3jvYX7gVj5VlZAefeckTA"
)

print(f"DEBUG: Using API key: {API_KEY[:20]}..." if API_KEY else "DEBUG: No API key found")


def _rule_based_fallback(context):
    duplicate = bool(context.get("duplicate", False))

    if duplicate:
        return "Decision: Fraud\nReason: duplicate image detected."

    return "Decision: Genuine\nReason: image accepted (LLM unavailable, duplicate not detected)."

def fraud_decision(context):
    prompt = f"""
You are an intelligent fraud detection system specializing in AI-generated image and refund fraud detection.

Input:
- Duplicate Image: {context['duplicate']}
- AI Signal from CLIP: {context['ai_generated']} (Note: CLIP has high false-positive rate, use judgment)

Your Task:
Analyze these signals and determine if the image is likely fraudulent.

Rules:
- If duplicate → Fraud (strong signal)
- If CLIP flags as AI AND image looks synthetic/unrealistic → Fraud
- If CLIP flags AI but image looks photorealistic → Genuine (CLIP error)
- Otherwise → Genuine

Answer in format:
Decision: Fraud or Genuine
Reason: short explanation based on your analysis
"""

    if not API_KEY:
        print("Gemini error: missing GEMINI_API_KEY/GOOGLE_API_KEY, using fallback")
        return _rule_based_fallback(context)

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code != 200:
            print(f"Gemini HTTP error: {response.status_code} - {response.text[:300]}")
            return _rule_based_fallback(context)

        data = response.json()

        candidates = data.get("candidates") or []
        if not candidates:
            print(f"Gemini unexpected payload: {str(data)[:300]}")
            return _rule_based_fallback(context)

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text = "\n".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        ).strip()

        if not text:
            print("Gemini response had no text parts, using fallback")
            return _rule_based_fallback(context)

        print("Gemini RAW:", text)
        return text.strip()

    except Exception as e:
        print("Gemini error:", e)
        return _rule_based_fallback(context)