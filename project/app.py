# app.py
# Full backend that serves index.html (from ./static) plus two API endpoints:
#  - POST /api/generate_story  -> { culture, theme, length, tone, prompt? }
#  - POST /api/generate_image  -> { prompt, n, size }
#
# Setup: pip install -r requirements.txt
# Then export OPENAI_API_KEY and run: python app.py
import os
import time
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable (see .env.example)")

# Optional overrides
MODEL_CHAT = os.getenv("MODEL_CHAT", "gpt-4o-mini")
MODEL_IMAGE = os.getenv("MODEL_IMAGE", "gpt-image-1")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com")

HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)  # allows requests from any origin; tighten in production

# Serve frontend
@app.route("/")
def index():
    # Serve the index.html from ./static
    return send_from_directory("static", "index.html")


def build_story_prompt(culture: str, theme: str, length: str = "short", tone: str = "warm", user_prompt: str = None):
    if user_prompt and user_prompt.strip():
        # Give user prompt priority (but keep system guard text)
        return user_prompt.strip()
    length_map = {"short": "2-4 short paragraphs", "medium": "4-7 paragraphs", "long": "8+ paragraphs"}
    length_hint = length_map.get(length, "3-5 paragraphs")
    prompt = (
        f"You are a respectful storyteller specialized in {culture} folktales.\n"
        f"Write a {length_hint} story about the theme: {theme}.\n"
        f"Tone: {tone}.\n"
        "Include a title line, the story body, and a single-line moral at the end.\n"
        "Be culturally accurate, avoid harmful stereotypes or offensive content.\n"
    )
    return prompt


@app.route("/api/generate_story", methods=["POST"])
def api_generate_story():
    """
    Expected JSON:
    {
      "culture": "...",
      "theme": "...",
      "length": "short|medium|long",
      "tone": "...",
      "prompt": "optional user prompt (overrides built from fields)"
    }
    """
    payload = request.json or {}
    culture = payload.get("culture", "generic folktale")
    theme = payload.get("theme", "wisdom")
    length = payload.get("length", "short")
    tone = payload.get("tone", "warm")
    user_prompt = payload.get("prompt")

    prompt_text = build_story_prompt(culture, theme, length, tone, user_prompt)

    url = f"{OPENAI_API_BASE}/v1/chat/completions"
    body = {
        "model": MODEL_CHAT,
        "messages": [
            {"role": "system", "content": "You are a creative storyteller and cultural advisor."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": float(payload.get("temperature", 0.8)),
        "max_tokens": int(payload.get("max_tokens", 700))
    }

    try:
        r = requests.post(url, json=body, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Extract content from chat completion response
        content = None
        if "choices" in data and len(data["choices"]) > 0:
            ch = data["choices"][0]
            if "message" in ch and "content" in ch["message"]:
                content = ch["message"]["content"]
            elif "text" in ch:
                content = ch["text"]
        if content is None:
            content = data.get("text") or str(data)
        return jsonify({"ok": True, "story": content, "meta": {"took": time.time()}}), 200
    except Exception as e:
        # include status code/message where available
        err_msg = str(e)
        try:
            err_body = getattr(e, "response").text
            err_msg += " | " + str(err_body)
        except Exception:
            pass
        return jsonify({"ok": False, "error": err_msg}), 500


@app.route("/api/generate_image", methods=["POST"])
def api_generate_image():
    """
    Expected JSON:
    {
      "prompt": "text of image prompt",
      "n": 1,
      "size": "1024x1024"
    }
    Returns: {"ok": True, "images": [ { "url": "..."} or {"b64": "..." } ]}
    """
    payload = request.json or {}
    prompt = payload.get("prompt")
    n = int(payload.get("n", 1))
    size = payload.get("size", "1024x1024")

    if not prompt or not prompt.strip():
        return jsonify({"ok": False, "error": "prompt is required"}), 400

    url = f"{OPENAI_API_BASE}/v1/images/generations"
    body = {"model": MODEL_IMAGE, "prompt": prompt, "n": n, "size": size}

    try:
        r = requests.post(url, json=body, headers=HEADERS, timeout=120)
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data.get("data", []):
            # OpenAI may return b64_json or url
            if item.get("b64_json"):
                out.append({"b64": item["b64_json"]})
            elif item.get("url"):
                out.append({"url": item["url"]})
            else:
                # unknown structure, include raw object
                out.append({"raw": item})
        return jsonify({"ok": True, "images": out}), 200
    except Exception as e:
        err_msg = str(e)
        try:
            err_body = getattr(e, "response").text
            err_msg += " | " + str(err_body)
        except Exception:
            pass
        return jsonify({"ok": False, "error": err_msg}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True, "models": {"chat": MODEL_CHAT, "image": MODEL_IMAGE}})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
