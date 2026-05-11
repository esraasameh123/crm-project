# -*- coding: utf-8 -*-

import os
import uuid
import sqlite3
from datetime import datetime, timezone

import requests
import whisper
from flask import Flask, jsonify, request, render_template_string
from transformers import pipeline
from werkzeug.utils import secure_filename

# =========================
# CONFIG
# =========================

TEXT_MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
WHISPER_MODEL_NAME = "base"

UPLOAD_FOLDER = "uploads"
DB_PATH = "crm_sentiment.db"
MAX_CONTENT_LENGTH_MB = 25

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "webm", "mp4", "mpeg"}

# =========================
# LOAD MODELS (once)
# =========================

print("Loading sentiment model...")
classifier = pipeline("sentiment-analysis", model=TEXT_MODEL_NAME)

print("Loading whisper model...")
speech_model = whisper.load_model(WHISPER_MODEL_NAME)

print("Models loaded successfully")

# =========================
# FLASK APP
# =========================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

# =========================
# DATABASE
# =========================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customer_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT,
            input_type TEXT,
            original_text TEXT,
            transcribed_text TEXT,
            sentiment TEXT,
            sentiment_ar TEXT,
            confidence REAL,
            recommended_action TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()

# =========================
# HELPERS
# =========================

def now():
    return datetime.now(timezone.utc).isoformat()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def map_sentiment(label):
    label = (label or "").lower()

    if "positive" in label:
        return "Happy", "سعيد"
    elif "negative" in label:
        return "Angry", "غاضب"
    return "Neutral", "محايد"


def action(sentiment):
    return {
        "Angry": "Escalate immediately",
        "Happy": "Thank the customer",
        "Neutral": "Monitor conversation"
    }.get(sentiment, "Manual review")


def analyze_text(text):
    result = classifier(text)[0]
    sentiment, sentiment_ar = map_sentiment(result["label"])

    return {
        "sentiment": sentiment,
        "sentiment_ar": sentiment_ar,
        "confidence": float(result["score"]),
        "recommended_action": action(sentiment)
    }


def transcribe(path):
    result = speech_model.transcribe(path, fp16=False)
    return result["text"].strip()

# =========================
# SAVE
# =========================

def save(data):
    conn = get_db()
    cur = conn.execute("""
        INSERT INTO customer_interactions (
            customer_name, input_type, original_text,
            transcribed_text, sentiment, sentiment_ar,
            confidence, recommended_action, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["customer_name"],
        data["input_type"],
        data.get("original_text"),
        data.get("transcribed_text"),
        data["sentiment"],
        data["sentiment_ar"],
        data["confidence"],
        data["recommended_action"],
        data["created_at"]
    ))

    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return "CRM Sentiment API is running"


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/analyze_text", methods=["POST"])
def analyze_text_api():
    data = request.get_json()

    customer = data.get("customer_name", "Unknown")
    text = data.get("text", "")

    if not text:
        return jsonify({"ok": False, "error": "text required"}), 400

    analysis = analyze_text(text)

    payload = {
        "customer_name": customer,
        "input_type": "text",
        "original_text": text,
        "transcribed_text": text,
        **analysis,
        "created_at": now()
    }

    record_id = save(payload)

    return jsonify({
        "ok": True,
        "record_id": record_id,
        **payload
    })


@app.route("/analyze_voice", methods=["POST"])
def analyze_voice_api():
    customer = request.form.get("customer_name", "Unknown")
    file = request.files.get("audio")

    if not file:
        return jsonify({"ok": False, "error": "audio required"}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "invalid format"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
    file.save(path)

    try:
        text = transcribe(path)
        analysis = analyze_text(text)

        payload = {
            "customer_name": customer,
            "input_type": "voice",
            "original_text": None,
            "transcribed_text": text,
            **analysis,
            "created_at": now()
        }

        record_id = save(payload)

        return jsonify({
            "ok": True,
            "record_id": record_id,
            **payload
        })

    finally:
        if os.path.exists(path):
            os.remove(path)


@app.route("/history")
def history():
    conn = get_db()
    rows = conn.execute("SELECT * FROM customer_interactions ORDER BY id DESC LIMIT 20").fetchall()
    conn.close()

    return jsonify([dict(r) for r in rows])