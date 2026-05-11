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

N8N_WEBHOOK_URL = "https://asmaamamdouh2005.app.n8n.cloud/webhook/crm-result"

TEXT_MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
WHISPER_MODEL_NAME = "base"

UPLOAD_FOLDER = "uploads"
DB_PATH = "crm_sentiment.db"
MAX_CONTENT_LENGTH_MB = 25

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {
    "wav", "mp3", "m4a", "ogg", "webm", "mp4", "mpeg"
}

# =========================
# LOAD MODELS
# =========================

print("Loading sentiment model...")
classifier = pipeline(
    "sentiment-analysis",
    model=TEXT_MODEL_NAME
)

print("Loading whisper model...")
speech_model = whisper.load_model(WHISPER_MODEL_NAME)

print("Models loaded successfully.")

# =========================
# DATABASE
# =========================

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS customer_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            input_type TEXT NOT NULL,
            original_text TEXT,
            transcribed_text TEXT,
            sentiment TEXT NOT NULL,
            sentiment_ar TEXT NOT NULL,
            confidence REAL NOT NULL,
            recommended_action TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()

# =========================
# HELPERS
# =========================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def allowed_audio_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_AUDIO_EXTENSIONS
    )


def map_sentiment(label):
    value = (label or "").lower().strip()

    if "positive" in value:
        return "Happy", "سعيد"

    elif "negative" in value:
        return "Angry", "غاضب"

    else:
        return "Neutral", "محايد"


def recommended_action(sentiment):
    mapping = {
        "Angry": "Escalate immediately to customer support",
        "Happy": "Thank the customer and continue engagement",
        "Neutral": "Monitor conversation and collect more context",
    }

    return mapping.get(sentiment, "Manual review")


def analyze_text_message(text):
    result = classifier(text)[0]

    sentiment, sentiment_ar = map_sentiment(
        result.get("label")
    )

    return {
        "sentiment": sentiment,
        "sentiment_ar": sentiment_ar,
        "confidence": round(
            float(result.get("score", 0.0)),
            4
        ),
        "recommended_action":
            recommended_action(sentiment),
    }


def transcribe_audio(audio_path):
    result = speech_model.transcribe(
        audio_path,
        fp16=False
    )

    return (result.get("text") or "").strip()


def save_result(payload):
    conn = get_db_connection()

    cursor = conn.execute("""
        INSERT INTO customer_interactions
        (
            customer_name,
            input_type,
            original_text,
            transcribed_text,
            sentiment,
            sentiment_ar,
            confidence,
            recommended_action,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        payload["customer_name"],
        payload["input_type"],
        payload.get("original_text"),
        payload.get("transcribed_text"),
        payload["sentiment"],
        payload["sentiment_ar"],
        payload["confidence"],
        payload["recommended_action"],
        payload["created_at"],
    ))

    conn.commit()

    row_id = cursor.lastrowid

    conn.close()

    return row_id


def notify_n8n(payload):

    if not N8N_WEBHOOK_URL:
        return {
            "sent": False,
            "message": "N8N webhook not configured"
        }

    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=10
        )

        return {
            "sent": True,
            "status_code": response.status_code
        }

    except Exception as e:
        return {
            "sent": False,
            "message": str(e)
        }


def build_text_payload(customer_name, text):

    analysis = analyze_text_message(text)

    return {
        "customer_name": customer_name,
        "input_type": "text",
        "original_text": text,
        "transcribed_text": text,
        **analysis,
        "created_at": now_iso(),
    }


def build_voice_payload(customer_name, file_storage):

    if not file_storage or file_storage.filename == "":
        raise ValueError("Audio file is required")

    if not allowed_audio_file(file_storage.filename):
        raise ValueError("Unsupported audio format")

    ext = file_storage.filename.rsplit(".", 1)[1].lower()

    safe_name = secure_filename(
        file_storage.filename
    )

    temp_name = (
        f"{uuid.uuid4().hex}_"
        f"{safe_name.rsplit('.', 1)[0]}.{ext}"
    )

    temp_path = os.path.join(
        UPLOAD_FOLDER,
        temp_name
    )

    file_storage.save(temp_path)

    try:
        text = transcribe_audio(temp_path)

        if not text:
            raise ValueError(
                "Could not transcribe audio"
            )

        analysis = analyze_text_message(text)

        return {
            "customer_name": customer_name,
            "input_type": "voice",
            "original_text": None,
            "transcribed_text": text,
            **analysis,
            "created_at": now_iso(),
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# =========================
# FRONTEND
# =========================

HOME_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>CRM Sentiment Analysis</title>

<style>

body{
font-family:Arial,sans-serif;
background:#f5f7fb;
margin:0;
padding:0;
color:#222;
}

.container{
width:92%;
max-width:1100px;
margin:30px auto;
}

h1{
text-align:center;
color:#0b4f8a;
}

.grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:20px;
}

.card{
background:white;
border-radius:16px;
padding:20px;
box-shadow:0 4px 15px rgba(0,0,0,0.08);
}

input,textarea,button{
width:100%;
padding:12px;
margin-top:10px;
border-radius:10px;
border:1px solid #ccc;
box-sizing:border-box;
}

button{
background:#0b4f8a;
color:white;
font-weight:bold;
cursor:pointer;
}

.result{
margin-top:25px;
background:#111827;
color:#e5e7eb;
border-radius:14px;
padding:18px;
white-space:pre-wrap;
}

</style>
</head>

<body>

<div class="container">

<h1>CRM Sentiment Analysis</h1>

<div class="grid">

<div class="card">

<h2>تحليل رسالة نصية</h2>

<input
type="text"
id="textCustomerName"
placeholder="اسم العميل"
/>

<textarea
id="textMessage"
placeholder="رسالة العميل"
></textarea>

<button onclick="analyzeText()">
تحليل النص
</button>

</div>

<div class="card">

<h2>تحليل رسالة صوتية</h2>

<input
type="text"
id="voiceCustomerName"
placeholder="اسم العميل"
/>

<input
type="file"
id="audioFile"
/>

<button onclick="analyzeVoice()">
تحليل الصوت
</button>

</div>

</div>

<div class="result" id="resultBox">
النتيجة ستظهر هنا...
</div>

</div>

<script>

const resultBox =
document.getElementById("resultBox");

function showResult(data){
resultBox.textContent =
JSON.stringify(data,null,2);
}

async function analyzeText(){

const customer_name =
document.getElementById(
"textCustomerName"
).value.trim();

const text =
document.getElementById(
"textMessage"
).value.trim();

const response =
await fetch("/analyze_text",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
customer_name,
text
})
});

const data = await response.json();

showResult(data);
}

async function analyzeVoice(){

const customer_name =
document.getElementById(
"voiceCustomerName"
).value.trim();

const audioFile =
document.getElementById(
"audioFile"
).files[0];

const formData = new FormData();

formData.append(
"customer_name",
customer_name
);

formData.append(
"audio",
audioFile
);

const response =
await fetch("/analyze_voice",{
method:"POST",
body:formData
});

const data = await response.json();

showResult(data);
}

</script>

</body>
</html>
"""

# =========================
# FLASK APP
# =========================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_CONTENT_LENGTH_MB
    * 1024
    * 1024
)

@app.route("/")
def home():
    return render_template_string(
        HOME_HTML
    )


@app.route("/health")
def health():

    return jsonify({
        "ok": True,
        "message":
            "CRM Sentiment API is running"
    })


@app.route("/history")
def history():

    limit = min(
        int(request.args.get("limit", 10)),
        50
    )

    conn = get_db_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM customer_interactions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

    conn.close()

    return jsonify(
        [dict(row) for row in rows]
    )


@app.route(
    "/analyze_text",
    methods=["POST"]
)
def analyze_text_api():

    try:

        data = (
            request.get_json(silent=True)
            or
            request.form.to_dict()
        )

        customer_name = (
            data.get("customer_name")
            or
            "Unknown Customer"
        ).strip()

        text = (
            data.get("text")
            or
            ""
        ).strip()

        if not text:
            return jsonify({
                "ok": False,
                "error": "Text is required"
            }), 400

        payload = build_text_payload(
            customer_name,
            text
        )

        record_id = save_result(payload)

        n8n_result = notify_n8n({
            **payload,
            "record_id": record_id
        })

        return jsonify({
            "ok": True,
            "record_id": record_id,
            **payload,
            "n8n": n8n_result
        })

    except Exception as e:

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route(
    "/analyze_voice",
    methods=["POST"]
)
def analyze_voice_api():

    try:

        customer_name = (
            request.form.get(
                "customer_name"
            )
            or
            "Unknown Customer"
        ).strip()

        audio_file = request.files.get(
            "audio"
        )

        payload = build_voice_payload(
            customer_name,
            audio_file
        )

        record_id = save_result(payload)

        n8n_result = notify_n8n({
            **payload,
            "record_id": record_id
        })

        return jsonify({
            "ok": True,
            "record_id": record_id,
            **payload,
            "n8n": n8n_result
        })

    except ValueError as e:

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400

    except Exception as e:

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )