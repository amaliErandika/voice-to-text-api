from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
import whisper
from tempfile import NamedTemporaryFile
import os
from groq import Groq

app = FastAPI(title="Voice-to-Text API")

# Global Whisper model
model = None

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.on_event("startup")
async def load_whisper_model():
    """Load Whisper model at startup"""
    global model
    try:
        model = whisper.load_model("tiny")  # tiny = fast and small
        print("Whisper model loaded successfully")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")

@app.get("/", response_class=RedirectResponse)
async def root():
    return "/docs"

@app.get("/health")
async def health():
    """Health check endpoint for Railway"""
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if model is None:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")

    with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()

        # Whisper transcription
        try:
            result = model.transcribe(tmp.name)
            transcript = result["text"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Whisper error: {e}")

        # LLM response via Groq
        try:
            chat = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": transcript}
                ]
            )
            reply = chat.choices[0].message.content
        except Exception as e:
            reply = f"LLM error: {str(e)}"

    return JSONResponse({
        "transcription": transcript,
        "reply": reply
    })
