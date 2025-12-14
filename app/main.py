from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os

from app.audio_processing import load_audio, extract_mfcc
from app.model import TinyMLModel
from app.config import ALLOWED_EXTENSIONS

app = FastAPI(
    title="Audio Recognition Web Service",
    description="TinyML-style audio recognition (stub model)",
    version="1.0"
)

model = TinyMLModel()


@app.post("/recognize")
async def recognize_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format"
        )

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio, sr = load_audio(tmp_path)
        features = extract_mfcc(audio, sr)
        predicted_class, confidence = model.predict(features)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Error processing audio file"
        )
    finally:
        os.remove(tmp_path)

    return {
        "class": predicted_class,
        "confidence": confidence
    }
