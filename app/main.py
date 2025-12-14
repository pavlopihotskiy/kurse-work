from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tempfile, os

from app.audio_processing import load_audio, extract_mfcc
from app.model import TinyMLModel
from app.config import ALLOWED_EXTENSIONS

app = FastAPI(title="Yes/No Audio Recognition")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = TinyMLModel()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid format")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        audio, sr = load_audio(path)
        features = extract_mfcc(audio, sr)
        label, confidence = model.predict(features)
    finally:
        os.remove(path)

    return {"class": label, "confidence": confidence}
