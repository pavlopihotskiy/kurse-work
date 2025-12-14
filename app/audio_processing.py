import librosa
import numpy as np
from app.config import SAMPLE_RATE, N_MFCC


def load_audio(path: str):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio, sr


def extract_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )
    return np.mean(mfcc.T, axis=0).astype(np.float32)
