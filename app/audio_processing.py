import numpy as np
import librosa
from app.config import SAMPLE_RATE, N_MFCC


def load_audio(file_path: str):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )
    return audio, sr


def extract_mfcc(audio: np.ndarray, sr: int):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.astype(np.float32)
