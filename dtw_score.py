import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from functools import lru_cache

def extract_mfcc(filepath):
    y, sr = librosa.load(filepath, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

# Cache reference audio MFCCs — these never change between patients
# so no reason to recompute them every assessment
@lru_cache(maxsize=64)
def _cached_ref_mfcc(reference_audio):
    return extract_mfcc(reference_audio)

def compute_dtw(reference_audio, patient_audio):
    try:
        ref_mfcc = _cached_ref_mfcc(reference_audio)
        pat_mfcc = extract_mfcc(patient_audio)  # patient audio always fresh

        distance, path = fastdtw(ref_mfcc, pat_mfcc, dist=euclidean)
        norm_distance = distance / len(path)

        score = 100 / (1 + norm_distance / 30)
        return round(score, 2)

    except Exception:
        return 0