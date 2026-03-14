import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def extract_mfcc(filepath):

    y, sr = librosa.load(filepath, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    return mfcc.T


def compute_dtw(reference_audio, patient_audio):

    try:

        ref_mfcc = extract_mfcc(reference_audio)
        pat_mfcc = extract_mfcc(patient_audio)

        distance, path = fastdtw(ref_mfcc, pat_mfcc, dist=euclidean)

        norm_distance = distance / len(path)

        # Convert to similarity using inverse scaling
        score = 100 / (1 + norm_distance / 50)

        return round(score, 2)

    except Exception:
        return 0