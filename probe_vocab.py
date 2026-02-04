#!/usr/bin/env python3

import wave
import json
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)

MODEL_PATH = "/opt/kaldi/egs/vosk-model-small-hi-0.22"
AUDIO_DIR = "audio_16k"
SAMPLE_RATE = 16000

model = Model(MODEL_PATH)

def decode_free(wav_path):
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
    return json.loads(rec.FinalResult()).get("text", "").strip()

for i in range(1, 16):
    utt = f"utt{i:02d}"
    text = decode_free(f"{AUDIO_DIR}/{utt}.wav")
    print(f"{utt}: {text}")
