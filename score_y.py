import os
import csv
import json
import wave
from vosk import Model, KaldiRecognizer
from difflib import SequenceMatcher

# CONFIG
MODEL_PATH = "/Users/adityaminhas/Desktop/hindi_asr/vosk-model-small-hi-0.22"
AUDIO_DIR = "audio_sent"
REFERENCE_FILE = "sentences.csv"
OUTPUT_FILE = "y_results.csv"
SAMPLE_RATE = 16000

model = Model(MODEL_PATH)

def sentence_score(ref, hyp):
    return SequenceMatcher(None, ref, hyp).ratio() * 100

def decode_sentence(wav_path):
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    text = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text += result.get("text", "")

    final = json.loads(rec.FinalResult())
    text += final.get("text", "")

    return text.strip()

with open(REFERENCE_FILE, newline="", encoding="utf-8") as ref_f, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out_f:

    reader = csv.DictReader(ref_f)
    writer = csv.DictWriter(
        out_f,
        fieldnames=["utt_id", "reference", "hypothesis", "y"]
    )
    writer.writeheader()

    y_values = []

    for row in reader:
        utt_id = row["utt_id"]
        reference = row["reference"]

        wav_path = os.path.join(AUDIO_DIR, f"{utt_id}.wav")
        hypothesis = decode_sentence(wav_path)

        score = round(sentence_score(reference, hypothesis), 2)
        y_values.append(score)

        writer.writerow({
            "utt_id": utt_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "y": score
        })

session_y = sum(y_values) / len(y_values)

print("\n===== SENTENCE SESSION RESULT =====")
print(f"Sentences tested : {len(y_values)}")
print(f"Session Y score  : {session_y:.2f}")
print("===================================")

