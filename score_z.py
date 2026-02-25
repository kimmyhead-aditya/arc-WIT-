
import os
import csv
import json
import wave
from vosk import Model, KaldiRecognizer
from difflib import SequenceMatcher

def intelligibility_score(ref, hyp):
    return SequenceMatcher(None, ref, hyp).ratio() * 100

# ---------------- CONFIG ----------------
MODEL_PATH = "/Users/adityaminhas/Desktop/hindi_asr/vosk-model-small-hi-0.22"
AUDIO_DIR = "audio_16k"
REFERENCsE_FILE = "references.csv"
OUTPUT_FILE = "z_results.csv"
SAMPLE_RATE = 16000

WORDLIST = [
    "कलम", "पानी", "बच्चा", "घर", "किताब",
    "दूध", "फल", "नमक", "सड़क", "मछली",
    "टोपी", "कमरा", "जूता", "लड़का", "चावल"
]

GRAMMAR = json.dumps(WORDLIST, ensure_ascii=False)
# ----------------------------------------


# ---------------- CONFIG ----------------
MODEL_PATH = "/Users/adityaminhas/Desktop/hindi_asr/vosk-model-small-hi-0.22"
AUDIO_DIR = "audio_16k"
REFERENCE_FILE = "references.csv"
OUTPUT_FILE = "z_results.csv"
SAMPLE_RATE = 16000




# ----------------------------------------

model = Model(MODEL_PATH)

def decode_word(wav_path):
    wf = wave.open(wav_path, "rb")

    assert wf.getnchannels() == 1
    assert wf.getsampwidth() == 2
    assert wf.getframerate() == SAMPLE_RATE

    rec = KaldiRecognizer(model, SAMPLE_RATE, GRAMMAR)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    result = json.loads(rec.FinalResult())

    text = result.get("text", "").strip()

    confidence = 0.0
    if "confidence" in result:
        confidence = float(result["confidence"])
    elif "result" in result and len(result["result"]) > 0:
        confidence = sum(w["conf"] for w in result["result"]) / len(result["result"])

    del rec
    return text, confidence



with open(REFERENCE_FILE, newline="", encoding="utf-8") as ref_f, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out_f:

    reader = csv.DictReader(ref_f)
    writer = csv.DictWriter(
        out_f,
        fieldnames=["utt_id", "reference", "hypothesis", "error_type", "z"]
    )
    writer.writeheader()

    for row in reader:
        utt_id = row["utt_id"]
        reference = row["reference"]

        wav_path = os.path.join(AUDIO_DIR, f"{utt_id}.wav")
        hypothesis, confidence = decode_word(wav_path)

        if hypothesis == "":
            error_type = "deletion"
            z = 0
        else:
            score = intelligibility_score(reference, hypothesis)

            if score > 90:
                error_type = "correct"
            else:
                error_type = "substitution"

            z = round(score, 2)

        writer.writerow({
            "utt_id": utt_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "error_type": error_type,
            "z": z
        })
 
                       
           
            
            
        
# -------- SESSION Z AGGREGATION --------

z_values = []

with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        z_values.append(float(row["z"]))

session_z = sum(z_values) / len(z_values)

print("\n===== SESSION RESULT =====")
print(f"Words tested     : {len(z_values)}")
print(f"Session Z score  : {session_z:.2f}")
print("==========================")
