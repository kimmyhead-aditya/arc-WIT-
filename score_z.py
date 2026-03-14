
import os
import csv
import json
import wave
from vosk import Model, KaldiRecognizer
from difflib import SequenceMatcher
from per_score import compute_per
from dtw_score import compute_dtw





# ---------------- CONFIG ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model")
AUDIO_DIR = os.path.join(BASE_DIR, "audio_16k")

REFERENCE_FILE = "references.csv"
WORD_FILE = "words_test.txt"
OUTPUT_FILE = "z_results.csv"

SAMPLE_RATE = 16000

# load word list from master stimulus file
with open(WORD_FILE, encoding="utf-8") as f:
    WORDLIST = [w.strip() for w in f if w.strip()]

# build constrained grammar for Vosk
GRAMMAR = json.dumps([[w] for w in WORDLIST] + [["[unk]"]], ensure_ascii=False)

# ----------------------------------------


def decode_word(wav_path):
    if not os.path.exists(wav_path):
        return "", 0.0

    wf = wave.open(wav_path, "rb")

    assert wf.getnchannels() == 1
    assert wf.getsampwidth() == 2
    assert wf.getframerate() == SAMPLE_RATE

    rec = KaldiRecognizer(model, SAMPLE_RATE, GRAMMAR)
    rec.SetWords(True)
    while True:
        data = wf.readframes(8000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    result = json.loads(rec.FinalResult())

    text = result.get("text", "").strip()
    text = text.split()[0] if text else ""

    confidence = 0.0
    if "confidence" in result:
        confidence = float(result["confidence"])
    elif "result" in result and len(result["result"]) > 0:
        confidence = sum(w["conf"] for w in result["result"]) / len(result["result"])

    del rec
    return text, confidence


if __name__ == "__main__":
    with open(REFERENCE_FILE, newline="", encoding="utf-8") as ref_f, \
        open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out_f:

        reader = csv.DictReader(ref_f)
        writer = csv.DictWriter(
            out_f,
            fieldnames=["utt_id","reference","hypothesis","error_type","z","per","dtw"]
        )
        writer.writeheader()

        for row in reader:
            utt_id = row["utt_id"]
            reference = row["reference"]

            wav_path = os.path.join(AUDIO_DIR, f"{utt_id}.wav")
            hypothesis, confidence = decode_word(wav_path)

            # confidence-based intelligibility
            z = max(0, min(100, confidence * 100))

            per_score = compute_per(reference, hypothesis)

            ref_audio = os.path.join("audio_prompts_wav", f"{utt_id}.wav")
            dtw_score = compute_dtw(ref_audio, wav_path)

            if hypothesis == "":
                error_type = "deletion"
            elif hypothesis == reference:
                error_type = "correct"
            else:
                error_type = "substitution"

            writer.writerow({
                "utt_id": utt_id,
                "reference": reference,
                "hypothesis": hypothesis,
                "error_type": error_type,
                "z": z,
                "per": per_score,
                "dtw": dtw_score
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
