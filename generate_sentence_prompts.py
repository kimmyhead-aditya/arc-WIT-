import csv
import os
from gtts import gTTS
import subprocess

SENTENCE_FILE = "sentences.csv"
OUTPUT_DIR = "audio_prompts_sent"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Generating sentence prompts...")

with open(SENTENCE_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:

        utt_id = row["utt_id"]
        sentence = row["reference"]

        mp3_file = f"{utt_id}.mp3"
        wav_file = os.path.join(OUTPUT_DIR, f"{utt_id}.wav")

        # Generate TTS
        tts = gTTS(sentence, lang="hi")
        tts.save(mp3_file)

        # Convert to 16k WAV
        subprocess.run([
            "ffmpeg",
            "-loglevel", "quiet",
            "-y",
            "-i", mp3_file,
            "-ar", "16000",
            "-ac", "1",
            wav_file
        ])

        os.remove(mp3_file)

print("Sentence prompts created successfully.")