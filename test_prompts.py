import os
import subprocess
from gtts import gTTS

WORD_FILE = "words_test.txt"

print("Generating prompt audio...")

os.makedirs("audio_prompts", exist_ok=True)
os.makedirs("audio_prompts_wav", exist_ok=True)

words = [w.strip() for w in open(WORD_FILE, encoding="utf-8") if w.strip()]

# generate mp3 prompts
for i, word in enumerate(words):
    mp3 = f"audio_prompts/utt{i+1:02d}.mp3"
    gTTS(word, lang="hi").save(mp3)

print("Converting to WAV...")

# convert mp3 → wav
for f in os.listdir("audio_prompts"):
    if f.endswith(".mp3"):
        inp = os.path.join("audio_prompts", f)
        out = os.path.join("audio_prompts_wav", f.replace(".mp3", ".wav"))

        subprocess.run([
            "ffmpeg",
            "-y",
            "-loglevel",
            "quiet",
            "-i",
            inp,
            "-ar", "16000",
            "-ac", "1",
            out
        ])

print("Swapping folders for test...")

os.rename("audio_16k", "audio_patient")
os.rename("audio_prompts_wav", "audio_16k")

print("Running Z score test...")

subprocess.run([os.sys.executable, "score_z.py"])

print("Restoring patient recordings...")

os.rename("audio_16k", "audio_prompts_wav")
os.rename("audio_patient", "audio_16k")

print("Test complete.")