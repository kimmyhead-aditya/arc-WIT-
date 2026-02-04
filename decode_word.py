import json
import wave
from vosk import Model, KaldiRecognizer

MODEL_PATH = "/opt/kaldi/egs/vosk-model-small-hi-0.22"
GRAMMAR_PATH = "/workspace/z/grammar.json"
WAV_PATH = "/workspace/z/audio/utt1.wav"

# Load grammar
with open(GRAMMAR_PATH, "r", encoding="utf-8") as f:
    grammar = f.read()

# Load model
model = Model(MODEL_PATH)

# Open audio
wf = wave.open(WAV_PATH, "rb")
assert wf.getnchannels() == 1
assert wf.getsampwidth() == 2
assert wf.getframerate() == 16000

# Create recognizer with strict grammar
rec = KaldiRecognizer(model, wf.getframerate(), grammar)
rec.SetWords(True)

# Decode
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    rec.AcceptWaveform(data)

result = json.loads(rec.FinalResult())
print(result.get("text", "").strip())
