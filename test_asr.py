from vosk import Model
from asr import decode_audio

model = Model("model")

test_file = "audio_16k/utt01.wav"  # change if needed

text = decode_audio(test_file, model)

print("OUTPUT:", text)