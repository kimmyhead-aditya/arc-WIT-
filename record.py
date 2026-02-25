import sounddevice as sd
import soundfile as sf

duration = 4  # seconds
samplerate = 16000
filename = "test_live.wav"

print("Speak now...")

audio = sd.rec(int(duration * samplerate),
               samplerate=samplerate,
               channels=1,
               dtype='int16')

sd.wait()

sf.write(filename, audio, samplerate)

print(f"Recording saved as {filename}")


