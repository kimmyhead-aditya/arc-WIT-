import sounddevice as sd
import numpy as np
import soundfile as sf

print("Available devices:")
print(sd.query_devices())
print()
print("Default input device:", sd.query_devices(kind='input'))
print()
print("Recording 3 seconds...")

try:
    audio = sd.rec(3 * 16000, samplerate=16000, channels=1, dtype='float32', device=None)
    sd.wait()
    rms = np.sqrt(np.mean(audio**2))
    print(f"Recording done. RMS level: {rms:.6f}")
    if rms < 0.001:
        print("PROBLEM: Audio captured but completely silent. Mic is blocked.")
    else:
        print("SUCCESS: Audio captured correctly. Mic is working.")
    sf.write("mic_test.wav", audio, 16000)
    print("Saved to mic_test.wav")
except Exception as e:
    print(f"FAILED with error: {e}")
