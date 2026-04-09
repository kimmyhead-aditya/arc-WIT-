import json
import soundfile as sf
from vosk import KaldiRecognizer


def decode_audio(filepath, model, grammar_list=None):
    """
    Unified ASR decoding
    """

    with sf.SoundFile(filepath) as f:

        if grammar_list:
            grammar = json.dumps(grammar_list + ["[unk]"])
            rec = KaldiRecognizer(model, f.samplerate, grammar)
        else:
            rec = KaldiRecognizer(model, f.samplerate)

        rec.SetWords(True)

        transcript = ""

        while True:
            data = f.read(4000, dtype="int16")

            if len(data) == 0:
                break

            data = data.tobytes()

            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                transcript += " " + res.get("text", "")

        final = json.loads(rec.FinalResult())
        transcript += " " + final.get("text", "")

    return transcript.strip()