import os
import pandas as pd
from score_z import decode_word
from per_score import compute_per
from dtw_score import compute_dtw

def score_words_inline(word_dir, words, model):
    results = []

    for i, ref in enumerate(words):
        wav_path = os.path.join(word_dir, f"utt{i+1:02d}.wav")

        if not os.path.exists(wav_path):
            results.append({
                "reference": ref,
                "hypothesis": "",
                "z": 0,
                "per": 0,
                "dtw": 0
            })
            continue

        hyp, conf = decode_word(wav_path, model)

        z = conf * 100
        per = compute_per(ref, hyp)
        ref_audio_path = f"audio_prompts_wav/utt{i+1:02d}.wav"

        if os.path.exists(ref_audio_path):
            dtw = compute_dtw(wav_path, ref_audio_path)
        else:
            dtw = 0

        results.append({
            "reference": ref,
            "hypothesis": hyp,
            "z": z,
            "per": per,
            "dtw": dtw
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    from vosk import Model
    import os

    WORD_AUDIO_DIR = "audio_16k"
    WORD_FILE = "words_test.txt"

    # load words
    with open(WORD_FILE, encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]

    # load model
    model = Model("model")

    df = score_words_inline(WORD_AUDIO_DIR, words, model)

    print("\n=== SCORING DEBUG OUTPUT ===")
    print(df.head(10))
    print("\nZ mean:", df["z"].mean())
    print("PER mean:", df["per"].mean())
    print("DTW mean:", df["dtw"].mean())