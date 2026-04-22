import os
import pandas as pd
from score_z import decode_word
from per_score import compute_per
from dtw_score import compute_dtw


def compute_wer(ref, hyp):
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()

    n = len(ref_words)
    m = len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitute
                    dp[i][j - 1],      # insert
                    dp[i - 1][j]       # delete
                )

    return dp[n][m] / max(1, n)

def score_words_inline(word_dir, words, model):
    import time
    _t0 = time.time()
    results = []

    for i, ref in enumerate(words):
        _loop_start = time.time()

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

        hyp, _ = decode_word(wav_path, model)
        t_decode = time.time() - _loop_start
        _t1 = time.time()

        ref_clean = ref.strip()
        hyp_clean = hyp.strip()

        wer = compute_wer(ref_clean, hyp_clean)
        z_asr = (1 - wer) * 100
        per = compute_per(ref_clean, hyp_clean)
        t_per = time.time() - _t1
        _t2 = time.time()

        ref_audio_path = f"audio_prompts_wav/utt{i+1:02d}.wav"
        if os.path.exists(ref_audio_path):
            dtw = compute_dtw(wav_path, ref_audio_path)
        else:
            dtw = 50
        t_dtw = time.time() - _t2

        z = 0.65 * z_asr + 0.25 * dtw + 0.10 * per
        z = max(0, min(100, z))

        print(f"[TIMING] word '{ref}' → decode:{t_decode:.2f}s  per:{t_per:.2f}s  dtw:{t_dtw:.2f}s  total:{time.time()-_loop_start:.2f}s")

        results.append({
            "reference": ref,
            "hypothesis": hyp,
            "z": z,
            "per": per,
            "dtw": dtw
        })

    print(f"[TIMING] total score_words_inline: {time.time()-_t0:.2f}s")
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