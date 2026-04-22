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
                    dp[i - 1][j - 1],
                    dp[i][j - 1],
                    dp[i - 1][j]
                )

    return dp[n][m] / max(1, n)


def compute_y(ref, hyp):
    if not hyp.strip():
        return 0.0

    wer = compute_wer(ref, hyp)
    score = (1 - wer) ** 1.5 * 100
    return round(max(0, min(100, score)), 2)


# 🧪 TEST CASES
tests = [
    ("मेरा नाम राहुल है", "मेरा नाम राहुल है"),     # perfect
    ("मेरा नाम राहुल है", "मेरा नाम राहुल"),        # missing word
    ("मेरा नाम राहुल है", "मेरा राहुल है"),         # missing word middle
    ("मेरा नाम राहुल है", "तेरा नाम राहुल है"),     # wrong word
    ("मेरा नाम राहुल है", ""),                      # empty
]

print("\n=== Y SCORE TEST ===\n")

for ref, hyp in tests:
    y = compute_y(ref, hyp)
    print(f"REF: {ref}")
    print(f"HYP: {hyp}")
    print(f"Y: {y}")
    print("------")