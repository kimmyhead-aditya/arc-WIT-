from per_score import compute_per

tests = [
    ("आम", "आम"),
    ("घर", "गर"),
    ("दूध", "दू"),
    ("पानी", "पानी"),
]

for ref, hyp in tests:
    score = compute_per(ref, hyp)
    print(f"REF: {ref}  HYP: {hyp}  PER score: {score:.2f}")