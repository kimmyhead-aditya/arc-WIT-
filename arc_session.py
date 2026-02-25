import csv

def read_session_score(file, column):
    values = []
    with open(file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return sum(values) / len(values)

# Read word-level score
z_score = read_session_score("z_results.csv", "z")

# Read sentence-level score
y_score = read_session_score("y_results.csv", "y")

# Combine into ARC score
arc_score = (0.6 * z_score) + (0.4 * y_score)

print("\n===== ARC FINAL RESULT =====")
print(f"Word score (Z): {z_score:.2f}")
print(f"Sentence score (Y): {y_score:.2f}")
print(f"ARC score: {arc_score:.2f}")
print("============================")


