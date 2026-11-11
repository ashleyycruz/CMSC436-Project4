"""
CMSC 436 Project 4, Part 2: Build the term document matrix

Run:  python3 tdm_matrix.py
"""
import csv
from feature_vector import main as run_part1


def build_matrix(per_paragraph, feature_vector):
    matrix = []
    for counts in per_paragraph: 
        row = []
        for feature in feature_vector: 
            # Look up this feature's count in this paragraph
            row.append(counts[feature])
        matrix.append(row)
    return matrix


def main():
    paragraphs, stemmed, per_paragraph, feature_vector = run_part1()
    matrix = build_matrix(per_paragraph, feature_vector)
    print(f"\nMatrix dimensions: {len(matrix)} paragraphs x {len(matrix[0])} features")

    # Saves matrix to a CSV
    with open("tdm.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + feature_vector)
        for i, row in enumerate(matrix):
            writer.writerow([f"Paragraph {i+1}"] + row)
    print("Saved matrix to tdm.csv")


    # Returned so Part 3 can do `from tdm_matrix import main` and reuse
    # everything from Part 1 and 2
    return paragraphs, stemmed, per_paragraph, feature_vector, matrix

if __name__ == "__main__":
    main()