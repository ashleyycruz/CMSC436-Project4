# CMSC 436 Project 4, Part 3: Cluster the paragraphs (Kohonen Winner-Take-All)
# Run:  python3 cluster.py (needs numpy)

import contextlib
import csv
import io
import numpy as np
from tdm_matrix import main as run_part2


# passes over all paragraphs
EPOCHS = 60

# starting learning rate
RATE = 0.5

# random starts per k, used to check the clusters are stable
RUNS = 10 


# words printed for each cluster
TOP_WORDS = 10


def kohonen(X, k, rng):

    # One WTA run: each cluster has a prototype
    # only the closest one learns
    protos = X[rng.choice(len(X), k, replace=False)].copy()   # start on k random paragraphs
    steps = EPOCHS * len(X)
    step = 0

    for _ in range(EPOCHS):

        for i in rng.permutation(len(X)):
            rate = RATE * (1 - step / steps)
            step += 1


            winner = ((protos - X[i]) ** 2).sum(axis=1).argmin()
            protos[winner] += rate * (X[i] - protos[winner])  # move winner toward the paragraph


    # each paragraph joins its closest prototype
    labels = ((X[:, None] - protos[None]) ** 2).sum(axis=2).argmin(axis=1)
    error = ((X - protos[labels]) ** 2).sum()

    return labels, error




def agreement(a, b):
    # A fraction of paragraph pairs two runs treat the same
    same_a = a[:, None] == a[None, :]
    same_b = b[:, None] == b[None, :]

    return (same_a == same_b).mean()




def main():


    with contextlib.redirect_stdout(io.StringIO()):    # hide Part 1's long output/print
        _, _, _, features, matrix = run_part2()


    # Scale each paragraph to length 1 so long reviews arent 
    # treated as more important than short ones 
    # just because they have more words
    X = np.array(matrix, dtype=float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    rng = np.random.default_rng(436) # rand seed

    # Part 3a: try several clustesrs and see which is most stable
    print("k error stability (1 = every run gives the same clusters)")
    best_k, best_stability, best_labels = None, -1, None

    for k in range(2, 7):

        runs = sorted((kohonen(X, k, rng) for _ in range(RUNS)), key=lambda r: r[1])
        labels = [r[0] for r in runs]
        pairs = [agreement(labels[i], labels[j]) for i in range(RUNS) for j in range(i)]
        stability = np.mean(pairs)

        print(f"{k}   {runs[0][1]:.2f}   {stability:.2f}")


        if stability > best_stability:
            best_k, best_stability, best_labels = k, stability, labels[0]


    # show the clusters that are the most stable
    print(f"\nChosen k = {best_k}\n")

    for c in range(best_k):
        members = np.where(best_labels == c)[0]
        top = np.argsort(-X[members].mean(axis=0))[:TOP_WORDS]


        print(f"Cluster {c+1}: {len(members)} paragraphs")
        print("  Top words :", ", ".join(features[j] for j in top))
        print("  Paragraphs:", ", ".join(str(i + 1) for i in members), "\n")


    # CSV stuff
    with open("cluster_assignments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Paragraph", "Cluster"])

        for i, c in enumerate(best_labels):
            writer.writerow([i + 1, c + 1])

            
    print("Saved clusters to cluster_assignments.csv")

    return features, X, best_labels


if __name__ == "__main__":
    main()