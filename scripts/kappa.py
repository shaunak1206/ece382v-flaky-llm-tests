#!/usr/bin/env python3
"""Cohen's kappa between the root cause labels in results/root_causes/labels.json and the blind second labeling in
results/root_causes/blind/sheet.csv (see the README there). Writes results/root_causes/blind/kappa.json."""
import csv, json, os
from collections import Counter

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(c):
    c = (c or "").strip().lower()
    if "hash" in c and "salt" in c: return "randomness salted hash"
    if "object hash" in c or "unstable" in c: return "unstable object hash"
    if "order" in c: return "test order dependency"
    if "random" in c: return "randomness unseeded"
    return c


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0, round(po, 3)


def main():
    labels = json.load(open(f"{R}/results/root_causes/labels.json"))
    first = {(r["project"], r["test"]): r for r in labels}
    rows = [r for r in csv.DictReader(open(f"{R}/results/root_causes/blind/sheet.csv")) if r["your_category"].strip()]
    cat, od = [], []
    for r in rows:
        f = first[(r["project"], r["test"])]
        sub = f.get("subcategory", "")
        cat.append((norm(f["category"] + " " + sub if "hash" in sub else f["category"]), norm(r["your_category"])))
        od.append((f["order_dependent"], r["your_order_dependent"].strip().lower()))
    out = {"labeled": len(rows), "of": len(labels), "category": dict(zip(("kappa", "agreement"), kappa(cat))),
           "order_dependent": dict(zip(("kappa", "agreement"), kappa(od))),
           "disagreements": [{"project": r["project"], "test": r["test"], "first": a, "second": b}
                             for r, (a, b) in zip(rows, cat) if a != b]}
    json.dump(out, open(f"{R}/results/root_causes/blind/kappa.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}))


if __name__ == "__main__":
    main()
