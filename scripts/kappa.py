#!/usr/bin/env python3
"""Cohen's kappa between the root cause labels in results/root_causes/labels.json and a blind second labeling.

  python scripts/kappa.py                      the human relabel in results/root_causes/blind/sheet.csv -> kappa.json
  python scripts/kappa.py --second results/root_causes/blind/llm_labels.csv
                                               another labeling -> kappa_<file name>.json

Who made the second labels is read from the file: a "labeler" column if the file has one (the LLM labeling sets it
to an LLM description), otherwise "human relabel (sheet.csv)". The output records it as second_labeler.
"""
import argparse, csv, json, os
from collections import Counter

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(c):
    c = (c or "").strip().lower()
    if "hash" in c and "salt" in c: return "randomness salted hash"
    if "object hash" in c or "unstable" in c: return "unstable object hash"
    if "order" in c: return "test order dependency"
    if "random" in c: return "randomness unseeded"
    if "floating" in c: return "floating point"
    return c


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0, round(po, 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--second", default=f"{R}/results/root_causes/blind/sheet.csv")
    a = ap.parse_args()
    labels = json.load(open(f"{R}/results/root_causes/labels.json"))
    first = {(r["project"], r["test"]): r for r in labels}
    rows = list(csv.DictReader(open(a.second)))
    for r in rows:  # the LLM file uses category/order_dependent; the human sheet your_category/your_order_dependent
        r.setdefault("your_category", r.get("category", "")); r.setdefault("your_order_dependent", r.get("order_dependent", ""))
    who = rows[0].get("labeler") if rows and rows[0].get("labeler") else "human relabel (sheet.csv)"
    rows = [r for r in rows if r["your_category"].strip()]
    cat, od = [], []
    for r in rows:
        f = first[(r["project"], r["test"])]
        sub = f.get("subcategory", "")
        cat.append((norm(f["category"] + " " + sub if "hash" in sub else f["category"]), norm(r["your_category"])))
        od.append((f["order_dependent"], r["your_order_dependent"].strip().lower()))
    out = {"second_labeler": who, "second_file": os.path.relpath(a.second, R), "labeled": len(rows), "of": len(labels), "category": dict(zip(("kappa", "agreement"), kappa(cat))),
           "order_dependent": dict(zip(("kappa", "agreement"), kappa(od))),
           "disagreements": [{"project": r["project"], "test": r["test"], "first": a, "second": b}
                             for r, (a, b) in zip(rows, cat) if a != b]}
    name = "kappa.json" if a.second.endswith("sheet.csv") else f"kappa_{os.path.basename(a.second).rsplit('.', 1)[0]}.json"
    json.dump(out, open(f"{R}/results/root_causes/blind/{name}", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}))


if __name__ == "__main__":
    main()
