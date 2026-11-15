# Blind second labeling (for Cohen's kappa)

`sheet.csv` lists the 78 records of `../labels.json` in a shuffled order, with only the project, the test name, and the
test code (E2 tests). It does not show which model wrote the test, the existing label, or the evidence. The second
labeler fills in `your_category` and `your_order_dependent` (yes or no) for every row, without opening `labels.json`.

Categories (Luo et al. and Gruber et al., as used in the reports):
randomness (unseeded generator), randomness (salted hash), unstable object hash in the code, test order dependency,
time, floating point on a clock value, concurrency, async wait, network, IO, unordered collection, other.

The reproduction logs in `results/matched*/` and `results/od_diagnose/` may be used, since they do not name the model.
`scripts/kappa.py` (to be run after the sheet is filled) compares the two labelings and writes `kappa.json`.
