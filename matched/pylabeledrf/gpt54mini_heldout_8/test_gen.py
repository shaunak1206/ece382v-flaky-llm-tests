import pytest
import dendropy

from pylabeledrf.computeLRF import (
    computeLRF,
    mutateLabeledTree,
    collapse_node,
    flip_node,
    extend_node,
    randomLabels,
    parseEnsemblLabels,
    identify_labelled_bipartitions,
    ordered_bipartitions,
)


def make_tree(newick, rooted=True):
    taxa = dendropy.TaxonNamespace()
    return dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=taxa,
        rooting="force-rooted" if rooted else "default-rooted",
    )


def label_internal_nodes(tree, labels=None, root_label=None):
    internal = list(tree.internal_nodes())
    if labels is None:
        labels = ["speciation"] * len(internal)
    assert len(labels) >= len(internal)
    for node, lab in zip(internal, labels):
        node.label = lab
    if root_label is not None:
        tree.seed_node.label = root_label
    return tree


def two_internal_tree():
    # Rooted binary tree with 4 leaves and 2 non-root internal nodes
    t = make_tree("((A,B),(C,D));", rooted=True)
    # Set root label to None, and the two child internal nodes labeled
    t.seed_node.label = None
    for n in t.internal_nodes():
        if n is not t.seed_node:
            n.label = "speciation"
    return t


def assert_trees_equal_structure(t1, t2):
    assert t1.as_string(schema="newick") == t2.as_string(schema="newick")


def test_computeLRF_identical_labeled_trees_is_zero():
    t1 = two_internal_tree()
    t2 = two_internal_tree()
    assert computeLRF(t1, t2) == 0


def test_computeLRF_is_symmetric():
    t1 = two_internal_tree()
    t2 = two_internal_tree()
    # change one label in t2
    internal = [n for n in t2.internal_nodes() if n is not t2.seed_node]
    internal[0].label = "duplication"
    d12 = computeLRF(t1, t2)
    d21 = computeLRF(t2, t1)
    assert d12 == d21


def test_computeLRF_same_topology_label_mismatch_positive():
    t1 = two_internal_tree()
    t2 = two_internal_tree()
    internal = [n for n in t2.internal_nodes() if n is not t2.seed_node]
    internal[0].label = "duplication"
    assert computeLRF(t1, t2) > 0


def test_computeLRF_topology_difference_positive():
    t1 = make_tree("((A,B),(C,D));", rooted=True)
    t2 = make_tree("((A,C),(B,D));", rooted=True)
    label_internal_nodes(t1, ["speciation", "speciation", None])
    label_internal_nodes(t2, ["speciation", "speciation", None])
    assert computeLRF(t1, t2) > 0


def test_computeLRF_does_not_mutate_inputs():
    t1 = make_tree("((A,B),(C,D));", rooted=True)
    t2 = make_tree("((A,C),(B,D));", rooted=True)
    label_internal_nodes(t1, ["speciation", "duplication", None])
    label_internal_nodes(t2, ["duplication", "speciation", None])

    before1 = t1.as_string(schema="newick")
    before2 = t2.as_string(schema="newick")
    _ = computeLRF(t1, t2)
    assert t1.as_string(schema="newick") == before1
    assert t2.as_string(schema="newick") == before2


def test_computeLRF_star_trees_without_good_edges():
    # Both trees are stars: no internal bipartitions, one island each
    t1 = make_tree("(A,B,C,D);", rooted=True)
    t2 = make_tree("(A,B,C,D);", rooted=True)
    t1.seed_node.label = None
    t2.seed_node.label = None
    assert computeLRF(t1, t2) == 0


def test_identify_labelled_bipartitions_returns_empty_for_star_tree():
    t = make_tree("(A,B,C,D);", rooted=True)
    t.seed_node.label = None
    biparts, mapping = identify_labelled_bipartitions(t)
    assert biparts == []
    assert mapping == {}


def test_ordered_bipartitions_contains_internal_nodes():
    t = two_internal_tree()
    ob = ordered_bipartitions(t)
    assert len(ob) == len(list(t.internal_nodes()))
    # root should be present, keyed by leaf set
    assert tuple(sorted(["A", "B", "C", "D"])) in ob


def test_flip_node_swaps_speciation_and_duplication():
    t = two_internal_tree()
    n = [x for x in t.internal_nodes() if x is not t.seed_node][0]
    n.label = "speciation"
    flip_node(n)
    assert n.label == "duplication"
    flip_node(n)
    assert n.label == "speciation"


def test_flip_node_raises_on_invalid_label():
    t = two_internal_tree()
    n = [x for x in t.internal_nodes() if x is not t.seed_node][0]
    n.label = "invalid"
    with pytest.raises(ValueError):
        flip_node(n)


def test_collapse_node_lrf_collapses_edge():
    t = make_tree("((A,B),C);", rooted=True)
    label_internal_nodes(t, ["speciation", None])
    internal = [n for n in t.internal_nodes() if n is not t.seed_node][0]
    before = t.as_string(schema="newick")
    collapse_node(internal, model="LRF")
    after = t.as_string(schema="newick")
    assert after != before


def test_collapse_node_elrf_requires_matching_labels():
    t = make_tree("((A,B),C);", rooted=True)
    label_internal_nodes(t, ["speciation", None])
    internal = [n for n in t.internal_nodes() if n is not t.seed_node][0]
    internal.parent_node.label = "duplication"
    with pytest.raises(ValueError):
        collapse_node(internal, model="ELRF")


def test_extend_node_lrf_with_forced_randomness(monkeypatch):
    # Root has 4 children, so extend_node can split off 2 of them.
    t = make_tree("(A,B,C,D);", rooted=True)
    t.seed_node.label = "speciation"

    # force deterministic behavior
    monkeypatch.setattr("random.shuffle", lambda x: None)
    monkeypatch.setattr("random.randint", lambda a, b: 2)

    before_children = len(t.seed_node.child_nodes())
    extend_node(t.seed_node, model="LRF")
    after_children = len(t.seed_node.child_nodes())

    assert after_children == before_children + 1
    assert any(not c.is_leaf() for c in t.seed_node.child_nodes())


def test_extend_node_raises_on_insufficient_degree():
    t = make_tree("((A,B),C);", rooted=True)
    label_internal_nodes(t, ["speciation", None])
    internal = [n for n in t.internal_nodes() if n is not t.seed_node][0]
    with pytest.raises(ValueError):
        extend_node(internal, model="LRF")


def test_mutateLabeledTree_forced_flip_only(monkeypatch):
    t = two_internal_tree()

    # Always choose flip branch, and choose first labeled internal node
    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.randint", lambda a, b: 0)

    before = [n.label for n in t.internal_nodes()]
    mutated = mutateLabeledTree(t, n=1, p_flip=1.0, model="LRF")
    after = [n.label for n in mutated.internal_nodes()]

    assert before != after
    assert set(after) <= {"speciation", "duplication", None}


def test_mutateLabeledTree_forced_collapse_only(monkeypatch):
    t = make_tree("((A,B),C);", rooted=True)
    label_internal_nodes(t, ["speciation", None])

    # Avoid flip branch; force collapse over expand by picking index 0
    monkeypatch.setattr("random.random", lambda: 1.0)
    monkeypatch.setattr("random.randint", lambda a, b: 0)

    before_newick = t.as_string(schema="newick")
    mutated = mutateLabeledTree(t, n=1, p_flip=0.0, model="LRF")
    after_newick = mutated.as_string(schema="newick")

    assert after_newick != before_newick


def test_randomLabels_sets_only_valid_labels_and_keeps_input_unchanged():
    t = two_internal_tree()
    before = t.as_string(schema="newick")

    monkeypatch_values = [0.0, 1.0, 0.2, 0.8]

    def fake_random():
        return monkeypatch_values.pop(0)

    # use monkeypatch via local assignment in this test
    import pylabeledrf.computeLRF as mod

    old_random = mod.random.random
    mod.random.random = fake_random
    try:
        out = randomLabels(t, p_speciation=0.5)
    finally:
        mod.random.random = old_random

    assert t.as_string(schema="newick") == before
    labels = [n.label for n in out.internal_nodes()]
    assert all(lab in {"speciation", "duplication", None} for lab in labels)


def test_parseEnsemblLabels_assigns_labels_from_annotations():
    taxa = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(
        data="((A,B),(C,D));",
        schema="newick",
        taxon_namespace=taxa,
        rooting="force-rooted",
    )
    # Add annotations expected by parseEnsemblLabels
    for n in t.internal_nodes():
        if n is t.seed_node:
            continue
        n.annotations.add_new("D", "Y")
    parsed = parseEnsemblLabels(t)
    non_root_internal = [n for n in parsed.internal_nodes() if n is not parsed.seed_node]
    assert all(n.label == "duplication" for n in non_root_internal)
    assert parsed.seed_node.label is None
