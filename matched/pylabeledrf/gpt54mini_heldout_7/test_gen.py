import pytest
import dendropy

from pylabeledrf.computeLRF import (
    computeLRF,
    mutateLabeledTree,
    collapse_node,
    flip_node,
    extend_node,
    parseEnsemblLabels,
    randomLabels,
    identify_labelled_bipartitions,
)


def make_tree(newick, rooted=True):
    taxa = dendropy.TaxonNamespace()
    return dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=taxa,
        rooting="force-rooted" if rooted else "default-unrooted",
        preserve_underscores=True,
    )


def labels_by_internal_nodes(tree):
    return [n.label for n in tree.internal_nodes()]


def set_internal_labels(tree, labels):
    nodes = list(tree.internal_nodes())
    assert len(nodes) == len(labels)
    for node, label in zip(nodes, labels):
        node.label = label
    return tree


def test_compute_lrf_identical_trees_is_zero():
    t1 = make_tree("((A,B)X,(C,D)Y)R;")
    t2 = make_tree("((A,B)X,(C,D)Y)R;")
    assert computeLRF(t1, t2) == 0


def test_compute_lrf_is_symmetric():
    t1 = make_tree("((A,B)X,(C,D)Y)R;")
    t2 = make_tree("((A,C)X,(B,D)Y)R;")
    d1 = computeLRF(t1, t2)
    d2 = computeLRF(t2, t1)
    assert d1 == d2


def test_compute_lrf_detects_label_difference_on_same_topology():
    t1 = make_tree("((A,B)X,(C,D)Y)R;")
    t2 = make_tree("((A,B)X,(C,D)Y)R;")
    # Change one internal label only.
    nodes = list(t2.internal_nodes())
    nodes[1].label = "duplication" if nodes[1].label != "duplication" else "speciation"
    assert computeLRF(t1, t2) > 0


def test_compute_lrf_unlabeled_and_labeled_star_like_trees():
    # No internal bipartitions except the root; should be a valid comparison.
    t1 = make_tree("(A,B,C,D)R;")
    t2 = make_tree("(A,B,C,D)R;")
    assert computeLRF(t1, t2) == 0


def test_identify_labelled_bipartitions_returns_expected_counts():
    t = make_tree("((A,B)X,(C,D)Y)R;")
    biparts, mapping = identify_labelled_bipartitions(t)
    assert len(biparts) == 2
    assert len(mapping) == 2
    assert all(b in mapping for b in biparts)


def test_collapse_node_lrf_collapses_without_label_restriction():
    t = make_tree("((A,B)X,C)R;")
    internal = [n for n in t.internal_nodes() if n.parent_node is not None][0]
    before = len(list(t.internal_nodes()))
    collapse_node(internal, model="LRF")
    after = len(list(t.internal_nodes()))
    assert after == before - 1


def test_collapse_node_elrf_requires_matching_labels():
    t = make_tree("((A,B)X,C)R;")
    nodes = list(t.internal_nodes())
    child = [n for n in nodes if n.parent_node is not None][0]
    parent = child.parent_node
    child.label = "speciation"
    parent.label = "duplication"
    with pytest.raises(ValueError):
        collapse_node(child, model="ELRF")


def test_flip_node_switches_speciation_to_duplication():
    t = make_tree("((A,B)X,C)R;")
    n = [n for n in t.internal_nodes() if n.parent_node is not None][0]
    n.label = "speciation"
    flip_node(n)
    assert n.label == "duplication"


def test_flip_node_switches_duplication_to_speciation():
    t = make_tree("((A,B)X,C)R;")
    n = [n for n in t.internal_nodes() if n.parent_node is not None][0]
    n.label = "duplication"
    flip_node(n)
    assert n.label == "speciation"


def test_flip_node_invalid_label_raises():
    t = make_tree("((A,B)X,C)R;")
    n = [n for n in t.internal_nodes() if n.parent_node is not None][0]
    n.label = "unexpected"
    with pytest.raises(ValueError):
        flip_node(n)


def test_extend_node_invalid_model_raises():
    t = make_tree("(((A,B)X,C)Y,D)R;")
    n = [n for n in t.internal_nodes() if n.parent_node is not None and len(n.child_nodes()) > 2][0]
    with pytest.raises(ValueError):
        extend_node(n, model="BAD")


def test_extend_node_insufficient_degree_raises():
    t = make_tree("((A,B)X,C)R;")
    n = [n for n in t.internal_nodes() if n.parent_node is not None][0]
    with pytest.raises(ValueError):
        extend_node(n, model="LRF")


def test_random_labels_assigns_expected_labels_when_monkeypatched(monkeypatch):
    t = make_tree("((A,B)X,(C,D)Y)R;")

    seq = iter([0.1, 0.9, 0.2])  # three internal nodes in traversal order
    monkeypatch.setattr("pylabeledrf.computeLRF.random.random", lambda: next(seq))

    out = randomLabels(t, p_speciation=0.5)
    internal = list(out.internal_nodes())
    assert internal[-1].label is None  # seed node gets reset
    assert [n.label for n in internal[:-1]] == ["speciation", "duplication"]


def test_mutate_labeled_tree_flip_path(monkeypatch):
    # Build a tree where only flipping is chosen.
    t = make_tree("((A,B)X,(C,D)Y)R;")
    original_labels = labels_by_internal_nodes(t)

    monkeypatch.setattr("pylabeledrf.computeLRF.random.random", lambda: 0.0)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.randint", lambda a, b: 0)

    def fail_choice(seq):
        pytest.fail("choice should not be called in flip path")

    def fail_shuffle(seq):
        pytest.fail("shuffle should not be called in flip path")

    monkeypatch.setattr("pylabeledrf.computeLRF.random.choice", fail_choice)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.shuffle", fail_shuffle)

    out = mutateLabeledTree(t, n=1, p_flip=1.0, model="LRF")
    new_labels = labels_by_internal_nodes(out)

    assert original_labels != new_labels
    assert set(new_labels).issubset({"speciation", "duplication", None})


def test_mutate_labeled_tree_invalid_model_in_collapse_branch(monkeypatch):
    # Force the collapse branch and then hit the invalid-model check.
    t = make_tree("((A,B)X,C)R;")

    monkeypatch.setattr("pylabeledrf.computeLRF.random.random", lambda: 0.9)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.randint", lambda a, b: 0)

    with pytest.raises(ValueError):
        mutateLabeledTree(t, n=1, p_flip=0.0, model="BAD")
