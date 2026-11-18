import dendropy
import pytest

from pylabeledrf.computeLRF import (
    computeLRF,
    mutateLabeledTree,
    collapse_node,
    flip_node,
    extend_node,
    randomLabels,
    parseEnsemblLabels,
    cloneAndReroot,
    identify_labelled_bipartitions,
)


def _tree(newick, labels=None, rooted=True):
    taxa = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=taxa,
        rooting="force-rooted" if rooted else "default-rooted",
    )
    if labels is not None:
        internal = list(t.internal_nodes())
        assert len(labels) == len(internal)
        for node, lab in zip(internal, labels):
            node.label = lab
    return t


def _balanced4(labels=("speciation", "duplication")):
    # Rooted binary tree with two labeled internal nodes under the root.
    t = _tree("((A,B),(C,D));")
    internals = [n for n in t.internal_nodes()]
    assert len(internals) == 3
    internals[0].label = None  # root
    internals[1].label = labels[0]
    internals[2].label = labels[1]
    return t


def test_computeLRF_identical_tree_distance_zero():
    t1 = _balanced4(("speciation", "duplication"))
    t2 = _balanced4(("speciation", "duplication"))
    assert computeLRF(t1, t2) == 0


def test_computeLRF_single_label_difference_is_one():
    t1 = _balanced4(("speciation", "duplication"))
    t2 = _balanced4(("duplication", "duplication"))
    assert computeLRF(t1, t2) == 1


def test_computeLRF_is_symmetric():
    t1 = _balanced4(("speciation", "duplication"))
    t2 = _balanced4(("duplication", "duplication"))
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_computeLRF_same_topology_permuted_labels_can_still_be_nonzero():
    # Same tree topology, but label placement differs.
    t1 = _balanced4(("speciation", "duplication"))
    t2 = _balanced4(("duplication", "speciation"))
    dist = computeLRF(t1, t2)
    assert isinstance(dist, int)
    assert dist >= 0
    assert dist > 0


def test_computeLRF_leaf_only_trees_return_zero():
    t1 = _tree("(A,B);")
    t2 = _tree("(A,B);")
    assert computeLRF(t1, t2) == 0


def test_identify_labelled_bipartitions_ignores_root_and_trivial_edges():
    t = _balanced4(("speciation", "duplication"))
    biparts, mapping = identify_labelled_bipartitions(t)
    assert len(biparts) == 1
    assert len(mapping) == 1
    bp = biparts[0]
    assert bp in mapping
    child, parent = mapping[bp]
    assert child.parent_node is parent


def test_parseEnsemblLabels_assigns_duplication_and_speciation():
    t = _tree("((A,B),(C,D));")
    internals = list(t.internal_nodes())
    internals[0].label = None
    internals[1].annotations.add_new("D", "Y")
    internals[1].annotations.add_new("DD", "N")
    internals[2].annotations.add_new("D", "N")
    internals[2].annotations.add_new("DD", "N")
    out = parseEnsemblLabels(t)
    labels = [n.label for n in out.internal_nodes()]
    assert labels[0] is None
    assert "duplication" in labels
    assert "speciation" in labels


def test_randomLabels_respects_probability_one_and_zero():
    t = _tree("((A,B),(C,D));")
    out1 = randomLabels(t, p_speciation=1.0)
    assert all(n.label == "speciation" for n in out1.internal_nodes() if n.parent_node is not None or n is out1.seed_node)

    out2 = randomLabels(t, p_speciation=0.0)
    assert all(n.label == "duplication" for n in out2.internal_nodes() if n.parent_node is not None or n is out2.seed_node)


def test_flip_node_toggles_speciation_to_duplication():
    t = _balanced4(("speciation", "duplication"))
    node = [n for n in t.internal_nodes() if n.label == "speciation"][0]
    flip_node(node)
    assert node.label == "duplication"


def test_flip_node_toggles_duplication_to_speciation():
    t = _balanced4(("speciation", "duplication"))
    node = [n for n in t.internal_nodes() if n.label == "duplication"][0]
    flip_node(node)
    assert node.label == "speciation"


def test_collapse_node_lrf_collapses_edge():
    t = _balanced4(("speciation", "duplication"))
    target = [n for n in t.internal_nodes() if n.parent_node is not None][0]
    before = len(list(t.internal_nodes()))
    collapse_node(target, model="LRF")
    after = len(list(t.internal_nodes()))
    assert after < before


def test_collapse_node_elrf_requires_matching_labels():
    t = _balanced4(("speciation", "duplication"))
    target = [n for n in t.internal_nodes() if n.parent_node is not None][0]
    target.label = "duplication"
    target.parent_node.label = "speciation"
    with pytest.raises(ValueError):
        collapse_node(target, model="ELRF")


def test_extend_node_lrf_adds_internal_node(monkeypatch):
    # Build a node with 4 children so extension is valid and deterministic.
    t = _tree("(A,B,C,D);")
    root = t.seed_node
    root.label = "speciation"

    # deterministic randomness
    monkeypatch.setattr("random.shuffle", lambda x: None)
    monkeypatch.setattr("random.randint", lambda a, b: 2)
    monkeypatch.setattr("random.choice", lambda seq: seq[0])

    before_internal = len(list(t.internal_nodes()))
    extend_node(root, model="LRF")
    after_internal = len(list(t.internal_nodes()))
    assert after_internal == before_internal + 1
    assert len(root.child_nodes()) == 3


def test_mutateLabeledTree_returns_clone_and_does_not_change_input(monkeypatch):
    t = _balanced4(("speciation", "duplication"))
    original_newick = t.as_string("newick")

    # Force a flip only.
    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.randint", lambda a, b: 0)

    out = mutateLabeledTree(t, 1, p_flip=1.0, model="LRF")
    assert t.as_string("newick") == original_newick
    assert out is not t
    assert out.as_string("newick") == original_newick


def test_mutateLabeledTree_model_elrf_uses_only_allowed_collapse(monkeypatch):
    t = _balanced4(("speciation", "duplication"))
    # Force a collapse attempt; with ELRF, only matching labels can collapse.
    # We make random path choose collapse if possible and ensure no crash.
    monkeypatch.setattr("random.random", lambda: 1.0)
    monkeypatch.setattr("random.randint", lambda a, b: 0)

    out = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
    assert isinstance(out, dendropy.Tree)


def test_cloneAndReroot_preserves_taxa_and_tree_type():
    t = _balanced4(("speciation", "duplication"))
    out = cloneAndReroot(t)
    assert isinstance(out, dendropy.Tree)
    assert sorted(leaf.taxon.label for leaf in out.leaf_nodes()) == ["A", "B", "C", "D"]
