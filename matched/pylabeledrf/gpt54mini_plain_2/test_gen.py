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
    cloneAndReroot,
)


def make_tree(newick, taxon_namespace=None):
    if taxon_namespace is None:
        taxon_namespace = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=taxon_namespace,
        preserve_underscores=True,
    )
    return tree


def set_root_label_none(tree):
    tree.seed_node.label = None
    return tree


def build_same_tree_pair():
    taxa = dendropy.TaxonNamespace(["A", "B", "C", "D"])
    t1 = make_tree("((A,B)speciation,(C,D)duplication)root;", taxa)
    t2 = make_tree("((A,B)speciation,(C,D)duplication)root;", taxa)
    set_root_label_none(t1)
    set_root_label_none(t2)
    return t1, t2


def build_rotated_pair():
    taxa = dendropy.TaxonNamespace(["A", "B", "C", "D"])
    t1 = make_tree("((A,B)speciation,(C,D)duplication)root;", taxa)
    t2 = make_tree("((B,A)speciation,(D,C)duplication)root;", taxa)
    set_root_label_none(t1)
    set_root_label_none(t2)
    return t1, t2


def build_different_label_pair():
    taxa = dendropy.TaxonNamespace(["A", "B", "C", "D"])
    t1 = make_tree("((A,B)speciation,(C,D)duplication)root;", taxa)
    t2 = make_tree("((A,B)duplication,(C,D)speciation)root;", taxa)
    set_root_label_none(t1)
    set_root_label_none(t2)
    return t1, t2


def build_different_topology_pair():
    taxa = dendropy.TaxonNamespace(["A", "B", "C", "D"])
    t1 = make_tree("((A,B)speciation,(C,D)duplication)root;", taxa)
    t2 = make_tree("((A,C)speciation,(B,D)duplication)root;", taxa)
    set_root_label_none(t1)
    set_root_label_none(t2)
    return t1, t2


def build_star_tree():
    taxa = dendropy.TaxonNamespace(["A", "B", "C", "D"])
    tree = make_tree("(A,B,C,D)root;", taxa)
    tree.seed_node.label = None
    return tree


def build_three_taxa_tree():
    taxa = dendropy.TaxonNamespace(["A", "B", "C"])
    tree = make_tree("((A,B)speciation,C)root;", taxa)
    tree.seed_node.label = None
    return tree


def test_computeLRF_identical_trees_zero_distance():
    t1, t2 = build_same_tree_pair()
    assert computeLRF(t1, t2) == 0


def test_computeLRF_is_symmetric_for_simple_pair():
    t1, t2 = build_different_label_pair()
    d12 = computeLRF(t1, t2)
    d21 = computeLRF(t2, t1)
    assert d12 == d21
    assert d12 >= 0


def test_computeLRF_returns_positive_distance_for_label_swaps():
    t1, t2 = build_different_label_pair()
    assert computeLRF(t1, t2) > 0


def test_computeLRF_returns_positive_distance_for_topology_change():
    t1, t2 = build_different_topology_pair()
    assert computeLRF(t1, t2) > 0


def test_computeLRF_handles_rotations_as_zero_distance():
    t1, t2 = build_rotated_pair()
    assert computeLRF(t1, t2) == 0


def test_computeLRF_star_trees_with_same_taxa_have_zero_distance():
    t1 = build_star_tree()
    t2 = build_star_tree()
    assert computeLRF(t1, t2) == 0


def test_identify_labelled_bipartitions_returns_expected_internal_bipartition_count():
    t1, _ = build_same_tree_pair()
    biparts, mapping = identify_labelled_bipartitions(t1)
    assert len(biparts) == 2
    assert len(mapping) == 2
    labels = sorted(getattr(n, "label", None) for n in t1.internal_nodes() if n is not t1.seed_node)
    assert labels == ["duplication", "speciation"]


def test_cloneAndReroot_preserves_taxa_and_returns_new_tree():
    t1, _ = build_same_tree_pair()
    rooted = cloneAndReroot(t1)
    assert rooted is not t1
    assert sorted(leaf.taxon.label for leaf in rooted.leaf_node_iter()) == ["A", "B", "C", "D"]


def test_randomLabels_assigns_internal_labels_and_root_is_none():
    t1, _ = build_same_tree_pair()
    labeled = randomLabels(t1, p_speciation=1.0)
    assert labeled is not t1
    assert labeled.seed_node.label is None
    for n in labeled.internal_nodes():
        if n is not labeled.seed_node:
            assert n.label == "speciation"


def test_parseEnsemblLabels_uses_annotation_fields():
    taxa = dendropy.TaxonNamespace(["A", "B", "C"])
    tree = make_tree("((A,B),C)root;", taxa)
    for n in tree.internal_nodes():
        if n is tree.seed_node:
            continue
        n.annotations.add_new("D", "Y")
    parsed = parseEnsemblLabels(tree)
    for n in parsed.internal_nodes():
        if n is parsed.seed_node:
            continue
        assert n.label == "duplication"
    assert parsed.seed_node.label is None


def test_flip_node_swaps_speciation_and_duplication():
    taxa = dendropy.TaxonNamespace(["A", "B", "C"])
    tree = make_tree("((A,B)speciation,C)root;", taxa)
    node = next(n for n in tree.internal_nodes() if n is not tree.seed_node)
    flip_node(node)
    assert node.label == "duplication"
    flip_node(node)
    assert node.label == "speciation"


def test_flip_node_raises_on_missing_label():
    taxa = dendropy.TaxonNamespace(["A", "B", "C"])
    tree = make_tree("((A,B),C)root;", taxa)
    node = next(n for n in tree.internal_nodes() if n is not tree.seed_node)
    node.label = None
    with pytest.raises(ValueError):
        flip_node(node)


def test_collapse_node_lrf_collapses_edge():
    t1, _ = build_same_tree_pair()
    node = next(n for n in t1.internal_nodes() if n is not t1.seed_node)
    before = len(list(t1.internal_nodes()))
    collapse_node(node, model="LRF")
    after = len(list(t1.internal_nodes()))
    assert after < before


def test_collapse_node_elrf_requires_same_label():
    taxa = dendropy.TaxonNamespace(["A", "B", "C"])
    tree = make_tree("((A,B)speciation,C)root;", taxa)
    node = next(n for n in tree.internal_nodes() if n is not tree.seed_node)
    node.parent_node.label = "duplication"
    with pytest.raises(ValueError):
        collapse_node(node, model="ELRF")


def test_extend_node_lrf_increases_internal_nodes(monkeypatch):
    taxa = dendropy.TaxonNamespace(["A", "B", "C", "D"])
    tree = make_tree("(((A,B)speciation,C)speciation,D)root;", taxa)
    set_root_label_none(tree)

    node = next(n for n in tree.internal_nodes() if n is not tree.seed_node and len(n.child_nodes()) >= 3)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.shuffle", lambda x: None)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.randint", lambda a, b: 2)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.choice", lambda seq: seq[0])

    before = len(list(tree.internal_nodes()))
    extend_node(node, model="LRF")
    after = len(list(tree.internal_nodes()))
    assert after > before


def test_mutateLabeledTree_flip_only_when_probability_one(monkeypatch):
    t1, _ = build_same_tree_pair()

    monkeypatch.setattr("pylabeledrf.computeLRF.random.random", lambda: 0.0)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.randint", lambda a, b: 0)

    mutated = mutateLabeledTree(t1, n=2, p_flip=1.0, model="LRF")
    assert mutated is not t1
    labels = [n.label for n in mutated.internal_nodes() if n is not mutated.seed_node]
    assert set(labels).issubset({"speciation", "duplication"})
    assert any(lbl != "speciation" for lbl in labels)


def test_mutateLabeledTree_rejects_invalid_model():
    t1, _ = build_same_tree_pair()
    with pytest.raises(ValueError):
        collapse_node(next(n for n in t1.internal_nodes() if n is not t1.seed_node), model="BAD")
    with pytest.raises(ValueError):
        extend_node(next(n for n in t1.internal_nodes() if n is not t1.seed_node), model="BAD")
