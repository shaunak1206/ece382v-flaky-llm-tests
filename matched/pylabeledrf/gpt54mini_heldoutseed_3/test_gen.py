import random

import dendropy
import pytest

from pylabeledrf.computeLRF import (
    cloneAndReroot,
    collapse_node,
    computeLRF,
    extend_node,
    flip_node,
    getIslands,
    identify_labelled_bipartitions,
    mutateLabeledTree,
    ordered_bipartitions,
    parseEnsemblLabels,
    randomLabels,
)


def make_taxa():
    return dendropy.TaxonNamespace(["A", "B", "C", "D"])


def make_tree(newick, rooted=True):
    taxa = make_taxa()
    return dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=taxa,
        rooting="force-rooted" if rooted else "force-unrooted",
    )


def set_internal_labels_by_leafset(tree, mapping):
    tree.encode_bipartitions()
    for node in tree.internal_nodes():
        leaves = tuple(sorted(x.taxon.label for x in node.leaf_nodes()))
        if leaves in mapping:
            node.label = mapping[leaves]


def labeled_balanced_tree(label_left="speciation", label_right="duplication", root_label=None):
    t = make_tree("((A,B),(C,D));", rooted=True)
    set_internal_labels_by_leafset(
        t,
        {
            ("A", "B"): label_left,
            ("C", "D"): label_right,
            ("A", "B", "C", "D"): root_label,
        },
    )
    return t


def test_computeLRF_identical_trees_zero():
    t1 = labeled_balanced_tree()
    t2 = labeled_balanced_tree()
    assert computeLRF(t1, t2) == 0


def test_computeLRF_symmetric_for_different_topologies():
    t1 = labeled_balanced_tree()
    t2 = make_tree("((A,C),(B,D));", rooted=True)
    set_internal_labels_by_leafset(
        t2,
        {
            ("A", "C"): "speciation",
            ("B", "D"): "duplication",
            ("A", "B", "C", "D"): None,
        },
    )
    d12 = computeLRF(t1, t2)
    d21 = computeLRF(t2, t1)
    assert d12 == d21
    assert d12 > 0


def test_computeLRF_same_topology_label_substitution_costs_one():
    t1 = labeled_balanced_tree(label_left="speciation", label_right="duplication")
    t2 = labeled_balanced_tree(label_left="duplication", label_right="duplication")
    assert computeLRF(t1, t2) == 1


def test_computeLRF_returns_zero_for_same_topology_and_same_labels():
    t1 = labeled_balanced_tree(label_left="duplication", label_right="speciation")
    t2 = labeled_balanced_tree(label_left="duplication", label_right="speciation")
    assert computeLRF(t1, t2) == 0


def test_identify_labelled_bipartitions_counts_internal_nontrivial_edges():
    t = labeled_balanced_tree()
    biparts, node_map = identify_labelled_bipartitions(t)
    assert len(biparts) == 2
    assert len(node_map) == 2
    leafsets = {
        tuple(sorted(x.taxon.label for x in node_map[b].leaf_nodes()))
        for b in biparts
    }
    assert leafsets == {("A", "B"), ("C", "D")}


def test_getIslands_single_island_when_all_edges_good():
    t = labeled_balanced_tree()
    biparts, _ = identify_labelled_bipartitions(t)
    islands, node_island, sizes = getIslands(t.clone(), set(biparts))
    assert len(islands) == 2  # includes placeholder at index 0
    assert sizes == [0, 3]
    assert all(v == 1 for k, v in node_island.items() if k.is_internal())


def test_computeLRF_raises_on_mismatched_island_counts(monkeypatch):
    t1 = labeled_balanced_tree()
    t2 = labeled_balanced_tree()
    orig_get_islands = __import__("pylabeledrf.computeLRF", fromlist=["getIslands"]).getIslands

    def fake_get_islands(tree, goodedges):
        islands, node_island, sizes = orig_get_islands(tree, goodedges)
        if tree is t1 or tree.as_string(schema="newick") == t1.as_string(schema="newick"):
            return islands, node_island, sizes + [1]
        return islands, node_island, sizes

    monkeypatch.setattr("pylabeledrf.computeLRF.getIslands", fake_get_islands)
    with pytest.raises(ValueError, match="Unequal number of islands"):
        computeLRF(t1, t2)


def test_mutateLabeledTree_zero_edits_returns_clone_not_same_object():
    t = labeled_balanced_tree()
    mutated = mutateLabeledTree(t, 0)
    assert mutated is not t
    assert mutated.as_string(schema="newick") == t.as_string(schema="newick")


def test_flip_node_flips_speciation_to_duplication_and_back():
    t = make_tree("((A,B),(C,D));", rooted=True)
    node = next(n for n in t.internal_nodes() if len(n.leaf_nodes()) == 2)
    node.label = "speciation"
    flip_node(node)
    assert node.label == "duplication"
    flip_node(node)
    assert node.label == "speciation"


def test_flip_node_raises_for_unexpected_label():
    t = make_tree("((A,B),(C,D));", rooted=True)
    node = next(n for n in t.internal_nodes() if len(n.leaf_nodes()) == 2)
    node.label = "unknown"
    with pytest.raises(ValueError, match="Should not happen"):
        flip_node(node)


def test_collapse_node_elrf_requires_matching_labels():
    t = labeled_balanced_tree()
    node = next(n for n in t.internal_nodes() if tuple(sorted(x.taxon.label for x in n.leaf_nodes())) == ("A", "B"))
    node.parent_node.label = "speciation"
    node.label = "speciation"
    collapse_node(node, model="ELRF")  # should not raise


def test_collapse_node_elrf_rejects_mismatched_labels():
    t = labeled_balanced_tree()
    node = next(n for n in t.internal_nodes() if tuple(sorted(x.taxon.label for x in n.leaf_nodes())) == ("A", "B"))
    node.parent_node.label = "speciation"
    node.label = "duplication"
    with pytest.raises(ValueError, match="cannot be collapsed without a flip"):
        collapse_node(node, model="ELRF")


def test_collapse_node_lrf_always_collapses():
    t = labeled_balanced_tree()
    node = next(n for n in t.internal_nodes() if tuple(sorted(x.taxon.label for x in n.leaf_nodes())) == ("A", "B"))
    before = len(list(t.internal_nodes()))
    collapse_node(node, model="LRF")
    after = len(list(t.internal_nodes()))
    assert after == before - 1


def test_extend_node_with_insufficient_degree_raises():
    t = labeled_balanced_tree()
    node = next(n for n in t.internal_nodes() if tuple(sorted(x.taxon.label for x in n.leaf_nodes())) == ("A", "B"))
    with pytest.raises(ValueError, match="Insufficient degree"):
        extend_node(node, model="LRF")


def test_extend_node_with_invalid_model_raises():
    t = labeled_balanced_tree()
    node = next(n for n in t.internal_nodes() if tuple(sorted(x.taxon.label for x in n.leaf_nodes())) == ("A", "B"))
    with pytest.raises(ValueError, match="unsupported model"):
        extend_node(node, model="BAD")


def test_randomLabels_and_parseEnsemblLabels_work_on_copy(monkeypatch):
    t = labeled_balanced_tree()
    original = t.as_string(schema="newick")

    values = iter([0.0, 0.9, 0.1])
    monkeypatch.setattr(random, "random", lambda: next(values))

    labeled = randomLabels(t, p_speciation=0.5)
    assert t.as_string(schema="newick") == original
    internal = [n for n in labeled.internal_nodes() if n.parent_node is not None]
    assert {n.label for n in internal} <= {"speciation", "duplication", None}

    # build a tiny NHX-like tree with annotations to test parseEnsemblLabels
    taxa = make_taxa()
    nhx = dendropy.Tree.get(data="((A,B),(C,D));", schema="newick", taxon_namespace=taxa, rooting="force-rooted")
    for n in nhx.internal_nodes():
        if n.parent_node is None:
            continue
        n.annotations.add_new("D", "Y" if len(n.leaf_nodes()) == 2 else "N")
    parsed = parseEnsemblLabels(nhx)
    assert parsed is not nhx
    assert parsed.seed_node.label is None
    assert all(n.label == "duplication" for n in parsed.internal_nodes() if n.parent_node is not None)


def test_cloneAndReroot_returns_clone():
    t = labeled_balanced_tree()
    cloned = cloneAndReroot(t)
    assert cloned is not t
    assert sorted(x.taxon.label for x in cloned.leaf_nodes()) == ["A", "B", "C", "D"]
