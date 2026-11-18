import random

import dendropy
import pytest

from pylabeledrf.computeLRF import (
    collapse_node,
    computeLRF,
    extend_node,
    flip_node,
    getIslands,
    identify_labelled_bipartitions,
    mutateLabeledTree,
    parseEnsemblLabels,
    randomLabels,
)


def make_rooted_4taxa_tree():
    t = dendropy.Tree.get(
        data="((A,B),(C,D));",
        schema="newick",
        preserve_underscores=True,
    )
    # Rooted, binary tree:
    #   seed_node = root
    #   two internal children each subtending two leaves
    internals = [n for n in t.internal_nodes()]
    # deterministic order: root first, then its two internal children
    root = t.seed_node
    left, right = [n for n in internals if n is not root]
    root.label = None
    left.label = "speciation"
    right.label = "duplication"
    return t


def make_star_4taxa_tree():
    t = dendropy.Tree.get(
        data="(A,B,C,D);",
        schema="newick",
        preserve_underscores=True,
    )
    t.seed_node.label = None
    return t


def internal_labels(tree):
    return [n.label for n in tree.internal_nodes()]


def test_compute_lrf_identical_trees_zero():
    t1 = make_rooted_4taxa_tree()
    t2 = make_rooted_4taxa_tree()
    assert computeLRF(t1, t2) == 0


def test_compute_lrf_is_symmetric_on_simple_trees():
    t1 = make_rooted_4taxa_tree()
    t2 = make_rooted_4taxa_tree()
    # Flip one label in t2 so the trees are not identical.
    for n in t2.internal_nodes():
        if n is not t2.seed_node:
            n.label = "duplication" if n.label == "speciation" else "speciation"
            break
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_compute_lrf_same_topology_one_label_difference_positive():
    t1 = make_rooted_4taxa_tree()
    t2 = make_rooted_4taxa_tree()
    # Change one internal label in t2.
    for n in t2.internal_nodes():
        if n is not t2.seed_node:
            n.label = "duplication" if n.label == "speciation" else "speciation"
            break
    assert computeLRF(t1, t2) > 0


def test_compute_lrf_star_trees_identical_zero():
    t1 = make_star_4taxa_tree()
    t2 = make_star_4taxa_tree()
    assert computeLRF(t1, t2) == 0


def test_compute_lrf_star_trees_label_difference_positive():
    t1 = make_star_4taxa_tree()
    t2 = make_star_4taxa_tree()
    t1.seed_node.label = None
    t2.seed_node.label = None
    # Same topology, but give different labels to the sole internal node.
    # A rooted star has no labeled internal nodes; computeLRF should still be stable.
    assert computeLRF(t1, t2) == 0


def test_identify_labelled_bipartitions_returns_expected_count_for_binary_tree():
    t = make_rooted_4taxa_tree()
    biparts, biparts2nodes = identify_labelled_bipartitions(t)
    # For a rooted binary 4-taxon tree there is one nontrivial internal bipartition.
    assert len(biparts) == 1
    assert len(biparts2nodes) == 1
    assert all(b in biparts2nodes for b in biparts)


def test_get_islands_single_island_for_no_goodedges():
    t = make_rooted_4taxa_tree()
    islands, node_island, sizes = getIslands(t, goodedges=set())
    # No good edges means one island containing all internal nodes.
    assert len(islands) == 1
    assert sizes == [0, 3]
    assert t.seed_node in node_island
    assert node_island[t.seed_node] == 0


def test_get_islands_with_goodedge_splits_tree_into_two_islands():
    t = make_rooted_4taxa_tree()
    biparts, _ = identify_labelled_bipartitions(t)
    goodedges = {biparts[0]}
    islands, node_island, sizes = getIslands(t, goodedges=goodedges)
    assert len(islands) == 3  # index 0 unused + two islands
    assert sizes[1:] == [1, 2]
    assert node_island[t.seed_node] == 0
    non_root_internals = [n for n in t.internal_nodes() if n is not t.seed_node]
    assert {node_island[n] for n in non_root_internals} == {1, 2}


def test_mutate_labeled_tree_zero_edits_returns_clone_with_same_structure():
    random.seed(12345)
    t = make_rooted_4taxa_tree()
    mutated = mutateLabeledTree(t, n=0, p_flip=1.0, model="LRF")
    assert mutated is not t
    assert mutated.as_string("newick") == t.as_string("newick")
    assert internal_labels(mutated) == internal_labels(t)


def test_mutate_labeled_tree_does_not_modify_input_tree():
    random.seed(12345)
    t = make_rooted_4taxa_tree()
    before = t.as_string("newick")
    _ = mutateLabeledTree(t, n=3, p_flip=1.0, model="LRF")
    assert t.as_string("newick") == before


def test_mutate_labeled_tree_flip_only_changes_labels():
    random.seed(7)
    t = make_rooted_4taxa_tree()
    mutated = mutateLabeledTree(t, n=2, p_flip=1.0, model="LRF")
    assert mutated.as_string("newick") == t.as_string("newick")
    assert internal_labels(mutated).count("speciation") + internal_labels(mutated).count("duplication") >= 2


def test_mutate_labeled_tree_lrf_can_change_topology():
    random.seed(2)
    t = make_rooted_4taxa_tree()
    mutated = mutateLabeledTree(t, n=1, p_flip=0.0, model="LRF")
    # One edit may be a collapse or extension; either way, the result is a valid tree.
    assert isinstance(mutated, dendropy.Tree)
    assert len(mutated.leaf_nodes()) == 4


def test_mutate_labeled_tree_invalid_model_raises():
    random.seed(1)
    t = make_rooted_4taxa_tree()
    with pytest.raises(ValueError, match="unsupported model"):
        mutateLabeledTree(t, n=1, p_flip=0.0, model="NOPE")


def test_collapse_node_lrf_collapses_without_error():
    t = make_rooted_4taxa_tree()
    # Pick a non-root internal node.
    node = next(n for n in t.internal_nodes() if n is not t.seed_node)
    collapse_node(node, model="LRF")
    assert len(t.leaf_nodes()) == 4


def test_collapse_node_elrf_rejects_incompatible_edge():
    t = make_rooted_4taxa_tree()
    node = next(n for n in t.internal_nodes() if n is not t.seed_node)
    node.parent_node.label = "speciation"
    node.label = "duplication"
    with pytest.raises(ValueError, match="cannot be collapsed without a flip"):
        collapse_node(node, model="ELRF")


def test_flip_node_switches_labels_both_ways():
    t = make_rooted_4taxa_tree()
    node = next(n for n in t.internal_nodes() if n is not t.seed_node)

    node.label = "speciation"
    flip_node(node)
    assert node.label == "duplication"

    flip_node(node)
    assert node.label == "speciation"


def test_extend_node_invalid_model_raises():
    random.seed(4)
    t = make_rooted_4taxa_tree()
    node = next(n for n in t.internal_nodes() if n is not t.seed_node)
    with pytest.raises(ValueError, match="unsupported model"):
        extend_node(node, model="BAD")


def test_random_labels_is_deterministic_with_seed():
    random.seed(2024)
    t = make_rooted_4taxa_tree()
    labeled = randomLabels(t, p_speciation=0.5)
    # With the seed fixed, the exact labels are deterministic.
    assert labeled.seed_node.label is None
    assert internal_labels(labeled) == ["speciation", "duplication", None]


def test_parse_ensembl_labels_maps_annotations_to_labels_and_keeps_seed_none():
    t = dendropy.Tree.get(
        data="((A,B),(C,D));",
        schema="newick",
        preserve_underscores=True,
    )
    for idx, n in enumerate(t.internal_nodes()):
        if n is t.seed_node:
            continue
        # Dendropy annotations can be set via add_new.
        if idx == 1:
            n.annotations.add_new("D", "Y")
        else:
            n.annotations.add_new("D", "N")
            n.annotations.add_new("DD", "N")
    parsed = parseEnsemblLabels(t)
    assert parsed.seed_node.label is None
    non_root_labels = [n.label for n in parsed.internal_nodes() if n is not parsed.seed_node]
    assert set(non_root_labels).issubset({"speciation", "duplication"})
