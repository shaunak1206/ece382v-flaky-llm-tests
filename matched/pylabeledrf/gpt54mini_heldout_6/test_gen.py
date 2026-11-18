import pytest
import dendropy

from pylabeledrf.computeLRF import (
    computeLRF,
    identify_labelled_bipartitions,
    getIslands,
    collapse_node,
    flip_node,
    extend_node,
    mutateLabeledTree,
)


def make_tree(newick, labels=None, rooted=True):
    """
    Create a dendropy tree and optionally assign labels to internal nodes by leafset.
    labels: dict mapping tuple(sorted(leaf labels)) -> node label
    """
    t = dendropy.Tree.get(
        data=newick,
        schema="newick",
        rooting="force-rooted" if rooted else "default-unrooted",
        preserve_underscores=True,
    )
    # Ensure bipartitions can be computed reliably.
    t.is_rooted = rooted
    t.encode_bipartitions()

    if labels:
        for node in t.internal_nodes():
            if node.is_leaf():
                continue
            leafset = tuple(sorted(leaf.taxon.label for leaf in dendropy.Tree(seed_node=node).leaf_nodes()))
            if leafset in labels:
                node.label = labels[leafset]
    # Root label is always None for labeled RF code.
    t.seed_node.label = None
    return t


def leafset_labels(tree):
    out = {}
    for node in tree.internal_nodes():
        if node.is_leaf():
            continue
        leafset = tuple(sorted(leaf.taxon.label for leaf in dendropy.Tree(seed_node=node).leaf_nodes()))
        out[leafset] = node.label
    return out


def build_balanced_4taxa_tree(label_left="speciation", label_right="duplication"):
    return make_tree(
        "((A,B),(C,D));",
        labels={
            ("A", "B"): label_left,
            ("C", "D"): label_right,
        },
        rooted=True,
    )


def build_star_tree(label="speciation"):
    # Root with 4 leaf children.
    return make_tree("(A,B,C,D);", labels={}, rooted=True)


def test_compute_lrf_identical_trees_is_zero():
    t1 = build_balanced_4taxa_tree()
    t2 = build_balanced_4taxa_tree()
    assert computeLRF(t1, t2) == 0


def test_compute_lrf_is_symmetric():
    t1 = build_balanced_4taxa_tree(label_left="speciation", label_right="duplication")
    t2 = build_balanced_4taxa_tree(label_left="duplication", label_right="speciation")
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_compute_lrf_label_mismatch_on_same_topology_gives_positive_distance():
    t1 = build_balanced_4taxa_tree(label_left="speciation", label_right="speciation")
    t2 = build_balanced_4taxa_tree(label_left="duplication", label_right="speciation")
    assert computeLRF(t1, t2) > 0


def test_compute_lrf_same_topology_with_all_internal_labels_flipped_is_positive():
    t1 = build_balanced_4taxa_tree(label_left="speciation", label_right="duplication")
    t2 = build_balanced_4taxa_tree(label_left="duplication", label_right="speciation")
    assert computeLRF(t1, t2) > 0


def test_identify_labelled_bipartitions_finds_two_internal_bipartitions():
    t = build_balanced_4taxa_tree()
    biparts, mapping = identify_labelled_bipartitions(t)
    assert len(biparts) == 2
    assert len(mapping) == 2
    assert all(bp in mapping for bp in biparts)


def test_identify_labelled_bipartitions_mapping_uses_child_parent_pairs():
    t = build_balanced_4taxa_tree()
    biparts, mapping = identify_labelled_bipartitions(t)
    for bp in biparts:
        child, parent = mapping[bp]
        assert child.parent_node is parent
        assert child.bipartition == bp


def test_get_islands_with_no_goodedges_returns_single_island():
    t = build_balanced_4taxa_tree()
    islands, node_island, sizes = getIslands(t, goodedges=set())
    assert len(islands) == 1
    assert sizes == [0, 3] or sizes == [0, 2] or len(sizes) == 2
    assert t.seed_node in node_island


def test_get_islands_with_all_goodedges_splits_into_two_islands():
    t = build_balanced_4taxa_tree()
    biparts, _ = identify_labelled_bipartitions(t)
    islands, node_island, sizes = getIslands(t, goodedges=set(biparts))
    assert len(islands) == 3  # dummy 0th + two islands
    assert sizes[1] >= 1
    assert sizes[2] >= 1
    assert len(node_island) > 0


def test_collapse_node_lrf_reduces_internal_nodes():
    t = build_balanced_4taxa_tree()
    before = len(list(t.internal_nodes()))
    target = next(
        n for n in t.internal_nodes()
        if not n.is_leaf() and n is not t.seed_node and tuple(sorted(leaf.taxon.label for leaf in dendropy.Tree(seed_node=n).leaf_nodes())) == ("A", "B")
    )
    collapse_node(target, model="LRF")
    after = len(list(t.internal_nodes()))
    assert after == before - 1


def test_collapse_node_elrf_requires_matching_labels():
    t = build_balanced_4taxa_tree(label_left="speciation", label_right="duplication")
    target = next(
        n for n in t.internal_nodes()
        if not n.is_leaf() and n is not t.seed_node and tuple(sorted(leaf.taxon.label for leaf in dendropy.Tree(seed_node=n).leaf_nodes())) == ("A", "B")
    )
    with pytest.raises(ValueError):
        collapse_node(target, model="ELRF")


def test_flip_node_toggles_speciation_to_duplication():
    t = build_balanced_4taxa_tree()
    node = next(
        n for n in t.internal_nodes()
        if tuple(sorted(leaf.taxon.label for leaf in dendropy.Tree(seed_node=n).leaf_nodes())) == ("A", "B")
    )
    node.label = "speciation"
    flip_node(node)
    assert node.label == "duplication"


def test_flip_node_toggles_duplication_to_speciation():
    t = build_balanced_4taxa_tree()
    node = next(
        n for n in t.internal_nodes()
        if tuple(sorted(leaf.taxon.label for leaf in dendropy.Tree(seed_node=n).leaf_nodes())) == ("A", "B")
    )
    node.label = "duplication"
    flip_node(node)
    assert node.label == "speciation"


def test_flip_node_raises_on_none_label():
    t = build_balanced_4taxa_tree()
    node = next(
        n for n in t.internal_nodes()
        if tuple(sorted(leaf.taxon.label for leaf in dendropy.Tree(seed_node=n).leaf_nodes())) == ("A", "B")
    )
    node.label = None
    with pytest.raises(ValueError):
        flip_node(node)


def test_extend_node_lrf_adds_one_internal_node(monkeypatch):
    t = make_tree("(A,B,C,D);", rooted=True)
    root = t.seed_node
    root.label = "speciation"
    before_internal = len(list(t.internal_nodes()))
    before_children = len(root.child_nodes())

    monkeypatch.setattr("random.shuffle", lambda x: None)
    monkeypatch.setattr("random.randint", lambda a, b: 2)

    extend_node(root, model="LRF")

    after_internal = len(list(t.internal_nodes()))
    after_children = len(root.child_nodes())
    assert after_internal == before_internal + 1
    assert after_children == 3
    assert any(child.label == "speciation" for child in root.child_nodes())
    assert len(root.child_nodes()) != before_children


def test_extend_node_elrf_keeps_original_label_on_new_node(monkeypatch):
    t = make_tree("(A,B,C,D);", rooted=True)
    root = t.seed_node
    root.label = "duplication"

    monkeypatch.setattr("random.shuffle", lambda x: None)
    monkeypatch.setattr("random.randint", lambda a, b: 2)

    extend_node(root, model="ELRF")

    child_labels = [c.label for c in root.child_nodes() if c is not None]
    assert "duplication" in child_labels


def test_mutate_labeled_tree_forces_flip_branch(monkeypatch):
    t = build_balanced_4taxa_tree()
    original = leafset_labels(t)
    calls = []

    monkeypatch.setattr("pylabeledrf.computeLRF.flip_node", lambda node: calls.append(node))
    monkeypatch.setattr("random.random", lambda: 0.0)  # always flip
    monkeypatch.setattr("random.randint", lambda a, b: 0)

    out = mutateLabeledTree(t, n=3, p_flip=1.0, model="LRF")
    assert out is not t
    assert len(calls) == 3
    assert leafset_labels(t) == original


def test_mutate_labeled_tree_can_force_collapse_branch(monkeypatch):
    t = build_balanced_4taxa_tree()
    collapse_calls = []
    extend_calls = []

    monkeypatch.setattr("random.random", lambda: 0.99)  # never flip
    monkeypatch.setattr("random.randint", lambda a, b: 0)

    def fake_collapse(node, model="LRF"):
        collapse_calls.append((node, model))

    def fake_extend(node, model="LRF"):
        extend_calls.append((node, model))

    monkeypatch.setattr("pylabeledrf.computeLRF.collapse_node", fake_collapse)
    monkeypatch.setattr("pylabeledrf.computeLRF.extend_node", fake_extend)

    mutateLabeledTree(t, n=2, p_flip=0.0, model="LRF")
    assert len(collapse_calls) + len(extend_calls) == 2
    assert len(collapse_calls) >= 1
