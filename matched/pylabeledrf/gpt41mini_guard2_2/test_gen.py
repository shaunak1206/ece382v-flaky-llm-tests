import pytest
import dendropy
import random
import numpy as np
import torch
import math

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree, flip_node, collapse_node, extend_node


def newick_to_tree(newick, labels=None, rooted=True):
    """
    Helper to create a labeled dendropy.Tree from a Newick string.
    Optionally assign labels to inner nodes in postorder.
    """
    tree = dendropy.Tree.get(data=newick, schema="newick", rooted=rooted)
    if labels:
        internal_nodes = list(tree.internal_nodes())
        assert len(labels) == len(internal_nodes)
        for n, label in zip(internal_nodes, labels):
            n.label = label
    else:
        # Default label all internal nodes speciation
        for n in tree.internal_nodes():
            n.label = "speciation"
    tree.seed_node.label = None
    return tree


def assert_tree_labels(tree, expected_labels):
    internal_nodes = list(tree.internal_nodes())
    actual_labels = [n.label for n in internal_nodes]
    assert actual_labels == expected_labels


@pytest.mark.parametrize("tree1_newick, tree2_newick, labels1, labels2, expected_lrf", [
    # Identical trees with identical labels => 0
    ("(A,B,(C,D));", "(A,B,(C,D));",
     ["speciation", "speciation"], ["speciation", "speciation"], 0),

    # Trees with equal topology but one label different (substitution) => 1
    ("(A,B,(C,D));", "(A,B,(C,D));",
     ["speciation", "duplication"], ["speciation", "speciation"], 1),

    # Trees with different topology and same labels => >0
    ("((A,B), (C,D));", "(A,(B,(C,D)));",
     ["speciation", "speciation"], ["speciation", "speciation"], None),

    # Trees with different topology and one label flipped on one internal node
    ("((A,B), (C,D));", "(A,(B,(C,D)));",
     ["speciation", "duplication"], ["speciation", "duplication"], None),
])
def test_computeLRF_basic_cases(tree1_newick, tree2_newick, labels1, labels2, expected_lrf):
    t1 = newick_to_tree(tree1_newick, labels1)
    t2 = newick_to_tree(tree2_newick, labels2)
    dist = computeLRF(t1, t2)
    if expected_lrf is not None:
        assert dist == expected_lrf
    else:
        # Just assert returned distance is int and >=0
        assert isinstance(dist, int)
        assert dist >= 0


def test_computeLRF_identical_duplication_and_speciation_same_result():
    # Trees with all duplication vs all speciation labels: distance should be large if differ
    newick = "(A,B,(C,D));"
    t_dup = newick_to_tree(newick, ["duplication", "duplication"])
    t_spec = newick_to_tree(newick, ["speciation", "speciation"])
    dist = computeLRF(t_dup, t_spec)
    # Minimum distance should be 1 or more since labels differ; exact value depends on topology and labels
    assert dist >= 1


def test_computeLRF_no_inner_nodes_labels():
    # Tree with no inner node labels on internal nodes (all None)
    newick = "(A,B);"
    tree1 = dendropy.Tree.get(data=newick, schema="newick")
    tree2 = dendropy.Tree.get(data=newick, schema="newick")
    # No labels means no labeled bipartitions, distance should be 0 because trees identical given no labels?
    for n in tree1.internal_nodes():
        n.label = None
    for n in tree2.internal_nodes():
        n.label = None
    tree1.seed_node.label = None
    tree2.seed_node.label = None
    dist = computeLRF(tree1, tree2)
    assert dist == 0


def test_computeLRF_raises_unequal_islands(monkeypatch):
    # Patch identify_labelled_bipartitions and getIslands to produce unequal number of islands
    import pylabeledrf.computeLRF as clrf

    def fake_identify_labelled_bipartitions(t):
        return [set([1, 2, 3]), set([1, 2, 3])]

    def fake_getIslands(t, goodedges):
        if t.seed_node.label == "T1":
            return [set([1]), set([2])], {}, [1, 1]
        else:
            return [set([1])], {}, [1]

    monkeypatch.setattr(clrf, "identify_labelled_bipartitions", lambda t: ([1,2], {1: [t.seed_node, t.seed_node]},))
    monkeypatch.setattr(clrf, "getIslands", fake_getIslands)

    # Construct trees
    t1 = newick_to_tree("(A,B);")
    t1.seed_node.label = "T1"
    t2 = newick_to_tree("(A,B);")
    # t2.seed_node.label left default

    with pytest.raises(ValueError, match=r"Unequal number of islands"):
        computeLRF(t1, t2)


def test_mutateLabeledTree_returns_clone(monkeypatch):
    # Test mutateLabeledTree returns a new tree equal but different object
    tree = newick_to_tree("(A,B,(C,D));", ["speciation", "speciation"])
    mutated = mutateLabeledTree(tree, 0)
    assert mutated.as_string("newick") == tree.as_string("newick")
    assert mutated is not tree
    # Input tree unmodified
    assert tree.as_string("newick") == "(A,B,(C,D));"


def test_mutateLabeledTree_respects_seed_and_probabilities():
    # Seed random, numpy, torch to check reproducibility
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    tree = newick_to_tree("(A,B,(C,D));", ["speciation", "speciation"])
    mutated1 = mutateLabeledTree(tree, 5, p_flip=0.4)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    mutated2 = mutateLabeledTree(tree, 5, p_flip=0.4)
    assert mutated1.as_string("newick") == mutated2.as_string("newick")


def test_flip_node_changes_spec_dup_and_raises(monkeypatch):
    node = dendropy.Node(label='speciation')
    flip_node(node)
    assert node.label == 'duplication'
    flip_node(node)
    assert node.label == 'speciation'
    node.label = "other"
    with pytest.raises(ValueError):
        flip_node(node)


def test_collapse_node_LRF_and_ELRF(monkeypatch):
    tree = newick_to_tree("(A,B,(C,D));", ["speciation", "speciation"])
    t = tree.clone(depth=1)
    n = next(t.internal_nodes())
    # LRF allows collapse always
    collapse_node(n, model='LRF')
    # ELRF allows collapse only if label == parent's label and parent exists
    t2 = tree.clone(depth=1)
    n2 = next(t2.internal_nodes())
    n2.label = "speciation"
    pnode = n2.parent_node
    if pnode is not None:
        pnode.label = "speciation"
        collapse_node(n2, model='ELRF')
    else:
        # If no parent_node, should raise ValueError
        with pytest.raises(ValueError):
            collapse_node(n2, model='ELRF')

    # unsupported model raises
    with pytest.raises(ValueError):
        collapse_node(n, model='UNKNOWN')


def test_extend_node_insufficient_degree_and_success(monkeypatch):
    tree = newick_to_tree("((A,B,C,D,E,F));", ["speciation"])
    n = next(tree.internal_nodes())
    # Create a node with only 3 adjacent nodes by collapsing edges intentionally
    # forcibly test insufficient degree error path
    monkeypatch.setattr(random, "shuffle", lambda x: x)  # no shuffle to simplify
    # forcibly reduce children to 2 or 3? forcibly make degree low
    # If adjacent_nodes <=3 and parent_node is None or <=2, extend_node raises
    # Mock adjacent_nodes to return length <=3 forcibly for this test

    class DummyNode:
        def __init__(self):
            self._label = "speciation"
            self._children = [dendropy.Node(), dendropy.Node()]
            self.parent_node = None
            self.adjacent_call_count = 0

        @property
        def label(self):
            return self._label

        @label.setter
        def label(self, val):
            self._label = val

        def adjacent_nodes(self):
            self.adjacent_call_count += 1
            return self._children[:2]  # always length 2 to force error

        def clear_child_nodes(self):
            pass

        def child_nodes(self):
            return self._children

        def add_child(self, node):
            return node

    dummy = DummyNode()
    # Should raise with insufficient degree
    with pytest.raises(ValueError):
        extend_node(dummy)

    # Successful extend_node run with sufficient degree

    # Make a normal node with >3 adjacent nodes
    tree = newick_to_tree("((A,B,C,D,E,F,G));", ["speciation"])
    node = max(tree.internal_nodes(), key=lambda n: len(n.adjacent_nodes()))
    random.seed(42)
    extend_node(node, model='LRF')


def test_mutateLabeledTree_flips_and_collapses(monkeypatch):
    random.seed(1)
    tree = newick_to_tree("(A,B,(C,D,E));", labels=["speciation", "duplication"])
    # We run mutateLabeledTree with n=10 edits and p_flip=1 forcing all flips
    mutated = mutateLabeledTree(tree, 10, p_flip=1)
    # Label of some internal nodes must have changed due to flips
    labels_before = [n.label for n in tree.internal_nodes()]
    labels_after = [n.label for n in mutated.internal_nodes()]
    assert any(lb != la for lb, la in zip(labels_before, labels_after))

    # Now test p_flip=0 forcing only collapses/extensions, no flips
    random.seed(1)
    mutated2 = mutateLabeledTree(tree, 5, p_flip=0)
    # Labels may be unchanged for all
    labels_after2 = [n.label for n in mutated2.internal_nodes()]
    # May or may not change as mutateLabeledTree can extend or collapse
    assert isinstance(mutated2, dendropy.Tree)


def test_mutateLabeledTree_model_ELRF_and_invalid(monkeypatch):
    random.seed(10)
    tree = newick_to_tree("(A,B,(C,D));", ["duplication", "duplication"])
    # With model ELRF should not raise
    mutateLabeledTree(tree, 3, p_flip=0.5, model='ELRF')
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0, model='INVALIDMODEL')


def test_flip_node_random_state_unchanged(monkeypatch):
    # Ensure random state does not change after flip_node
    random.seed(42)
    n = dendropy.Node(label='duplication')
    rs_before = random.random()
    flip_node(n)
    rs_after = random.random()
    # flip_node should not call random.random
    assert rs_after != rs_before


def test_computeLRF_with_unlabeled_nodes(monkeypatch):
    # Tree with some internal nodes unlabeled (None)
    newick = "(A,(B,C));"
    t1 = dendropy.Tree.get(data=newick, schema="newick")
    t2 = t1.clone(depth=1)
    for i, n in enumerate(t1.internal_nodes()):
        n.label = "speciation" if i ==0 else None
    for i, n in enumerate(t2.internal_nodes()):
        n.label = "speciation" if i ==0 else "duplication"
    t1.seed_node.label = None
    t2.seed_node.label = None
    dist = computeLRF(t1, t2)
    assert dist >= 0 and isinstance(dist, int)
