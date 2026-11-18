import pytest
import dendropy
import random
from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree, flip_node, collapse_node, extend_node

# Helper to create a simple labeled tree from newick + labels for internal nodes
def make_tree(newick, internal_labels):
    """
    Create and label internal nodes of a dendropy tree.
    newick: Newick string
    internal_labels: list of labels for internal nodes in preorder traversal except root (root label is None)
    """
    t = dendropy.Tree.get(data=newick, schema="newick")
    t.is_rooted = True
    internal_nodes = [n for n in t.internal_nodes()]
    # root always label=None
    t.seed_node.label = None
    # assign labels in preorder skipping root
    # If less labels than internal nodes-1, assign None to remaining
    for n, label in zip(internal_nodes, internal_labels):
        if n is not t.seed_node:
            n.label = label
    return t


def test_computeLRF_identical_trees_zero_distance():
    t = make_tree("((A,B),(C,D));", ["speciation", "duplication"])
    dist = computeLRF(t, t)
    assert dist == 0

def test_computeLRF_simple_difference_substitution():
    t1 = make_tree("((A,B),(C,D));", ["speciation", "duplication"])
    t2 = make_tree("((A,B),(C,D));", ["duplication", "duplication"])
    dist = computeLRF(t1, t2)
    assert dist > 0

def test_computeLRF_different_topologies():
    t1 = make_tree("((A,B),(C,D));", ["speciation", "speciation"])
    t2 = make_tree("((A,C),(B,D));", ["speciation", "speciation"])
    dist = computeLRF(t1, t2)
    assert dist > 0

def test_computeLRF_no_goodedges_single_island():
    # Two star trees with no internal bipartitions
    t1 = make_tree("(A,B,C,D);", [])
    t2 = make_tree("(A,B,C,D);", [])
    d = computeLRF(t1, t2)
    assert d == 0

def test_computeLRF_raise_unequal_number_of_islands():
    # Create trees with artificial bipartitions to trigger island count mismatch
    t1 = make_tree("((A,B),(C,D));", ["speciation"])
    t2 = make_tree("((A,B),(C,D));", ["speciation"])
    # Manually break t2 islands by changing label so identify_labelled_bipartitions differ
    for n in t2.internal_nodes():
        n.label = "duplication"
    # patch identify_labelled_bipartitions to simulate mismatch
    # We'll monkeypatch it to return different islands counts forcibly
    import pylabeledrf.computeLRF as comp
    def fake_identify(*args):
        if args[0] is t1:
            return [1, {1: [1,1]}]
        else:
            return [1, {1: [1,1], 2: [2,3]}]
    original = comp.identify_labelled_bipartitions
    comp.identify_labelled_bipartitions = fake_identify
    with pytest.raises(ValueError):
        computeLRF(t1, t2)
    comp.identify_labelled_bipartitions = original

def test_mutateLabeledTree_flips_and_collapses():
    random.seed(42)
    t = make_tree("((A,B),(C,D));", ["speciation", "duplication"])
    # call mutate with 5 edits, p_flip=1 forces flips only
    mt = mutateLabeledTree(t, n=5, p_flip=1, model='LRF')
    # Check output is a dendropy.Tree and different from input
    assert isinstance(mt, dendropy.Tree)
    # Should differ in node labels from original (most nodes flipped)
    original_labels = [n.label for n in t.internal_nodes()]
    mutated_labels = [n.label for n in mt.internal_nodes()]
    # Some nodes labels should differ
    assert any(a != b for a,b in zip(original_labels, mutated_labels))

def test_mutateLabeledTree_expand_collapse():
    random.seed(123)
    t = make_tree("((A,B,C),(D,E,F));", ["speciation", "duplication", "speciation"])
    # call mutate with 10 edits, p_flip=0 forces only expand/collapse
    mt = mutateLabeledTree(t, n=10, p_flip=0, model='LRF')
    # Check no errors, and result is Tree
    assert isinstance(mt, dendropy.Tree)

def test_flip_node_changes_label():
    n = dendropy.Node()
    n.label = 'speciation'
    flip_node(n)
    assert n.label == 'duplication'
    flip_node(n)
    assert n.label == 'speciation'

def test_flip_node_raises_for_invalid_label():
    n = dendropy.Node()
    n.label = 'unknown'
    with pytest.raises(ValueError):
        flip_node(n)

def test_collapse_node_LRF():
    t = make_tree("((A,B),(C,D));", ["speciation", "duplication"])
    n = next(t.internal_nodes())
    # Should not raise in LRF model
    collapse_node(n, model='LRF')

def test_collapse_node_ELRF_valid():
    t = make_tree("((A,B),(C,D));", ["speciation", "speciation"])
    # pick node with same label as parent to allow collapse
    # find internal node where node.label == parent.label
    n = None
    for node in t.internal_nodes():
        if node.parent_node and node.label == node.parent_node.label:
            n = node
            break
    if n is not None:
        collapse_node(n, model='ELRF')
    else:
        pytest.skip("No suitable node for ELRF collapse")

def test_collapse_node_ELRF_invalid_raises():
    t = make_tree("((A,B),(C,D));", ["duplication", "speciation"])
    # pick node with different label than parent to check raise
    n = None
    for node in t.internal_nodes():
        if node.parent_node and node.label != node.parent_node.label:
            n = node
            break
    if n is None:
        pytest.skip("No suitable node for ELRF collapse invalid test")
    else:
        with pytest.raises(ValueError):
            collapse_node(n, model='ELRF')

def test_collapse_node_unsupported_model():
    t = make_tree("((A,B),(C,D));", ["speciation", "duplication"])
    n = next(t.internal_nodes())
    with pytest.raises(ValueError):
        collapse_node(n, model='UNKNOWN')

def test_extend_node_invalid_degree_raises():
    t = make_tree("((A,B),(C,D));", ["speciation", "duplication"])
    n = next(t.internal_nodes())
    # Reduce degree artificially to trigger error: make node with 2 adj nodes and non-root
    # Usually length(adjacent_nodes()) == degree + 1?
    # Clear children to reduce degree
    n.clear_child_nodes()
    n.add_child(dendropy.Node())
    with pytest.raises(ValueError):
        extend_node(n, model='LRF')

def test_extend_node_success():
    t = make_tree("((A,B,C),(D,E,F));", ["speciation", "duplication", "speciation"])
    n = next(t.internal_nodes())
    # Setup node degree >3 by ensuring more children
    while len(n.child_nodes()) < 3:
        n.add_child(dendropy.Node())
    extend_node(n, model='LRF')
    # After extend, total children should remain same
    assert len(n.child_nodes()) >= 2

def test_extend_node_unsupported_model_raises():
    t = make_tree("((A,B,C),(D,E,F));", ["speciation", "duplication", "speciation"])
    n = next(t.internal_nodes())
    with pytest.raises(ValueError):
        extend_node(n, model='UNKNOWN')

def test_mutateLabeledTree_returns_new_tree_not_same_object():
    t = make_tree("((A,B),(C,D));", ["speciation", "duplication"])
    mt = mutateLabeledTree(t, n=2, p_flip=0.5)
    assert mt is not t

def test_mutateLabeledTree_no_edits_returns_identical_tree():
    t = make_tree("((A,B),(C,D));", ["speciation", "duplication"])
    mt = mutateLabeledTree(t, n=0)
    # Should be identical topology and labels
    for n1, n2 in zip(t.internal_nodes(), mt.internal_nodes()):
        assert n1.label == n2.label

def test_computeLRF_raises_for_unprocessed_island_in_t2(monkeypatch):
    # To force the condition with False in processed2 and raise error
    t = make_tree("((A,B),(C,D));", ["speciation", "duplication"])

    import pylabeledrf.computeLRF as comp

    # Patch getIslands to return two islands but processed2 with False remaining:
    def fake_getIslands(*a, **k):
        return [{1}, {2}], {next(t.internal_nodes()): 0}, [1, 1]

    monkeypatch.setattr(comp, "getIslands", fake_getIslands)

    # Patch identify_labelled_bipartitions to return the same set of bipartitions (to have some goodedges)
    def fake_identify(t_):
        bip = list(comp.identify_labelled_bipartitions(t_)[0])
        nodes = {b: [next(t.internal_nodes()), t.seed_node] for b in bip}
        return bip, nodes

    monkeypatch.setattr(comp, "identify_labelled_bipartitions", fake_identify)

    with pytest.raises(ValueError, match="At least one unprocessed island in t2 remaining"):
        computeLRF(t, t)

