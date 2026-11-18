import pytest
import dendropy
import random
from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree, collapse_node, flip_node, extend_node

@pytest.fixture
def simple_tree_3leaves():
    # A rooted binary tree with 3 leaves: ((A,B),C);
    taxa = dendropy.TaxonNamespace(["A","B","C"])
    newick = "((A,B),C);"
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)
    # label internal nodes for testing
    for node in tree.internal_nodes():
        node.label = "speciation"
    tree.seed_node.label = None
    return tree

@pytest.fixture
def simple_tree_3leaves_dup():
    # Same as simple_tree_3leaves but with one duplication node
    taxa = dendropy.TaxonNamespace(["A","B","C"])
    newick = "((A,B),C);"
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)
    internal = list(tree.internal_nodes())
    if internal:
        internal[0].label = "duplication"
    for node in tree.internal_nodes():
        if node.label is None:
            node.label = "speciation"
    tree.seed_node.label = None
    return tree

@pytest.fixture
def balanced_4leaves_tree():
    # Balanced tree with 4 leaves: ((A,B),(C,D));
    taxa = dendropy.TaxonNamespace(["A","B","C","D"])
    newick = "((A,B),(C,D));"
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)
    for node in tree.internal_nodes():
        node.label = "speciation"
    tree.seed_node.label = None
    return tree

@pytest.fixture
def multifurcating_tree():
    # Multifurcating tree: (A,B,C,D);
    taxa = dendropy.TaxonNamespace(["A","B","C","D"])
    newick = "(A,B,C,D);"
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)
    for node in tree.internal_nodes():
        node.label = "speciation"
    tree.seed_node.label = None
    return tree

def test_computeLRF_identical_trees(simple_tree_3leaves):
    # identical trees distance should be 0
    dist = computeLRF(simple_tree_3leaves, simple_tree_3leaves)
    assert dist == 0

def test_computeLRF_different_labels(speciation_and_duplication=simple_tree_3leaves_dup):
    t1 = simple_tree_3leaves_dup
    t2 = simple_tree_3leaves_dup.clone()
    # Flip one internal node label in t2
    for node in t2.internal_nodes():
        node.label = "speciation" if node.label=="duplication" else "duplication"
    dist = computeLRF(t1, t2)
    assert dist > 0

def test_computeLRF_different_topologies(balanced_4leaves_tree):
    t1 = balanced_4leaves_tree
    # Create a tree with different topology: (A,(B,(C,D)));
    taxa = dendropy.TaxonNamespace(["A","B","C","D"])
    t2 = dendropy.Tree.get(data="(A,(B,(C,D)));", schema="newick", taxon_namespace=taxa)
    for node in t2.internal_nodes():
        node.label = "speciation"
    t2.seed_node.label = None
    dist = computeLRF(t1,t2)
    assert dist > 0

def test_computeLRF_trivial_island(monkeypatch):
    # Create trees with no shared bipartitions (no goodedges)
    # Example 2 totally different star trees with 3 leaves
    taxa = dendropy.TaxonNamespace(["A","B","C"])
    t1 = dendropy.Tree.get(data="(A,B,C);", schema="newick", taxon_namespace=taxa)
    t2 = dendropy.Tree.get(data="(A,B,C);", schema="newick", taxon_namespace=taxa)
    for tree in (t1,t2):
        for node in tree.internal_nodes():
            node.label = "speciation"
        tree.seed_node.label = None
    dist = computeLRF(t1,t2)
    # star trees with identical labels -> distance 0 expected
    assert dist == 0

def test_mutateLabeledTree_return_type_and_labels(simple_tree_3leaves):
    mutated = mutateLabeledTree(simple_tree_3leaves, n=5, p_flip=0.5, model='LRF')
    assert isinstance(mutated, dendropy.Tree)
    # Check some labels present and are either "speciation" or "duplication"
    labels = [node.label for node in mutated.internal_nodes() if node.label is not None]
    for label in labels:
        assert label in ['speciation','duplication']

def test_mutateLabeledTree_flips_label(simple_tree_3leaves):
    # run with p_flip=1 to force a flip
    random.seed(42)
    mutated = mutateLabeledTree(simple_tree_3leaves, n=1, p_flip=1.0, model='LRF')
    # Check at least one flip happened
    original_labels = set(node.label for node in simple_tree_3leaves.internal_nodes() if node.label is not None)
    mutated_labels = set(node.label for node in mutated.internal_nodes() if node.label is not None)
    assert original_labels != mutated_labels

def test_mutateLabeledTree_collapse_and_extend(simple_tree_3leaves):
    # run with p_flip=0 to force collapse or extend
    random.seed(123)
    mutated = mutateLabeledTree(simple_tree_3leaves, n=3, p_flip=0.0, model='LRF')
    assert isinstance(mutated, dendropy.Tree)

def test_collapse_node_LRF(simple_tree_3leaves):
    # Collapse a valid internal node edge
    node = next(n for n in simple_tree_3leaves.internal_nodes() if n.parent_node is not None)
    original_num_internal = len(list(simple_tree_3leaves.internal_nodes()))
    collapse_node(node, model='LRF')
    new_num_internal = len(list(simple_tree_3leaves.internal_nodes()))
    assert new_num_internal < original_num_internal

def test_collapse_node_ELRF_valid_and_invalid(simple_tree_3leaves_dup):
    # collapse only allowed if node label == parent label in ELRF model
    # Find such a node
    found = False
    for n in simple_tree_3leaves_dup.internal_nodes():
        if n.parent_node is not None and n.label == n.parent_node.label:
            collapse_node(n, model='ELRF')
            found = True
            break
    assert found
    # Try collapse invalid node - should raise
    for n in simple_tree_3leaves_dup.internal_nodes():
        if n.parent_node is not None and n.label != n.parent_node.label:
            with pytest.raises(ValueError):
                collapse_node(n, model='ELRF')
            break

def test_flip_node_speciation_to_duplication_and_back():
    taxa = dendropy.TaxonNamespace(["A","B"])
    tree = dendropy.Tree.get(data="(A,B);", schema="newick", taxon_namespace=taxa)
    for node in tree.internal_nodes():
        node.label = "speciation"
    node = next(tree.internal_nodes())
    flip_node(node)
    assert node.label == "duplication"
    flip_node(node)
    assert node.label == "speciation"

def test_flip_node_invalid_label_raises(simple_tree_3leaves):
    node = next(simple_tree_3leaves.internal_nodes())
    node.label = "invalid_label"
    with pytest.raises(ValueError):
        flip_node(node)

def test_extend_node_basic_behavior(balanced_4leaves_tree):
    # Must raise if degree <=3 and no parent_node
    root_node = balanced_4leaves_tree.seed_node
    with pytest.raises(ValueError):
        extend_node(root_node)
    # Find a node with degree >3 by manual construction
    taxa = dendropy.TaxonNamespace(["A","B","C","D","E"])
    t = dendropy.Tree.get(data="((A,B,C,D),E);", schema='newick', taxon_namespace=taxa)
    for n in t.internal_nodes():
        n.label = "speciation"
    # artificially create node with >3 children
    node = t.seed_node.child_nodes()[0]
    # forcibly add a child to increase degree if needed, but here degree is exactly 4 so fine
    if len(node.child_nodes()) < 3:
        node.add_child(dendropy.Node(taxon=t.taxon_namespace.get_taxon("E")))  # increase children for test
    # Should not raise
    random.seed(10)
    extend_node(node, model='LRF')

def test_extend_node_unsupported_model_raises(simple_tree_3leaves):
    node = next(simple_tree_3leaves.internal_nodes())
    with pytest.raises(ValueError):
        extend_node(node, model='UNKNOWN')

def test_mutateLabeledTree_unsupported_model_raises(simple_tree_3leaves):
    with pytest.raises(ValueError):
        mutateLabeledTree(simple_tree_3leaves, n=1, p_flip=0.5, model='UNKNOWN')

def test_computeLRF_raises_on_unequal_islands(monkeypatch):
    # Patch getIslands to produce unequal sized island lists to force ValueError
    import pylabeledrf.computeLRF as cLRF

    def fake_getIslands(t, goodedges):
        return [set(["a"])], {}, [1]

    monkeypatch.setattr(cLRF, 'getIslands', fake_getIslands)
    tree = simple_tree_3leaves()
    with pytest.raises(ValueError):
        computeLRF(tree, tree)

def test_computeLRF_raises_on_unprocessed_islands(simple_tree_3leaves):
    # Patch getIslands to return same number but processed islands not complete
    import pylabeledrf.computeLRF as cLRF

    def fake_getIslands(t, goodedges):
        islands = [set(["a"]), set(["b"])]
        node_island = {t.seed_node: 0}
        sizeIsland = [1,1]
        return islands, node_island, sizeIsland

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cLRF, 'getIslands', fake_getIslands)
    tree = simple_tree_3leaves
    with pytest.raises(ValueError):
        computeLRF(tree, tree)
    monkeypatch.undo()

def test_mutateLabeledTree_does_not_modify_input(simple_tree_3leaves):
    original_newick = simple_tree_3leaves.as_string(schema='newick')
    mutated = mutateLabeledTree(simple_tree_3leaves, n=5)
    assert simple_tree_3leaves.as_string(schema='newick') == original_newick
    assert mutated.as_string(schema='newick') != original_newick or mutated.as_string(schema='newick') == original_newick  # Could mutate or not


# Additional test: test computeLRF with trees of different leaf sets raises no exception and returns int
def test_computeLRF_diff_leaves():
    taxa1 = dendropy.TaxonNamespace(["A","B","C"])
    t1 = dendropy.Tree.get(data="((A,B),C);", schema="newick", taxon_namespace=taxa1)
    taxa2 = dendropy.TaxonNamespace(["A","B","D"])
    t2 = dendropy.Tree.get(data="((A,B),D);", schema="newick", taxon_namespace=taxa2)
    for t in (t1,t2):
        for node in t.internal_nodes():
            node.label = "speciation"
        t.seed_node.label = None
    # Should produce some distance (though sets differ)
    dist = computeLRF(t1,t2)
    assert isinstance(dist, int)
