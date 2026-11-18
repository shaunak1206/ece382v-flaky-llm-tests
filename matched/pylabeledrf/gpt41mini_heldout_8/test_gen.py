import pytest
import dendropy
from pylabeledrf import computeLRF, mutateLabeledTree


def _build_simple_tree(newick, labels):
    """
    Helper to create a Dendropy tree from Newick string and assign inner node labels.
    Leaves are unlabeled; inner nodes are assigned labels from list in preorder.
    """
    tree = dendropy.Tree.get(data=newick, schema="newick")
    inner_nodes = [n for n in tree.internal_nodes() if n != tree.seed_node]
    # Assign labels, matching number of inner_nodes and labels passed (skip seed_node)
    if len(labels) != len(inner_nodes):
        raise ValueError("Labels length must match number of internal (non-root) nodes")
    for node, label in zip(inner_nodes, labels):
        node.label = label
    tree.seed_node.label = None
    return tree


def test_computeLRF_identical_trees():
    newick = "(A,B,(C,D));"
    labels = ["speciation", "duplication"]  # arbitrary
    t1 = _build_simple_tree(newick, labels)
    t2 = _build_simple_tree(newick, labels)
    dist = computeLRF(t1, t2)
    assert dist == 0


def test_computeLRF_different_labels():
    newick = "(A,B,(C,D));"
    t1 = _build_simple_tree(newick, ["speciation", "duplication"])
    t2 = _build_simple_tree(newick, ["duplication", "duplication"])
    dist = computeLRF(t1, t2)
    assert dist > 0


def test_computeLRF_different_topologies():
    t1 = _build_simple_tree("((A,B),C,D);", ["speciation", "duplication"])
    t2 = _build_simple_tree("(A,(B,(C,D)));", ["speciation", "duplication"])
    dist = computeLRF(t1, t2)
    assert dist > 0


def test_computeLRF_with_substitution_needed():
    # Two trees with disjoint label sets in islands should cause substitution increment
    newick = "(A,B,(C,D));"
    t1 = _build_simple_tree(newick, ["speciation", "speciation"])
    t2 = _build_simple_tree(newick, ["duplication", "duplication"])
    dist = computeLRF(t1, t2)
    assert dist >= 1


def test_computeLRF_error_on_unequal_islands(monkeypatch):
    # Patch getIslands to return different number of islands to trigger the ValueError
    import pylabeledrf.computeLRF as computeLRFmod

    def fake_getIslands(t, goodedges):
        # Returns different number of islands for t2 purposely
        if t.seed_node.is_rooted:
            return [set(["a"]), set(["b"])], {t.seed_node: 0}, [1, 1]
        else:
            return [set([1])], {t.seed_node: 0}, [1]

    monkeypatch.setattr(computeLRFmod, "getIslands", fake_getIslands)
    t = _build_simple_tree("(A,B);", [])

    with pytest.raises(ValueError):
        computeLRFmod.computeLRF(t, t)


def test_mutateLabeledTree_flip_only(monkeypatch):
    # Test that mutateLabeledTree flips labels when p_flip=1
    newick = "(A,B,(C,D));"
    tree = _build_simple_tree(newick, ["speciation", "duplication"])

    # Force random.random to always return 0 to trigger flip (because p_flip=0.3, random returns 0 <0.3)
    monkeypatch.setattr("random.random", lambda: 0.0)
    mutated = mutateLabeledTree(tree, n=1, p_flip=0.3, model='LRF')

    # At least one label must have flipped
    labels_orig = [n.label for n in tree.internal_nodes() if n.label is not None]
    labels_mut = [n.label for n in mutated.internal_nodes() if n.label is not None]
    # They can't be all the same (because a flip must have happened)
    assert labels_orig != labels_mut


def test_mutateLabeledTree_expand_or_collapse(monkeypatch):
    # Test expand or collapse path when p_flip=0 (no flip)
    newick = "((A,B),(C,D,E));"
    tree = _build_simple_tree(newick, ["speciation", "speciation", "duplication"])

    random_choices = iter([1, 0])  # Will pick expand or collapse indices

    monkeypatch.setattr("random.random", lambda: 0.99)  # > p_flip triggers collapse/expand
    monkeypatch.setattr("random.randint", lambda a, b=None: next(random_choices))

    mutated = mutateLabeledTree(tree, n=2, p_flip=0, model='LRF')
    # Result should still be a tree with valid inner nodes labeled:
    assert all(n.label in ('speciation', 'duplication', None) for n in mutated.internal_nodes())


def test_mutateLabeledTree_no_side_effects_on_input():
    newick = "(A,B,(C,D));"
    tree = _build_simple_tree(newick, ["speciation", "duplication"])
    tree_labels_before = [n.label for n in tree.internal_nodes()]
    mutated = mutateLabeledTree(tree, n=5)
    # Original tree labels remain unchanged
    tree_labels_after = [n.label for n in tree.internal_nodes()]
    assert tree_labels_before == tree_labels_after
    # Mutated should differ or equal but definitely be a separate instance
    assert mutated is not tree


def test_flip_node_switch_label():
    # Test flip_node switches speciation <-> duplication
    from pylabeledrf.computeLRF import flip_node

    newick = "(A,B,(C,D));"
    tree = _build_simple_tree(newick, ["speciation", "duplication"])

    internal_nodes = [n for n in tree.internal_nodes() if n.label is not None]
    n = internal_nodes[0]
    old_label = n.label
    flip_node(n)
    assert n.label != old_label
    flip_node(n)
    assert n.label == old_label


def test_collapse_node_LRF_and_ELRF_behavior(monkeypatch):
    from pylabeledrf.computeLRF import collapse_node

    newick = "(A,B,(C,D));"
    tree = _build_simple_tree(newick, ["speciation", "duplication"])

    nodes = [n for n in tree.internal_nodes() if n.label is not None]
    node = nodes[0]

    # LRF: always collapsible
    collapse_node(node, model='LRF')

    # ELRF: collapsible only if label matches parent's label
    node = nodes[1]
    node.parent_node.label = node.label
    # No error
    collapse_node(node, model='ELRF')

    # Mismatch raises error
    node.parent_node.label = 'speciation' if node.label != 'speciation' else 'duplication'
    with pytest.raises(ValueError):
        collapse_node(node, model='ELRF')

    with pytest.raises(ValueError):
        collapse_node(node, model='unsupported')


def test_extend_node_valid_and_invalid(monkeypatch):
    from pylabeledrf.computeLRF import extend_node

    newick = "(((A,B),C),D,E);"
    tree = _build_simple_tree(newick, ["speciation", "duplication", "speciation"])

    # Pick node with degree > 3: root's child (usually degree =3 in trifurcation)
    root_first_child = tree.seed_node.child_nodes()[0]

    # We need to create a node with enough children for extend_node to succeed:
    # manually add children to root_first_child to make its degree >3
    # Add two extra leaves as children
    for taxon_label in ["X", "Y"]:
        new_leaf = dendropy.Node(taxon=dendropy.Taxon(label=taxon_label))
        root_first_child.add_child(new_leaf)
    root_first_child.label = 'speciation'

    # Should not raise
    extend_node(root_first_child, model='LRF')

    # Now test node with degree 3 or less raises error
    leaf_node = next(tree.leaf_node_iter())

    with pytest.raises(ValueError):
        extend_node(leaf_node, model='LRF')

    with pytest.raises(ValueError):
        extend_node(root_first_child, model='INVALID')


def test_computeLRF_no_good_edges_single_island(monkeypatch):
    # Patch identify_labelled_bipartitions to return empty sets (no good edges)
    import pylabeledrf.computeLRF as computeLRFmod

    def fake_identify_labelled_bipartitions(t):
        return ([], {})

    monkeypatch.setattr(computeLRFmod, "identify_labelled_bipartitions", fake_identify_labelled_bipartitions)
    newick = "(A,B,C,D);"
    tree1 = _build_simple_tree(newick, ["speciation", "duplication"])
    tree2 = _build_simple_tree(newick, ["duplication", "speciation"])
    dist = computeLRFmod.computeLRF(tree1, tree2)
    assert dist >= 0


def test_mutateLabeledTree_with_ELRF_model(monkeypatch):
    newick = "((A,B),(C,D));"
    tree = _build_simple_tree(newick, ["speciation", "duplication"])
    # Force random.random for deterministic behavior
    monkeypatch.setattr("random.random", lambda: 0.99)
    # Force random.randint to always 0 to collapse
    monkeypatch.setattr("random.randint", lambda a, b=None: 0)
    mutated = mutateLabeledTree(tree, n=1, p_flip=0.0, model='ELRF')
    # Internal nodes should still have labels and not raise errors


def test_flip_node_invalid_label_raises():
    from pylabeledrf.computeLRF import flip_node
    node = dendropy.Node()
    node.label = "invalid_label"
    with pytest.raises(ValueError):
        flip_node(node)


def test_parseEnsemblLabels_assigns_labels():
    from pylabeledrf.computeLRF import parseEnsemblLabels
    newick = "(A,B,(C,D));"
    tree = dendropy.Tree.get(data=newick, schema="newick")
    # Add NHX-like annotations to internal nodes
    for n in tree.internal_nodes():
        n.annotations.add_new("D", "Y")
    parsed = parseEnsemblLabels(tree)
    # Check internal nodes are labeled duplication or speciation
    labels = [n.label for n in parsed.internal_nodes()]
    assert all(l in ('duplication', 'speciation', None) for l in labels)


def test_randomLabels_probability(monkeypatch):
    from pylabeledrf.computeLRF import randomLabels
    newick = "(A,B,(C,D));"
    tree = dendropy.Tree.get(data=newick, schema="newick")

    # Patch random.random to alternate < p_speciation and >= p_speciation
    values = iter([0.1, 0.8, 0.2, 0.9])
    monkeypatch.setattr("random.random", lambda: next(values))
    labeled = randomLabels(tree, p_speciation=0.7)
    labels = [n.label for n in labeled.internal_nodes()]
    assert all(l in ('speciation', 'duplication', None) for l in labels)


def test_ordered_bipartitions_output_type():
    from pylabeledrf.computeLRF import ordered_bipartitions
    newick = "(A,B,(C,D));"
    tree = dendropy.Tree.get(data=newick, schema="newick")
    obs = ordered_bipartitions(tree)
    # Should be dict keyed by tuple of leaf labels, values are nodes
    assert isinstance(obs, dict)
    for k, v in obs.items():
        assert isinstance(k, tuple)
        assert all(isinstance(x, str) for x in k)
        assert isinstance(v, dendropy.Node)


def test_cloneAndReroot_preserves_taxa_labels():
    from pylabeledrf.computeLRF import cloneAndReroot
    newick = "(A,B,(C,D));"
    tree = dendropy.Tree.get(data=newick, schema="newick")
    new_tree = cloneAndReroot(tree)
    orig_taxa = sorted([t.label for t in tree.taxon_namespace])
    new_taxa = sorted([t.label for t in new_tree.taxon_namespace])
    assert orig_taxa == new_taxa
    assert new_tree is not tree
    # Check reroot happened (root's first child is a leaf)
    root_children = list(new_tree.seed_node.child_nodes())
    assert any(c.is_leaf() for c in root_children)

