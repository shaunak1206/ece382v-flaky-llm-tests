import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree

SPEC = "speciation"
DUP = "duplication"


def _tree(newick, tns, label=SPEC, rooted=True):
    prefix = "[&R] " if rooted else "[&U] "
    t = dendropy.Tree.get(data=prefix + newick, schema="newick", taxon_namespace=tns)
    for nd in t.internal_nodes():
        nd.label = label
    if not rooted:
        t.seed_node.label = None
    return t


def _cluster(node):
    if node.is_leaf():
        return frozenset([node.taxon.label])
    result = set()
    for c in node.child_nodes():
        result |= _cluster(c)
    return frozenset(result)


def _signature(tree):
    return {_cluster(n): n.label for n in tree.internal_nodes()}


def _leaves(tree):
    return sorted(n.taxon.label for n in tree.leaf_node_iter())


def _set_label(tree, leaves, label):
    target = frozenset(leaves)
    for nd in tree.internal_nodes():
        if _cluster(nd) == target:
            nd.label = label
            return
    raise AssertionError("no node with leaves %r" % (leaves,))


def _balanced4(label=SPEC):
    tns = dendropy.TaxonNamespace()
    return _tree("((a,b),(c,d));", tns, label)


# ---------------------------------------------------------------- computeLRF


def test_lrf_identical_rooted_trees_is_zero():
    tns = dendropy.TaxonNamespace()
    t1 = _tree("((a,b),(c,d));", tns)
    t2 = _tree("((a,b),(c,d));", tns)
    assert computeLRF(t1, t2) == 0


def test_lrf_tree_against_its_clone_is_zero_and_int():
    tns = dendropy.TaxonNamespace()
    t1 = _tree("(((a,b),c),((d,e),f));", tns)
    _set_label(t1, "ab", DUP)
    result = computeLRF(t1, t1.clone(depth=1))
    assert result == 0
    assert isinstance(result, int)


def test_lrf_single_internal_label_difference_is_one():
    tns = dendropy.TaxonNamespace()
    t1 = _tree("((a,b),(c,d));", tns)
    t2 = _tree("((a,b),(c,d));", tns)
    _set_label(t2, "ab", DUP)
    assert computeLRF(t1, t2) == 1


def test_lrf_root_label_difference_is_one():
    tns = dendropy.TaxonNamespace()
    t1 = _tree("((a,b),(c,d));", tns)
    t2 = _tree("((a,b),(c,d));", tns)
    _set_label(t2, "abcd", DUP)
    assert computeLRF(t1, t2) == 1


def test_lrf_collapse_to_star_with_same_labels_is_two():
    tns = dendropy.TaxonNamespace()
    t1 = _tree("((a,b),(c,d));", tns)
    t2 = _tree("(a,b,c,d);", tns)
    assert computeLRF(t1, t2) == 2


def test_lrf_collapse_to_star_with_different_label_adds_substitution():
    tns = dendropy.TaxonNamespace()
    t1 = _tree("((a,b),(c,d));", tns, SPEC)
    t2 = _tree("(a,b,c,d);", tns, DUP)
    assert computeLRF(t1, t2) == 3


def test_lrf_topology_difference_counts_collapse_and_expansion():
    tns = dendropy.TaxonNamespace()
    t1 = _tree("(((a,b),c),(d,e));", tns)
    t2 = _tree("(((a,c),b),(d,e));", tns)
    assert computeLRF(t1, t2) == 2


def test_lrf_topology_difference_with_island_substitution():
    tns = dendropy.TaxonNamespace()
    t1 = _tree("(((a,b),c),(d,e));", tns, SPEC)
    t2 = _tree("(((a,c),b),(d,e));", tns, SPEC)
    _set_label(t2, "abc", DUP)
    _set_label(t2, "ac", DUP)
    assert computeLRF(t1, t2) == 3


def test_lrf_is_symmetric():
    tns = dendropy.TaxonNamespace()
    balanced = _tree("((a,b),(c,d));", tns, SPEC)
    star = _tree("(a,b,c,d);", tns, DUP)
    labelled = _tree("((a,b),(c,d));", tns, SPEC)
    _set_label(labelled, "cd", DUP)
    assert computeLRF(balanced, star) == computeLRF(star, balanced)
    assert computeLRF(balanced, labelled) == computeLRF(labelled, balanced)

    five_a = _tree("(((a,b),c),(d,e));", tns, SPEC)
    five_b = _tree("(((a,c),b),(d,e));", tns, DUP)
    assert computeLRF(five_a, five_b) == computeLRF(five_b, five_a)


def test_lrf_does_not_modify_input_trees():
    tns = dendropy.TaxonNamespace()
    t1 = _tree("(((a,b),c),(d,e));", tns, SPEC)
    t2 = _tree("(((a,c),b),(d,e));", tns, DUP)
    before1 = _signature(t1)
    before2 = _signature(t2)
    computeLRF(t1, t2)
    assert _signature(t1) == before1
    assert _signature(t2) == before2
    assert _leaves(t1) == ["a", "b", "c", "d", "e"]


def test_lrf_identical_unrooted_trees_is_zero():
    tns = dendropy.TaxonNamespace()
    t1 = _tree("((a,b),c,(d,e));", tns, SPEC, rooted=False)
    t2 = _tree("((a,b),c,(d,e));", tns, SPEC, rooted=False)
    assert computeLRF(t1, t2) == 0


# --------------------------------------------------------- mutateLabeledTree


def test_mutate_zero_edits_returns_equal_independent_copy():
    tree = _balanced4()
    _set_label(tree, "ab", DUP)
    result = mutateLabeledTree(tree, 0)
    assert result is not tree
    assert _signature(result) == _signature(tree)
    assert _leaves(result) == ["a", "b", "c", "d"]


def test_mutate_does_not_change_input_tree():
    tns = dendropy.TaxonNamespace()
    tree = _tree("(((a,b),c),((d,e),f));", tns)
    _set_label(tree, "de", DUP)
    before = _signature(tree)
    for seed in range(8):
        random.seed(seed)
        mutateLabeledTree(tree, 4, p_flip=0.4)
        assert _signature(tree) == before


def test_mutate_flip_only_changes_exactly_one_label_for_one_edit():
    tree = _balanced4()
    before = _signature(tree)
    for seed in range(10):
        random.seed(seed)
        result = mutateLabeledTree(tree, 1, p_flip=1.0)
        after = _signature(result)
        assert set(after) == set(before)
        changed = [k for k in before if before[k] != after[k]]
        assert len(changed) == 1
        assert after[changed[0]] == DUP


def test_mutate_flip_only_keeps_topology_and_valid_labels():
    tns = dendropy.TaxonNamespace()
    tree = _tree("(((a,b),c),((d,e),f));", tns)
    before = _signature(tree)
    random.seed(11)
    result = mutateLabeledTree(tree, 6, p_flip=1.0)
    after = _signature(result)
    assert set(after) == set(before)
    assert all(label in (SPEC, DUP) for label in after.values())


def test_mutate_single_collapse_removes_one_internal_node():
    tns = dendropy.TaxonNamespace()
    tree = _tree("(((a,b),c),(d,e));", tns)
    n_internal = len(tree.internal_nodes())
    for seed in range(6):
        random.seed(seed)
        result = mutateLabeledTree(tree, 1, p_flip=0.0, model="LRF")
        assert len(result.internal_nodes()) == n_internal - 1
        assert _leaves(result) == ["a", "b", "c", "d", "e"]


def test_mutate_single_expansion_of_star_adds_internal_node():
    tns = dendropy.TaxonNamespace()
    tree = _tree("(a,b,c,d);", tns, SPEC)
    for seed in range(6):
        random.seed(seed)
        result = mutateLabeledTree(tree, 1, p_flip=0.0)
        assert len(result.internal_nodes()) == 2
        assert _leaves(result) == ["a", "b", "c", "d"]
        assert all(n.label == SPEC for n in result.internal_nodes())
        new_nodes = [n for n in result.internal_nodes() if n.parent_node is not None]
        assert len(new_nodes) == 1
        assert len(new_nodes[0].child_nodes()) in (2, 3)


def test_mutate_elrf_collapses_nodes_with_matching_parent_label():
    tree = _balanced4(DUP)
    for seed in range(6):
        random.seed(seed)
        result = mutateLabeledTree(tree, 1, p_flip=0.0, model="ELRF")
        assert len(result.internal_nodes()) == 2
        assert _leaves(result) == ["a", "b", "c", "d"]


def test_mutate_elrf_with_no_possible_edit_raises_value_error():
    tree = _balanced4(SPEC)
    _set_label(tree, "abcd", DUP)
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0.0, model="ELRF")


def test_mutate_flip_on_unlabelled_tree_raises_value_error():
    tns = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(
        data="[&R] ((a,b),(c,d));", schema="newick", taxon_namespace=tns
    )
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=1.0)


def test_mutate_unsupported_model_raises_value_error():
    tree = _balanced4(DUP)
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0.0, model="XYZ")


def test_mutate_same_seed_gives_same_result():
    tns = dendropy.TaxonNamespace()
    tree = _tree("(((a,b),c),((d,e),f));", tns)
    random.seed(1234)
    first = mutateLabeledTree(tree, 5, p_flip=0.3)
    random.seed(1234)
    second = mutateLabeledTree(tree, 5, p_flip=0.3)
    assert _signature(first) == _signature(second)


def test_mutate_lrf_preserves_leaves_and_labels_over_many_seeds():
    tns = dendropy.TaxonNamespace()
    tree = _tree("(((a,b),c),((d,e),f));", tns)
    _set_label(tree, "ab", DUP)
    _set_label(tree, "def", DUP)
    for seed in range(20):
        random.seed(seed)
        result = mutateLabeledTree(tree, 6, p_flip=0.3, model="LRF")
        assert _leaves(result) == ["a", "b", "c", "d", "e", "f"]
        assert all(n.label in (SPEC, DUP) for n in result.internal_nodes())


def test_mutate_single_flip_is_lrf_distance_one_from_original():
    tree = _balanced4()
    for seed in range(10):
        random.seed(seed)
        result = mutateLabeledTree(tree, 1, p_flip=1.0)
        assert computeLRF(tree, result) == 1
