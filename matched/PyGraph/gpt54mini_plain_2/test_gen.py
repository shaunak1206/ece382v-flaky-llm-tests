import pytest

from py_graph_t.Graph import Graph
from py_graph_t.SimpleGraph import SimpleGraph
from py_graph_t.exceptions.SimpleGraphException import (
    VertexNotExistsException,
    EdgeNotFoundException,
    VertexDuplicatedException,
    EdgeDuplicatedException,
    EdgeNameExistsException,
    LoopDetectedException,
)


def build_graph_with_triangle():
    g = Graph()
    for v in ("A", "B", "C"):
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    g.add_edge("B", "C", "bc")
    g.add_edge("C", "A", "ca")
    return g


def test_add_vertex_returns_vertex_and_stores_it():
    g = Graph()
    v = g.add_vertex("A")
    assert g.vertex_exists("A") is True
    assert g.num_vertex() == 1
    assert v.get_value() == "A"


def test_add_vertex_duplicate_raises():
    g = Graph()
    g.add_vertex("A")
    with pytest.raises(VertexDuplicatedException):
        g.add_vertex("A")


def test_add_edge_requires_existing_vertices():
    g = Graph()
    g.add_vertex("A")
    with pytest.raises(VertexNotExistsException):
        g.add_edge("A", "B", "ab")


def test_add_edge_and_show_edge_work():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    edge = g.add_edge("A", "B", "ab")
    assert g.num_edges() == 1
    assert g.edge_exists("A", "B") is True
    assert g.show_edge("A", "B") == edge
    assert edge.get_name() == "ab"


def test_add_edge_duplicate_name_raises():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_vertex("C")
    g.add_edge("A", "B", "same")
    with pytest.raises(EdgeNameExistsException):
        g.add_edge("B", "C", "same")


def test_delete_edge_removes_edge_and_returns_it():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    edge = g.add_edge("A", "B", "ab")
    removed = g.delete_edge("A", "B")
    assert removed == edge
    assert g.num_edges() == 0
    assert g.edge_exists("A", "B") is False


def test_delete_edge_missing_raises():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    with pytest.raises(EdgeNotFoundException):
        g.delete_edge("A", "B")


def test_delete_vertex_removes_incident_edges():
    g = Graph()
    for v in ("A", "B", "C"):
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    g.add_edge("A", "C", "ac")

    removed = g.delete_vertex("A")
    assert removed.get_value() == "A"
    assert g.vertex_exists("A") is False
    assert g.num_vertex() == 2
    assert g.num_edges() == 0


def test_delete_vertex_missing_raises():
    g = Graph()
    with pytest.raises(VertexNotExistsException):
        g.delete_vertex("missing")


def test_vertex_neighbors_and_degree():
    g = Graph()
    for v in ("A", "B", "C"):
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    g.add_edge("A", "C", "ac")

    neigh_values = {v.get_value() for v in g.vertex_neighbors("A")}
    assert neigh_values == {"B", "C"}
    assert g.vertex_degree("A") == 2
    assert g.is_vertices_adjacents("A", "B") is True
    assert g.is_vertices_adjacents("B", "C") is False


def test_vertex_neighbors_missing_raises():
    g = Graph()
    with pytest.raises(VertexNotExistsException):
        g.vertex_neighbors("A")


def test_has_cycle_triangle_graph_true_and_tree_false():
    cyclic = build_graph_with_triangle()
    assert cyclic.has_cycle() is True

    tree = Graph()
    for v in ("A", "B", "C"):
        tree.add_vertex(v)
    tree.add_edge("A", "B", "ab")
    tree.add_edge("B", "C", "bc")
    assert tree.has_cycle() is False


def test_has_loop_detects_self_loop():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.edges.append(type(g.edges).__mro__[0] if False else None)  # no-op to keep lint-free? 
    # build loop via a raw edge object using the public API
    from py_graph_t.edges.SimpleEdge import SimpleEdge

    g.edges = [SimpleEdge(name="loop", vertex_a=g.vertices["A"], vertex_b=g.vertices["A"])]
    assert g.has_loop() is True
    assert g.has_cycle() is True


def test_check_regular_graph_true_and_false():
    regular = Graph()
    for v in ("A", "B", "C", "D"):
        regular.add_vertex(v)
    regular.add_edge("A", "B", "ab")
    regular.add_edge("A", "C", "ac")
    regular.add_edge("B", "D", "bd")
    regular.add_edge("C", "D", "cd")
    assert regular.check_regular_graph() is True

    non_regular = Graph()
    for v in ("A", "B", "C"):
        non_regular.add_vertex(v)
    non_regular.add_edge("A", "B", "ab")
    assert non_regular.check_regular_graph() is False


def test_adjacency_matrix_and_list_methods():
    g = Graph()
    for v in ("A", "B", "C"):
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    g.add_edge("B", "C", "bc")

    assert g.list_graph_vertices() == ["A", "B", "C"]
    assert g.list_graph_edges() == ["ab", "bc"]

    matrix = g.adjacency_matrix()
    assert matrix["A"]["A"] == 0
    assert matrix["A"]["B"] == 1
    assert matrix["B"]["A"] == 1
    assert matrix["B"]["C"] == 1
    assert matrix["C"]["A"] == 0


def test_incidence_list_has_expected_size_and_zero_entries():
    g = Graph()
    for v in ("A", "B"):
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")

    incidence = g.incidence_list()
    assert len(incidence) == 2  # vertices * edges

    tuples = {(item.value, item.edge_name, item.incidence) for item in incidence}
    assert ("A", "ab", 1) in tuples
    assert ("B", "ab", 1) in tuples


def test_simplegraph_add_edge_rejects_loops_and_duplicates():
    g = SimpleGraph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", "ab")

    with pytest.raises(LoopDetectedException):
        g.add_edge("A", "A", "loop")

    with pytest.raises(EdgeDuplicatedException):
        g.add_edge("B", "A", "ab2")
