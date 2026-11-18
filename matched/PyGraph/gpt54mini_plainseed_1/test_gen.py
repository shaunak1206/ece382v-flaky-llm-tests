import pytest

from py_graph_t.Graph import Graph
from py_graph_t.SimpleGraph import SimpleGraph
from py_graph_t.exceptions.SimpleGraphException import (
    VertexNotExistsException,
    EdgeNotFoundException,
    VertexDuplicatedException,
    EdgeNameExistsException,
    EdgeDuplicatedException,
    LoopDetectedException,
)


def build_triangle_graph():
    g = Graph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    g.add_edge("B", "C", "bc")
    g.add_edge("C", "A", "ca")
    return g


def test_add_vertex_returns_vertex_and_registers_it():
    g = Graph()
    v = g.add_vertex("A")
    assert g.vertex_exists("A") is True
    assert g.num_vertex() == 1
    assert v.get_value() == "A"


def test_add_duplicate_vertex_raises():
    g = Graph()
    g.add_vertex("A")
    with pytest.raises(VertexDuplicatedException):
        g.add_vertex("A")


def test_add_edge_and_show_edge_returns_same_edge():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")

    edge = g.add_edge("A", "B", "ab")
    shown = g.show_edge("A", "B")

    assert g.num_edges() == 1
    assert edge == shown
    assert edge.get_name() == "ab"


def test_add_edge_with_missing_vertex_raises():
    g = Graph()
    g.add_vertex("A")
    with pytest.raises(VertexNotExistsException):
        g.add_edge("A", "B", "ab")


def test_add_edge_duplicate_name_raises():
    g = Graph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    with pytest.raises(EdgeNameExistsException):
        g.add_edge("B", "C", "ab")


def test_delete_edge_removes_and_returns_edge():
    g = Graph()
    for v in ["A", "B"]:
        g.add_vertex(v)
    added = g.add_edge("A", "B", "ab")

    removed = g.delete_edge("A", "B")

    assert removed == added
    assert g.num_edges() == 0
    assert g.edge_exists("A", "B") is False


def test_delete_nonexistent_edge_raises():
    g = Graph()
    for v in ["A", "B"]:
        g.add_vertex(v)
    with pytest.raises(EdgeNotFoundException):
        g.delete_edge("A", "B")


def test_delete_vertex_removes_incident_edges():
    g = Graph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    g.add_edge("A", "C", "ac")

    removed = g.delete_vertex("A")

    assert removed.get_value() == "A"
    assert g.vertex_exists("A") is False
    assert g.num_edges() == 0


def test_vertex_neighbors_and_degree():
    g = Graph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    g.add_edge("A", "C", "ac")

    neighbors = g.vertex_neighbors("A")
    assert {v.get_value() for v in neighbors} == {"B", "C"}
    assert g.vertex_degree("A") == 2


def test_is_vertices_adjacents_true_and_false():
    g = Graph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")

    assert g.is_vertices_adjacents("A", "B") is True
    assert g.is_vertices_adjacents("A", "C") is False


def test_list_graph_vertices_and_edges():
    g = Graph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    g.add_edge("B", "C", "bc")

    assert g.list_graph_vertices() == ["A", "B", "C"]
    assert g.list_graph_edges() == ["ab", "bc"]


def test_adjacency_matrix_is_symmetric_and_counts_edges():
    g = Graph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    g.add_edge("A", "B", "ab2")  # allowed in Graph, different name

    matrix = g.adjacency_matrix()

    assert matrix["A"]["A"] == 0
    assert matrix["A"]["B"] == 2
    assert matrix["B"]["A"] == 2
    assert matrix["A"]["C"] == 0


def test_has_cycle_and_has_loop_detects_cycle_and_loop():
    g = Graph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    g.add_edge("B", "C", "bc")
    g.add_edge("C", "A", "ca")
    assert g.has_cycle() is True
    assert g.has_loop() is False


def test_check_regular_graph_true_for_triangle():
    g = build_triangle_graph()
    assert g.check_regular_graph() is True


def test_incidence_list_length_and_values_for_simple_edge():
    g = Graph()
    for v in ["A", "B"]:
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")

    incidence = g.incidence_list()

    assert len(incidence) == 2
    assert {(item.vertex_value, item.edge_name, item.incidence) for item in incidence} == {
        ("A", "ab", 1),
        ("B", "ab", 1),
    }


def test_simplegraph_add_edge_rejects_loop():
    g = SimpleGraph()
    g.add_vertex("A")
    with pytest.raises(LoopDetectedException):
        g.add_edge("A", "A", "aa")


def test_simplegraph_add_edge_rejects_missing_vertex():
    g = SimpleGraph()
    g.add_vertex("A")
    with pytest.raises(VertexNotExistsException):
        g.add_edge("A", "B", "ab")


def test_simplegraph_add_edge_rejects_parallel_edge():
    g = SimpleGraph()
    for v in ["A", "B"]:
        g.add_vertex(v)
    g.add_edge("A", "B", "ab")
    with pytest.raises(EdgeDuplicatedException):
        g.add_edge("B", "A", "ba")
