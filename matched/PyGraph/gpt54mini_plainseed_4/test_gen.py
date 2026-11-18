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


def build_graph_with_triangle():
    g = Graph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", name="e1")
    g.add_edge("B", "C", name="e2")
    g.add_edge("A", "C", name="e3")
    return g


def test_add_vertex_returns_vertex_and_increments_count():
    g = Graph()
    v = g.add_vertex("A")
    assert g.num_vertex() == 1
    assert g.vertex_exists("A") is True
    assert v.get_value() == "A"
    assert g.get_all_vertex()["A"] == v


def test_add_vertex_duplicate_raises():
    g = Graph()
    g.add_vertex("A")
    with pytest.raises(VertexDuplicatedException):
        g.add_vertex("A")


def test_add_edge_and_show_edge_returns_same_edge():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")

    edge = g.add_edge("A", "B", name="ab")
    shown = g.show_edge("A", "B")

    assert g.num_edges() == 1
    assert edge is shown
    assert shown.get_name() == "ab"
    assert g.edge_exists("A", "B") is True


def test_add_edge_with_missing_vertex_raises():
    g = Graph()
    g.add_vertex("A")
    with pytest.raises(VertexNotExistsException):
        g.add_edge("A", "B", name="ab")


def test_add_edge_duplicate_name_raises():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_vertex("C")
    g.add_edge("A", "B", name="same")
    with pytest.raises(EdgeNameExistsException):
        g.add_edge("B", "C", name="same")


def test_delete_edge_removes_edge_and_returns_it():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    edge = g.add_edge("A", "B", name="ab")

    removed = g.delete_edge("A", "B")

    assert removed is edge
    assert g.num_edges() == 0
    assert g.edge_exists("A", "B") is False


def test_delete_nonexistent_edge_raises():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    with pytest.raises(EdgeNotFoundException):
        g.delete_edge("A", "B")


def test_delete_vertex_removes_incident_edges():
    g = build_graph_with_triangle()
    removed = g.delete_vertex("B")

    assert removed.get_value() == "B"
    assert g.vertex_exists("B") is False
    assert g.num_vertex() == 2
    assert g.num_edges() == 1
    assert g.edge_exists("A", "C") is True


def test_vertex_neighbors_and_degree():
    g = build_graph_with_triangle()

    neighbors_a = g.vertex_neighbors("A")
    assert {v.get_value() for v in neighbors_a} == {"B", "C"}
    assert g.vertex_degree("A") == 2


def test_vertex_neighbors_missing_vertex_raises():
    g = Graph()
    g.add_vertex("A")
    with pytest.raises(VertexNotExistsException):
        g.vertex_neighbors("X")


def test_is_vertices_adjacents_and_list_methods():
    g = Graph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", name="ab")

    assert g.is_vertices_adjacents("A", "B") is True
    assert g.is_vertices_adjacents("A", "C") is False
    assert g.list_graph_vertices() == ["A", "B", "C"]
    assert g.list_graph_edges() == ["ab"]


def test_has_cycle_triangle_and_has_loop_false():
    g = build_graph_with_triangle()
    assert g.has_cycle() is True
    assert g.has_loop() is False


def test_simplegraph_rejects_loop_and_parallel_edge():
    g = SimpleGraph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="ab")

    with pytest.raises(LoopDetectedException):
        g.add_edge("A", "A", name="loop")

    with pytest.raises(EdgeDuplicatedException):
        g.add_edge("A", "B", name="ab2")


def test_simplegraph_requires_vertices_before_edge():
    g = SimpleGraph()
    g.add_vertex("A")
    with pytest.raises(VertexNotExistsException):
        g.add_edge("A", "B", name="ab")


def test_adjacency_matrix_reflects_undirected_edges():
    g = build_graph_with_triangle()
    matrix = g.adjacency_matrix()

    assert matrix["A"]["A"] == 0
    assert matrix["A"]["B"] == 1
    assert matrix["B"]["A"] == 1
    assert matrix["B"]["C"] == 1
    assert matrix["C"]["A"] == 1


def test_incidence_list_contains_expected_bindings():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="ab")

    incidence = g.incidence_list()
    assert len(incidence) == 2  # 2 vertices x 1 edge

    tuples = {(item.value, item.edge_name, item.incidence) for item in incidence}
    assert tuples == {("A", "ab", 1), ("B", "ab", 1)}


def test_check_regular_graph_true_and_false_cases():
    regular = Graph()
    for v in ["A", "B", "C"]:
        regular.add_vertex(v)
    regular.add_edge("A", "B", name="ab")
    regular.add_edge("B", "C", name="bc")
    regular.add_edge("A", "C", name="ac")
    assert regular.check_regular_graph() is True

    irregular = Graph()
    for v in ["A", "B", "C"]:
        irregular.add_vertex(v)
    irregular.add_edge("A", "B", name="ab")
    assert irregular.check_regular_graph() is False
