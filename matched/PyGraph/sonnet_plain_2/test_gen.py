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


def build_simple(values=("a", "b", "c")):
    g = SimpleGraph()
    for v in values:
        g.add_vertex(v)
    return g


def build_graph(values=("a", "b", "c")):
    g = Graph()
    for v in values:
        g.add_vertex(v)
    return g


def test_add_vertex_and_counts():
    g = Graph()
    vertex = g.add_vertex("a")
    assert vertex.value == "a"
    g.add_vertex("b")
    assert g.num_vertex() == 2
    assert g.vertex_exists("a")
    assert not g.vertex_exists("z")
    assert sorted(g.list_graph_vertices()) == ["a", "b"]
    assert set(g.get_all_vertex().keys()) == {"a", "b"}


def test_add_duplicate_vertex_raises():
    g = build_graph(("a",))
    with pytest.raises(VertexDuplicatedException):
        g.add_vertex("a")
    assert g.num_vertex() == 1


def test_graph_add_edge_and_show_edge():
    g = build_graph()
    edge = g.add_edge("a", "b", "e1")
    assert g.num_edges() == 1
    assert g.edge_exists("a", "b")
    assert g.show_edge("a", "b") is edge
    assert g.list_graph_edges() == ["e1"]


def test_graph_add_edge_missing_vertex_raises():
    g = build_graph(("a",))
    with pytest.raises(VertexNotExistsException):
        g.add_edge("a", "zzz", "e1")
    with pytest.raises(VertexNotExistsException):
        g.add_edge("zzz", "a", "e1")
    assert g.num_edges() == 0


def test_graph_add_edge_duplicate_name_raises():
    g = build_graph()
    g.add_edge("a", "b", "e1")
    with pytest.raises(EdgeNameExistsException):
        g.add_edge("b", "c", "e1")
    assert g.num_edges() == 1


def test_show_edge_errors():
    g = build_graph()
    with pytest.raises(VertexNotExistsException):
        g.show_edge("a", "missing")
    with pytest.raises(EdgeNotFoundException):
        g.show_edge("a", "b")


def test_delete_edge():
    g = build_graph()
    g.add_edge("a", "b", "e1")
    removed = g.delete_edge("a", "b")
    assert removed.get_name() == "e1"
    assert g.num_edges() == 0
    assert not g.edge_exists("a", "b")


def test_delete_missing_edge_raises():
    g = build_graph()
    with pytest.raises(EdgeNotFoundException):
        g.delete_edge("a", "b")


def test_delete_vertex_removes_incident_edges():
    g = build_graph(("a", "b", "c", "d"))
    g.add_edge("a", "b", "e1")
    g.add_edge("b", "c", "e2")
    g.add_edge("c", "d", "e3")
    removed = g.delete_vertex("b")
    assert removed.value == "b"
    assert not g.vertex_exists("b")
    assert g.num_vertex() == 3
    assert g.list_graph_edges() == ["e3"]


def test_delete_missing_vertex_raises():
    g = build_graph()
    with pytest.raises(VertexNotExistsException):
        g.delete_vertex("nope")


def test_neighbors_degree_and_adjacency():
    g = build_graph(("a", "b", "c", "d"))
    g.add_edge("a", "b", "e1")
    g.add_edge("a", "c", "e2")
    neighbors = g.vertex_neighbors("a")
    assert sorted(v.value for v in neighbors) == ["b", "c"]
    assert g.vertex_degree("a") == 2
    assert g.vertex_degree("d") == 0
    assert g.is_vertices_adjacents("a", "b")
    assert not g.is_vertices_adjacents("a", "d")
    with pytest.raises(VertexNotExistsException):
        g.vertex_neighbors("missing")
    with pytest.raises(VertexNotExistsException):
        g.vertex_degree("missing")


def test_is_terminal():
    g = build_graph()
    edge = g.add_edge("a", "b", "e1")
    assert g.is_terminal(edge, "a")
    assert g.is_terminal(edge, "b")
    assert not g.is_terminal(edge, "c")


def test_has_cycle_triangle_and_path():
    g = build_simple()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    assert g.has_cycle() is False
    g.add_edge("c", "a")
    assert g.has_cycle() is True


def test_check_regular_graph():
    g = build_simple()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    assert g.check_regular_graph() is False
    g.add_edge("c", "a")
    assert g.check_regular_graph() is True


def test_adjacency_matrix():
    g = build_simple()
    g.add_edge("a", "b")
    matrix = g.adjacency_matrix()
    assert matrix["a"]["b"] == 1
    assert matrix["b"]["a"] == 1
    assert matrix["a"]["c"] == 0
    assert matrix["a"]["a"] == 0
    assert set(matrix.keys()) == {"a", "b", "c"}


def test_incidence_list_size_and_str():
    g = build_graph()
    assert str(g) == ""
    g.add_edge("a", "b", "e1")
    g.add_edge("b", "c", "e2")
    assert len(g.incidence_list()) == 3 * 2
    text = str(g)
    assert text.endswith("\n")
    assert len(text.splitlines()) == 2


def test_simple_graph_add_edge_returns_edge():
    g = build_simple()
    edge = g.add_edge("a", "b", "e1")
    assert g.num_edges() == 1
    assert g.edge_exists("a", "b")
    assert g.show_edge("a", "b") is edge
    assert g.has_loop() is False


def test_simple_graph_rejects_loop_and_duplicate():
    g = build_simple()
    with pytest.raises(LoopDetectedException):
        g.add_edge("a", "a")
    g.add_edge("a", "b")
    with pytest.raises(EdgeDuplicatedException):
        g.add_edge("a", "b")
    assert g.num_edges() == 1
