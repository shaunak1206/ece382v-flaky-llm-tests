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


def build_simple(values, edges):
    graph = SimpleGraph()
    for v in values:
        graph.add_vertex(v)
    for i, (a, b) in enumerate(edges):
        graph.add_edge(a, b, "e%d" % i)
    return graph


def test_simple_graph_is_graph_and_starts_empty():
    graph = SimpleGraph()
    assert isinstance(graph, Graph)
    assert graph.num_vertex() == 0
    assert graph.num_edges() == 0
    assert str(graph) == ""


def test_add_vertex_and_exists():
    graph = Graph()
    vertex = graph.add_vertex("a")
    assert vertex.value == "a"
    assert graph.vertex_exists("a")
    assert not graph.vertex_exists("b")
    assert graph.num_vertex() == 1
    assert graph.get_all_vertex()["a"] is vertex


def test_add_duplicated_vertex_raises():
    graph = Graph()
    graph.add_vertex(1)
    with pytest.raises(VertexDuplicatedException):
        graph.add_vertex(1)
    assert graph.num_vertex() == 1


def test_delete_vertex_removes_incident_edges():
    graph = build_simple(["a", "b", "c"], [("a", "b"), ("b", "c")])
    removed = graph.delete_vertex("b")
    assert removed.value == "b"
    assert not graph.vertex_exists("b")
    assert graph.num_vertex() == 2
    assert graph.num_edges() == 0


def test_delete_vertex_keeps_unrelated_edges():
    graph = build_simple(["a", "b", "c", "d"], [("a", "b"), ("c", "d")])
    graph.delete_vertex("a")
    assert graph.num_edges() == 1
    assert graph.edge_exists("c", "d")


def test_delete_missing_vertex_raises():
    graph = Graph()
    with pytest.raises(VertexNotExistsException):
        graph.delete_vertex("x")


def test_graph_add_edge_and_show_edge():
    graph = Graph()
    graph.add_vertex("a")
    graph.add_vertex("b")
    edge = graph.add_edge("a", "b", "e1")
    assert graph.num_edges() == 1
    assert edge.get_name() == "e1"
    assert graph.show_edge("a", "b") is edge
    assert graph.edge_exists("a", "b")


def test_graph_add_edge_errors():
    graph = Graph()
    graph.add_vertex("a")
    graph.add_vertex("b")
    graph.add_vertex("c")
    with pytest.raises(VertexNotExistsException):
        graph.add_edge("a", "z", "e1")
    graph.add_edge("a", "b", "e1")
    with pytest.raises(EdgeNameExistsException):
        graph.add_edge("b", "c", "e1")
    assert graph.num_edges() == 1


def test_show_edge_errors():
    graph = Graph()
    graph.add_vertex("a")
    graph.add_vertex("b")
    with pytest.raises(EdgeNotFoundException):
        graph.show_edge("a", "b")
    with pytest.raises(VertexNotExistsException):
        graph.show_edge("a", "missing")


def test_delete_edge():
    graph = build_simple(["a", "b", "c"], [("a", "b"), ("b", "c")])
    removed = graph.delete_edge("a", "b")
    assert removed.get_name() == "e0"
    assert graph.num_edges() == 1
    assert not graph.edge_exists("a", "b")
    with pytest.raises(EdgeNotFoundException):
        graph.delete_edge("a", "b")


def test_simple_graph_rejects_loop():
    graph = SimpleGraph()
    graph.add_vertex("a")
    with pytest.raises(LoopDetectedException):
        graph.add_edge("a", "a", "l")
    assert graph.num_edges() == 0


def test_simple_graph_rejects_missing_vertex_and_duplicate_edge():
    graph = SimpleGraph()
    graph.add_vertex("a")
    graph.add_vertex("b")
    with pytest.raises(VertexNotExistsException):
        graph.add_edge("a", "zzz", "e")
    edge = graph.add_edge("a", "b", "e1")
    assert edge.get_name() == "e1"
    with pytest.raises(EdgeDuplicatedException):
        graph.add_edge("a", "b", "e2")
    assert graph.num_edges() == 1


def test_neighbors_and_degree():
    graph = build_simple(["a", "b", "c", "d"], [("a", "b"), ("a", "c")])
    neighbors = sorted(v.value for v in graph.vertex_neighbors("a"))
    assert neighbors == ["b", "c"]
    assert graph.vertex_degree("a") == 2
    assert graph.vertex_degree("b") == 1
    assert graph.vertex_degree("d") == 0
    with pytest.raises(VertexNotExistsException):
        graph.vertex_neighbors("missing")
    with pytest.raises(VertexNotExistsException):
        graph.vertex_degree("missing")


def test_is_vertices_adjacents():
    graph = build_simple(["a", "b", "c"], [("a", "b")])
    assert graph.is_vertices_adjacents("a", "b") is True
    assert graph.is_vertices_adjacents("b", "a") is True
    assert graph.is_vertices_adjacents("a", "c") is False


def test_list_vertices_and_edges():
    graph = build_simple(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert sorted(graph.list_graph_vertices()) == ["a", "b", "c"]
    assert graph.list_graph_edges() == ["e0", "e1"]


def test_has_cycle_triangle_and_path():
    triangle = build_simple(
        ["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")]
    )
    assert triangle.has_cycle() is True

    path = build_simple(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert path.has_cycle() is False

    assert SimpleGraph().has_cycle() is False


def test_has_loop():
    graph = Graph()
    graph.add_vertex("a")
    graph.add_vertex("b")
    graph.add_edge("a", "b", "e1")
    assert graph.has_loop() is False
    graph.add_edge("a", "a", "loop")
    assert graph.has_loop() is True


def test_check_regular_graph():
    triangle = build_simple(
        ["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")]
    )
    assert triangle.check_regular_graph() is True

    path = build_simple(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert path.check_regular_graph() is False

    assert SimpleGraph().check_regular_graph() is True


def test_adjacency_matrix():
    graph = build_simple(["a", "b", "c"], [("a", "b")])
    matrix = graph.adjacency_matrix()
    assert matrix["a"]["b"] == 1
    assert matrix["b"]["a"] == 1
    assert matrix["a"]["a"] == 0
    assert matrix["a"]["c"] == 0
    assert matrix["c"]["c"] == 0
    assert set(matrix.keys()) == {"a", "b", "c"}


def test_incidence_list_size():
    graph = build_simple(["a", "b", "c"], [("a", "b"), ("b", "c")])
    incidence = graph.incidence_list()
    assert len(incidence) == 3 * 2
    assert Graph().incidence_list() == []


def test_str_representation():
    graph = build_simple(["a", "b", "c"], [("a", "b"), ("b", "c")])
    text = str(graph)
    assert text.endswith("\n")
    assert len(text.strip()) > 0
    assert str(Graph()) == ""
