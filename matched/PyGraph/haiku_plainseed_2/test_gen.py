import pytest

from py_graph_t.Graph import Graph
from py_graph_t.SimpleGraph import SimpleGraph
from py_graph_t.exceptions.SimpleGraphException import (
    VertexNotExistsException,
    EdgeDuplicatedException,
    LoopDetectedException,
    EdgeNotFoundException,
    VertexDuplicatedException,
    EdgeNameExistsException,
)


@pytest.fixture
def triangle():
    """Simple graph with vertices a, b, c and edges a-b, b-c, a-c."""
    graph = SimpleGraph()
    for value in ("a", "b", "c"):
        graph.add_vertex(value)
    graph.add_edge("a", "b", "ab")
    graph.add_edge("b", "c", "bc")
    graph.add_edge("a", "c", "ac")
    return graph


@pytest.fixture
def path_graph():
    """Simple graph a-b-c (no cycle)."""
    graph = SimpleGraph()
    for value in ("a", "b", "c"):
        graph.add_vertex(value)
    graph.add_edge("a", "b", "ab")
    graph.add_edge("b", "c", "bc")
    return graph


def test_new_graph_is_empty():
    graph = SimpleGraph()
    assert graph.num_vertex() == 0
    assert graph.num_edges() == 0
    assert graph.list_graph_vertices() == []
    assert graph.list_graph_edges() == []


def test_add_vertex_registers_and_returns_vertex():
    graph = SimpleGraph()
    vertex = graph.add_vertex("a")
    assert graph.vertex_exists("a")
    assert graph.num_vertex() == 1
    assert vertex.get_value() == "a"


def test_add_vertex_duplicate_raises():
    graph = SimpleGraph()
    graph.add_vertex("a")
    with pytest.raises(VertexDuplicatedException):
        graph.add_vertex("a")
    assert graph.num_vertex() == 1


def test_simple_graph_add_edge_returns_named_edge(triangle):
    graph = SimpleGraph()
    graph.add_vertex("x")
    graph.add_vertex("y")
    edge = graph.add_edge("x", "y", "xy")
    assert edge.get_name() == "xy"
    assert graph.num_edges() == 1
    assert graph.edge_exists("x", "y")


def test_simple_graph_rejects_loop():
    graph = SimpleGraph()
    graph.add_vertex("a")
    with pytest.raises(LoopDetectedException):
        graph.add_edge("a", "a", "loop")
    assert graph.num_edges() == 0


def test_simple_graph_rejects_missing_vertex():
    graph = SimpleGraph()
    graph.add_vertex("a")
    with pytest.raises(VertexNotExistsException):
        graph.add_edge("a", "missing", "am")
    assert graph.num_edges() == 0


def test_simple_graph_rejects_parallel_edge(triangle):
    with pytest.raises(EdgeDuplicatedException):
        triangle.add_edge("a", "b", "another")
    assert triangle.num_edges() == 3


def test_base_graph_allows_loop_and_has_loop_detects_it():
    graph = Graph()
    graph.add_vertex("a")
    graph.add_edge("a", "a", "loop")
    assert graph.has_loop() is True


def test_base_graph_has_no_loop_when_absent(triangle):
    assert triangle.has_loop() is False


def test_base_graph_add_edge_duplicate_name_raises():
    graph = Graph()
    graph.add_vertex("a")
    graph.add_vertex("b")
    graph.add_vertex("c")
    graph.add_edge("a", "b", "same")
    with pytest.raises(EdgeNameExistsException):
        graph.add_edge("b", "c", "same")


def test_base_graph_add_edge_missing_vertex_raises():
    graph = Graph()
    graph.add_vertex("a")
    with pytest.raises(VertexNotExistsException):
        graph.add_edge("a", "zzz", "az")


def test_show_edge_returns_existing_edge(triangle):
    edge = triangle.show_edge("a", "b")
    assert edge.get_name() == "ab"


def test_show_edge_missing_edge_raises(path_graph):
    with pytest.raises(EdgeNotFoundException):
        path_graph.show_edge("a", "c")


def test_show_edge_missing_vertex_raises(triangle):
    with pytest.raises(VertexNotExistsException):
        triangle.show_edge("a", "nope")


def test_edge_exists_and_edge_name_exists(triangle):
    assert triangle.edge_exists("a", "b") is True
    assert triangle.edge_exists("b", "a") is True
    assert triangle.edge_name_exists("bc") is True
    assert triangle.edge_name_exists("unknown") is False


def test_delete_edge_removes_and_returns_edge(triangle):
    removed = triangle.delete_edge("a", "b")
    assert removed.get_name() == "ab"
    assert triangle.edge_exists("a", "b") is False
    assert triangle.num_edges() == 2


def test_delete_edge_missing_raises(path_graph):
    with pytest.raises(EdgeNotFoundException):
        path_graph.delete_edge("a", "c")
    assert path_graph.num_edges() == 2


def test_delete_vertex_removes_incident_edges(triangle):
    removed = triangle.delete_vertex("a")
    assert removed.get_value() == "a"
    assert triangle.vertex_exists("a") is False
    assert triangle.num_vertex() == 2
    # Only edge b-c should remain.
    assert triangle.list_graph_edges() == ["bc"]


def test_delete_vertex_missing_raises(triangle):
    with pytest.raises(VertexNotExistsException):
        triangle.delete_vertex("missing")
    assert triangle.num_vertex() == 3


def test_vertex_neighbors_and_degree_in_triangle(triangle):
    neighbor_values = sorted(v.get_value() for v in triangle.vertex_neighbors("a"))
    assert neighbor_values == ["b", "c"]
    assert triangle.vertex_degree("a") == 2


def test_vertex_neighbors_missing_vertex_raises(triangle):
    with pytest.raises(VertexNotExistsException):
        triangle.vertex_neighbors("missing")
    with pytest.raises(VertexNotExistsException):
        triangle.vertex_degree("missing")


def test_is_vertices_adjacents(path_graph):
    assert path_graph.is_vertices_adjacents("a", "b") is True
    assert path_graph.is_vertices_adjacents("a", "c") is False


def test_has_cycle_true_for_triangle(triangle):
    assert triangle.has_cycle() is True


def test_has_cycle_false_for_path(path_graph):
    assert path_graph.has_cycle() is False


def test_list_graph_vertices_and_edges(triangle):
    assert sorted(triangle.list_graph_vertices()) == ["a", "b", "c"]
    assert sorted(triangle.list_graph_edges()) == ["ab", "ac", "bc"]


def test_adjacency_matrix_is_symmetric(path_graph):
    matrix = path_graph.adjacency_matrix()
    assert matrix["a"]["b"] == 1
    assert matrix["b"]["a"] == 1
    assert matrix["b"]["c"] == 1
    assert matrix["c"]["b"] == 1
    assert matrix["a"]["c"] == 0
    assert matrix["a"]["a"] == 0


def test_incidence_list_has_one_entry_per_vertex_edge_pair(triangle):
    entries = triangle.incidence_list()
    assert len(entries) == triangle.num_vertex() * triangle.num_edges()


def test_str_is_empty_for_empty_graph_and_one_line_per_edge(triangle):
    assert str(SimpleGraph()) == ""
    text = str(triangle)
    assert text.count("\n") == triangle.num_edges()
