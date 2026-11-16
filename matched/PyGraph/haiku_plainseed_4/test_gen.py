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
def simple_graph():
    return SimpleGraph()


@pytest.fixture
def triangle():
    g = SimpleGraph()
    for v in ("a", "b", "c"):
        g.add_vertex(v)
    g.add_edge("a", "b", "ab")
    g.add_edge("b", "c", "bc")
    g.add_edge("c", "a", "ca")
    return g


@pytest.fixture
def path():
    g = SimpleGraph()
    for v in ("a", "b", "c"):
        g.add_vertex(v)
    g.add_edge("a", "b", "ab")
    g.add_edge("b", "c", "bc")
    return g


def test_add_vertex_increases_vertex_count(simple_graph):
    simple_graph.add_vertex("a")
    simple_graph.add_vertex("b")
    assert simple_graph.num_vertex() == 2


def test_add_vertex_returns_vertex_with_value(simple_graph):
    vertex = simple_graph.add_vertex("a")
    assert vertex.get_value() == "a"
    assert simple_graph.vertex_exists("a")


def test_add_duplicate_vertex_raises(simple_graph):
    simple_graph.add_vertex("a")
    with pytest.raises(VertexDuplicatedException):
        simple_graph.add_vertex("a")


def test_add_edge_adds_edge_and_returns_it(simple_graph):
    simple_graph.add_vertex("a")
    simple_graph.add_vertex("b")
    edge = simple_graph.add_edge("a", "b", "e1")
    assert simple_graph.num_edges() == 1
    assert edge.get_name() == "e1"
    assert simple_graph.edge_exists("a", "b")


def test_add_edge_with_missing_vertex_raises(simple_graph):
    simple_graph.add_vertex("a")
    with pytest.raises(VertexNotExistsException):
        simple_graph.add_edge("a", "z", "e1")


def test_simple_graph_rejects_loop(simple_graph):
    simple_graph.add_vertex("a")
    with pytest.raises(LoopDetectedException):
        simple_graph.add_edge("a", "a", "loop")
    assert simple_graph.num_edges() == 0


def test_simple_graph_rejects_parallel_edge(simple_graph):
    simple_graph.add_vertex("a")
    simple_graph.add_vertex("b")
    simple_graph.add_edge("a", "b", "e1")
    with pytest.raises(EdgeDuplicatedException):
        simple_graph.add_edge("a", "b", "e2")
    assert simple_graph.num_edges() == 1


def test_base_graph_add_edge_rejects_duplicate_name(simple_graph):
    g = Graph()
    g.add_vertex("a")
    g.add_vertex("b")
    g.add_vertex("c")
    g.add_edge("a", "b", "same")
    with pytest.raises(EdgeNameExistsException):
        g.add_edge("b", "c", "same")


def test_base_graph_allows_loop_and_has_loop_detects_it():
    g = Graph()
    g.add_vertex("a")
    g.add_edge("a", "a", "loop")
    assert g.has_loop() is True


def test_delete_vertex_removes_incident_edges(path):
    removed = path.delete_vertex("b")
    assert removed.get_value() == "b"
    assert path.num_vertex() == 2
    assert path.num_edges() == 0


def test_delete_missing_vertex_raises(simple_graph):
    with pytest.raises(VertexNotExistsException):
        simple_graph.delete_vertex("nope")


def test_delete_edge_returns_and_removes_edge(path):
    edge = path.delete_edge("a", "b")
    assert edge.get_name() == "ab"
    assert path.num_edges() == 1
    assert not path.edge_exists("a", "b")


def test_delete_missing_edge_raises(path):
    with pytest.raises(EdgeNotFoundException):
        path.delete_edge("a", "c")


def test_show_edge_missing_raises_edge_not_found(path):
    with pytest.raises(EdgeNotFoundException):
        path.show_edge("a", "c")


def test_show_edge_with_missing_vertex_raises(path):
    with pytest.raises(VertexNotExistsException):
        path.show_edge("a", "zzz")


def test_vertex_neighbors_and_degree_on_path(path):
    neighbors = path.vertex_neighbors("b")
    neighbor_values = sorted(v.get_value() for v in neighbors)
    assert neighbor_values == ["a", "c"]
    assert path.vertex_degree("b") == 2
    assert path.vertex_degree("a") == 1


def test_vertex_neighbors_missing_vertex_raises(path):
    with pytest.raises(VertexNotExistsException):
        path.vertex_neighbors("zzz")


def test_is_vertices_adjacents(path):
    assert path.is_vertices_adjacents("a", "b") is True
    assert path.is_vertices_adjacents("a", "c") is False


def test_has_cycle_true_for_triangle(triangle):
    assert triangle.has_cycle() is True


def test_has_cycle_false_for_path(path):
    assert path.has_cycle() is False


def test_check_regular_graph_triangle_is_regular(triangle):
    assert triangle.check_regular_graph() is True


def test_check_regular_graph_path_is_not_regular(path):
    assert path.check_regular_graph() is False


def test_adjacency_matrix_for_path(path):
    matrix = path.adjacency_matrix()
    assert matrix["a"]["b"] == 1
    assert matrix["b"]["a"] == 1
    assert matrix["a"]["c"] == 0
    assert matrix["b"]["c"] == 1
    assert matrix["c"]["b"] == 1


def test_list_graph_edges_and_vertices(path):
    assert sorted(path.list_graph_edges()) == ["ab", "bc"]
    assert sorted(path.list_graph_vertices()) == ["a", "b", "c"]


def test_incidence_list_has_one_entry_per_vertex_edge_pair(path):
    incidence = path.incidence_list()
    assert len(incidence) == path.num_vertex() * path.num_edges()


def test_str_empty_graph_is_empty_string(simple_graph):
    assert str(simple_graph) == ""


def test_edge_name_exists(path):
    assert path.edge_name_exists("ab") is True
    assert path.edge_name_exists("missing") is False
